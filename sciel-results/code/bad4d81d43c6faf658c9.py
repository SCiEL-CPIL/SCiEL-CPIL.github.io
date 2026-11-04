"""Program for rank-minimal tensor decomposition of 2x5 * 5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _build_schoolbook(n, m, p):
    """Build the schoolbook (trivial) decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _tensor_residual(U, V, W, n, m, p):
    """
    Compute the residual of the tensor equation.
    T[i,j,k] should be 1.
    Residual is sum of squared differences.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # We need to check T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is a trilinear form.
    # We can vectorize this by iterating over i, j, k.
    # For speed, we can pre-compute some contractions.
    # But for small n, m, p, direct loop is fine.
    
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                # U[i*m+j, :] is a vector of length rank
                # V[j*p+k, :] is a vector of length rank
                # W[k*n+i, :] is a vector of length rank
                # s = dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                s = np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i])
                residual += (s - 1.0) ** 2
    return residual

def _try_reduce_rank(n, m, p, U, V, W, target_rank, time_limit):
    """
    Attempt to find a decomposition of rank target_rank by removing 
    (current_rank - target_rank) columns and solving for the remaining ones.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    # We want to remove R = current_rank - target_rank columns.
    # We try to set R columns to zero and solve for the rest.
    # To make it tractable, we try to remove one column at a time, 
    # or a small set.
    # For now, try to remove 1 column.
    
    if current_rank == target_rank:
        return U, V, W, True
    
    # If we need to remove more than 1 column, this simple approach might fail.
    # But let's try to remove 1 column first.
    # If we can't remove 1, we might not be able to remove more.
    
    # We will try to remove each column index in turn.
    # For each column c, we set U[:,c]=0, V[:,c]=0, W[:,c]=0.
    # Then we have a system of equations for the remaining columns.
    # The equations are:
    # sum_{c' != c} U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] = 1
    # This is still trilinear in the remaining variables.
    # However, if we fix two of the matrices, it becomes linear in the third.
    
    # Strategy:
    # 1. Fix V and W to their current values (except the column to be removed).
    # 2. Solve for U.
    # 3. Check if the solution for U satisfies the tensor equation.
    # 4. If yes, we have a valid decomposition of rank current_rank - 1.
    
    # This is a linear system for U:
    # For each (i, j, k):
    # sum_{c' != c} U[i*m+j, c'] * (V[j*p+k, c'] * W[k*n+i, c']) = 1
    # Let A_{(i,j,k), (i', j', c')} ... this is not a standard linear system in U
    # because U appears multiplied by V and W which depend on j and k.
    # Specifically, U[i*m+j, c'] appears in the equation for (i, j, k) with coefficient
    # V[j*p+k, c'] * W[k*n+i, c'].
    # So for a fixed c', U[i*m+j, c'] is a variable.
    # The equation for (i, j, k) is:
    # sum_{c' != c} U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] = 1
    # This is linear in the variables U[*, c'].
    # We have n*m*p = 50 equations.
    # We have (current_rank - 1) * (n*m) variables? No, U has shape (n*m, rank).
    # The variables are U[i*m+j, c'] for all i, j, c' != c.
    # Total variables: (n*m) * (current_rank - 1) = 10 * 49 = 490.
    # Equations: 50.
    # This is underdetermined. We can find a solution.
    # But we must also ensure that the resulting U, V, W satisfy the equations for ALL (i,j,k).
    # Wait, the equations are exactly the tensor equations.
    # So if we solve for U such that the tensor equations are satisfied, 
    # we have a valid decomposition.
    
    # However, V and W are fixed.
    # So we solve a linear system: A * u = b
    # where u is the vector of U entries (excluding column c).
    # A has 50 rows and 490 columns.
    # We want a solution u.
    # We can use least squares to find the minimum norm solution.
    # Then we check if the residual is close to 0.
    
    Ua = U.copy()
    Va = V.copy()
    Wa = W.copy()
    
    best_U, best_V, best_W = None, None, None
    best_res = float('inf')
    
    for c_to_remove in range(current_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Construct the linear system for U
        # Variables: U[i*m+j, c'] for c' != c_to_remove
        # We map each variable to an index in a flat vector.
        # num_vars = (n*m) * (current_rank - 1)
        
        # Build A matrix (50 x num_vars)
        # A[row, var_idx] = coefficient of U[var] in equation row
        
        # Equation row index: idx = i*p*m + j*p + k? No, i*m + j for U, j*p+k for V, k*n+i for W.
        # Let's index equations by (i, j, k).
        
        A = np.zeros((n * m * p, (n * m) * (current_rank - 1)))
        b = np.ones(n * m * p)
        
        # Fill A
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Equation: sum_{c' != c} U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] = 1
                    var_idx = 0
                    for c_prime in range(current_rank):
                        if c_prime == c_to_remove:
                            continue
                        # Variable is U[i*m+j, c_prime]
                        # Its index in the flat vector:
                        # The flat vector is ordered by (i, j) then c_prime?
                        # Let's say we flatten U by columns first?
                        # U is (n*m, rank).
                        # Let's keep U as is and just solve for the specific row i*m+j.
                        # Actually, U[i*m+j, c'] only appears in equations where the row index is i*m+j.
                        # So for a fixed (i, j), the variables are U[i*m+j, c'] for c' != c.
                        # There are (current_rank - 1) such variables.
                        # They appear in p equations (for k=0..p-1).
                        
                        # This suggests we can decouple the system by (i, j)?
                        # No, because V and W depend on j and k, and k is summed over.
                        # But U[i*m+j, c'] is only in equations for that specific i, j.
                        # So the system is block-diagonal with respect to (i, j)!
                        # For each (i, j), we have a subsystem of p equations 
                        # and (current_rank - 1) variables (U[i*m+j, c']).
                        # The equations are:
                        # sum_{c' != c} U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] = 1 for k=0..p-1
                        # This is a linear system for the vector u_ij = [U[i*m+j, c'] for c' != c].
                        # Matrix B_ij has shape (p, current_rank - 1).
                        # B_ij[k, c'_idx] = V[j*p+k, c'] * W[k*n+i, c']
                        
                        # We can solve this small system for each (i, j).
                        
                        # Let's build B_ij.
                        # p = 5, current_rank - 1 = 49.
                        # B_ij is 5x49.
                        # We want to solve B_ij * u_ij = 1 (vector of ones, length 5).
                        # This is underdetermined. We can find a minimum norm solution.
                        # Then we check if the residual is 0.
                        
                        # If we can solve this for all (i, j), we have a valid U.
                        # And since V and W are fixed, the tensor equation is satisfied.
                        
                        # So the problem reduces to:
                        # For each (i, j), solve B_ij * u_ij = 1.
                        # If a solution exists (residual 0), we are good.
                        
                        # But wait, we also need to ensure that the resulting U, V, W 
                        # form a valid decomposition. 
                        # The tensor equation is exactly what we are solving.
                        # So if we find U such that the equations hold, it is valid.
                        
                        # The only constraint is that U, V, W must be consistent.
                        # Since we fixed V and W, and solved for U, it is consistent.
                        
                        # So we just need to check if for each (i, j), 
                        # the system B_ij * u_ij = 1 has a solution.
                        # This is true if and only if the vector of ones is in the 
                        # column space of B_ij.
                        
                        # We can check this by computing the residual of the 
                        # least squares solution.
                        
                        # Let's do this.
                        
                        pass # We'll implement this in the loop below.
                    eq_idx += 1
        
        # Instead of building the big A, let's solve per (i, j).
        
        valid = True
        U_new = Ua.copy()
        
        for i in range(n):
            for j in range(m):
                # Build B_ij (p x (current_rank - 1))
                # Columns correspond to c' != c_to_remove
                c_primes = [c for c in range(current_rank) if c != c_to_remove]
                B_ij = np.zeros((p, len(c_primes)))
                for idx_c, c_prime in enumerate(c_primes):
                    for k in range(p):
                        B_ij[k, idx_c] = Va[j * p + k, c_prime] * Wa[k * n + i, c_prime]
                
                # Solve B_ij * u_ij = 1
                # Use least squares
                u_ij, residuals, rank, sv = np.linalg.lstsq(B_ij, np.ones(p), rcond=None)
                
                # Check residual
                if len(residuals) > 0:
                    if residuals[0] > 1e-9:
                        valid = False
                        break
                else:
                    # If rank < p, check if 1 is in the column space
                    # lstsq returns the minimum norm solution.
                    # We should check the residual explicitly.
                    res = np.linalg.norm(B_ij @ u_ij - np.ones(p))
                    if res > 1e-9:
                        valid = False
                        break
                
                # Assign solution to U_new
                for idx_c, c_prime in enumerate(c_primes):
                    U_new[i * m + j, c_prime] = u_ij[idx_c]
                U_new[i * m + j, c_to_remove] = 0.0
            
            if not valid:
                break
        
        if valid:
            # We found a valid U for this column removal.
            # V and W are unchanged (except column c_to_remove is 0).
            V_new = Va.copy()
            W_new = Wa.copy()
            V_new[:, c_to_remove] = 0.0
            W_new[:, c_to_remove] = 0.0
            
            # Verify the full decomposition
            # (Optional, but good for safety)
            res = _tensor_residual(U_new, V_new, W_new, n, m, p)
            if res < 1e-9:
                return U_new, V_new, W_new, True
        
    return U, V, W, False

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 5 matrix by a 5 by 5 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    """
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    max_time = 140.0
    
    # Build schoolbook decomposition
    U, V, W = _build_schoolbook(n, m, p)
    
    # Try to reduce rank by 1
    # We have 50 columns. Try to remove one.
    # If successful, rank is 49.
    # We can try to remove more, but let's start with 1.
    
    U_curr, V_curr, W_curr = U, V, W
    rank_curr = U_curr.shape[1]
    
    # Try to reduce rank
    # We will try to reduce rank as much as possible, one column at a time.
    # But each step is expensive (50 * 10 * 5x49 lstsq).
    # 500 * 5x49 lstsq is fast.
    
    for target_rank in range(rank_curr - 1, 0, -1):
        if time.time() - start_time > max_time:
            break
        
        # Try to remove one column
        U_new, V_new, W_new, success = _try_reduce_rank(n, m, p, U_curr, V_curr, W_curr, target_rank, max_time - (time.time() - start_time))
        
        if success:
            U_curr, V_curr, W_curr = U_new, V_new, W_new
            # Continue trying to reduce further
        else:
            # If we can't reduce by 1, we likely can't reduce by more.
            # But it's possible that removing a different column works.
            # _try_reduce_rank tries all columns.
            # So if it fails, no single column removal works.
            break
    
    # Return the best found
    return {"U": U_curr, "V": V_curr, "W": W_curr}

# EVOLVE-BLOCK-END
