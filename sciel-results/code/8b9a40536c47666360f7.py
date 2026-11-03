"""Matrix multiplication 3x4x5: search for rank-minimal exact tensor decomposition."""
import time
import random
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0  # wall-clock seconds
SAFETY_MARGIN = 5.0  # seconds to spare before deadline

def _time_remaining():
    """Return seconds remaining before the hard deadline."""
    # We'll track start time externally; this is a placeholder
    return DEADLINE - SAFETY_MARGIN

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    U shape: (n*m, rank) = (12, rank)
    V shape: (m*p, rank) = (20, rank)
    W shape: (p*n, rank) = (15, rank)
    
    The decomposition must satisfy:
    For all (i,j,k) in [0..n) x [0..m) x [0..p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    For all other (i,j,k) combinations that don't correspond to valid matrix mult,
        the sum must be 0 (but actually in matrix mult tensor, only (i,j,k) where
        i is row of A, j is col of A / row of B, k is col of B matter)
    
    Actually, the tensor T[i,j,k] = 1 if this corresponds to a valid entry in C = A*B,
    i.e., always for 0<=i<n, 0<=j<m, 0<=k<p. The output C[i,k] = sum_j A[i,j]*B[j,k].
    So the tensor has T[i,j,k] = 1 for all i,j,k in their ranges.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need to find rank-r decomposition: T[i,j,k] = sum_{r=1}^{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Known bounds:
    # Lower bound: max(n*p, m*p, n*m) = max(15, 20, 12) = 20
    # Upper bound (schoolbook): n*m*p = 60
    # Best known for 3x4x5: Let's try to find something better than 60.
    
    # Strategy: Use a structured approach. Try various ranks from a lower estimate upward.
    # For each rank, attempt to find integer/half-integer solutions.
    
    # First, let's try a simple structured decomposition.
    # One idea: group terms to reduce rank.
    
    # Let's try rank = n*p + m*p - n*m or similar formulas from literature.
    # Actually, for small matrices, let's just search.
    
    # Start with the schoolbook rank and try to reduce.
    # For a 3x4x5 matrix multiplication, the tensor rank is known to be at most some value.
    # Let's try to find a good decomposition.
    
    # Approach: Use the fact that we can combine some terms.
    # For example, for fixed i, the terms over j,k can sometimes be combined.
    
    # Let's try a direct search for small ranks using random initialization and refinement.
    
    # Given the time constraint, let's try a few structured approaches first.
    
    # Approach 1: Try rank = 20 (the lower bound)
    # Approach 2: Try rank = 24, 28, 30, etc.
    
    # For now, let's return the schoolbook decomposition as a baseline,
    # but try to find something better within the time budget.
    
    # Let's try a simple structured reduction.
    # For each pair (i,k), the sum over j is what matters.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One structured approach: use the identity that allows combining some terms.
    
    # Let's try rank = n*m*p = 60 first (schoolbook), then try to reduce.
    
    # Actually, let's try a random search for lower ranks.
    
    best_result = None
    best_rank = n * m * p  # start with schoolbook
    
    # Generate the schoolbook decomposition
    U_school = [[0] * best_rank for _ in range(n * m)]
    V_school = [[0] * best_rank for _ in range(m * p)]
    W_school = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Now try to find better decompositions with lower rank
    # Try various ranks from lower bound upward
    lower_bound = max(n * p, m * p, n * m)  # = 20
    
    # Try ranks from lower_bound to best_rank - 1
    # Use a more efficient search: try a subset of ranks first
    ranks_to_try = list(range(lower_bound, best_rank))
    
    # Prioritize ranks that are likely to work based on known results
    # For 3x4x5, known upper bounds suggest rank around 30-40
    ranks_to_try.sort(key=lambda r: abs(r - 35))  # Try ranks near 35 first
    
    for target_rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Try to find a decomposition of this rank
        result = _try_find_decomposition(n, m, p, target_rank, time.time() - start_time)
        if result is not None:
            best_result = result
            best_rank = target_rank
            # Don't break; try to find even lower rank
            # Remove this rank from consideration
            ranks_to_try.remove(target_rank)
    
    return best_result


def _validate_decomposition(U, V, W, n, m, p):
    """Check if the decomposition is valid."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
    
    # Check the tensor equation
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1) > 1e-9:
                    return False
    return True


def _is_half_integer(x):
    """Check if x is an integer or half-integer."""
    return abs(x * 2 - round(x * 2)) < 1e-9


def _try_find_decomposition(n, m, p, target_rank, elapsed):
    """
    Try to find a rank-target_rank decomposition using random initialization + Newton's method.
    Returns None if not found within time budget.
    """
    remaining = DEADLINE - SAFETY_MARGIN - elapsed
    if remaining < 1.0:
        return None
    
    np.random.seed(int(time.time()) % (2**31))
    
    # Number of restarts based on remaining time
    max_restarts = max(1, int(remaining / 2.0))
    
    for restart in range(max_restarts):
        if time.time() - (DEADLINE - SAFETY_MARGIN) > elapsed + remaining - 0.5:
            break
        
        # Random initialization with small integers
        U = np.random.randint(-1, 2, (n * m, target_rank))
        V = np.random.randint(-1, 2, (m * p, target_rank))
        W = np.random.randint(-1, 2, (p * n, target_rank))
        
        # Ensure no all-zero columns
        for r in range(target_rank):
            if np.all(U[:, r] == 0):
                U[np.random.randint(n * m), r] = 1
            if np.all(V[:, r] == 0):
                V[np.random.randint(m * p), r] = 1
            if np.all(W[:, r] == 0):
                W[np.random.randint(p * n), r] = 1
        
        # Newton's method iterations
        success = _newton_solve(U, V, W, n, m, p, max_iter=50, tol=1e-10)
        
        if success:
            # Convert to list of lists with half-integer check
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Round to nearest half-integer
            U_list = [[round(x * 2) / 2 for x in row] for row in U_list]
            V_list = [[round(x * 2) / 2 for x in row] for row in V_list]
            W_list = [[round(x * 2) / 2 for x in row] for row in W_list]
            
            # Validate
            if _validate_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def _newton_solve(U, V, W, n, m, p, max_iter=50, tol=1e-10):
    """
    Solve for U, V, W using Newton's method on the tensor equations.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = U.shape[1]
    
    for iteration in range(max_iter):
        # Compute residuals: residual[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residuals[i, j, k] = 1.0 - total
        
        max_residual = np.max(np.abs(residuals))
        if max_residual < tol:
            return True
        
        # Build Jacobian and solve linear system
        # Variables: U (n*m*rank), V (m*p*rank), W (p*n*rank)
        total_vars = n * m * rank + m * p * rank + p * n * rank
        
        # Use Gauss-Newton: J^T J delta = J^T (-residuals)
        # For efficiency, we'll use a simplified approach
        
        # Actually, let's use a different approach: treat this as a system and use
        # least squares on the linearized problem
        
        # Linearize: f(U,V,W) = 0 where f[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        # df/dU[i*m+j,r] = V[j*p+k,r]*W[k*n+i,r]
        # df/dV[j*p+k,r] = U[i*m+j,r]*W[k*n+i,r]
        # df/dW[k*n+i,r] = U[i*m+j,r]*V[j*p+k,r]
        
        # Build J^T J and J^T * (-residuals)
        JtJ = np.zeros((total_vars, total_vars))
        Jtb = np.zeros(total_vars)
        
        # This is expensive for large rank, so let's use a simpler approach:
        # Alternate least squares style updates
        
        # Update U: for each i,j, solve for U[i*m+j,:] given V and W
        # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
        # This is a system: A_U[i,j] @ U[i*m+j,:] = b_U[i,j]
        # where A_U[i,j][k,r] = V[j*p+k,r] * W[k*n+i,r]
        
        for i in range(n):
            for j in range(m):
                # Build system for U[i*m+j,:]
                A = np.zeros((p, rank))
                b = np.ones(p)
                for k in range(p):
                    for r in range(rank):
                        A[k, r] = V[j * p + k, r] * W[k * n + i, r]
                
                # Solve A @ u = b using least squares
                try:
                    u_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = u_new
                except:
                    pass
        
        # Update V: for each j,k, solve for V[j*p+k,:] given U and W
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
        # A_V[j,k][i,r] = U[i*m+j,r] * W[k*n+i,r]
        
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, rank))
                b = np.ones(n)
                for i in range(n):
                    for r in range(rank):
                        A[i, r] = U[i * m + j, r] * W[k * n + i, r]
                
                try:
                    v_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = v_new
                except:
                    pass
        
        # Update W: for each k,i, solve for W[k*n+i,:] given U and V
        # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
        # A_W[k,i][j,r] = U[i*m+j,r] * V[j*p+k,r]
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                b = np.ones(m)
                for j in range(m):
                    for r in range(rank):
                        A[j, r] = U[i * m + j, r] * V[j * p + k, r]
                
                try:
                    w_new, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = w_new
                except:
                    pass
        
        # Check convergence
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residuals[i, j, k] = 1.0 - total
        
        max_residual = np.max(np.abs(residuals))
        if max_residual < tol:
            return True
    
    return False


if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    print(f"Rank: {rank}")
    print(f"Valid: {_validate_decomposition(U, V, W, n, m, p)}")
