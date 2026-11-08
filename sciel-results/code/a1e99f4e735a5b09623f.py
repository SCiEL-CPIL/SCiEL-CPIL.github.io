import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 56
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the constraint matrix structure
    # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # For fixed U, W: linear in V
    
    def try_rank(r, max_attempts=50):
        """Try to find a rank-r decomposition by fixing U,W and solving for V."""
        nm = n * m  # 8
        mp = m * p  # 28
        pn = p * n  # 14
        
        # Build constraint matrix A where A[eq_idx, var_idx]
        # eq_idx corresponds to (i, j, k)
        # var_idx corresponds to (j, k, c)
        # A[(i,j,k), (j',k',c)] = U[i*m+j, c] * W[k'*n+i, c] if j==j', k==k' else 0
        # Wait, V[j*p+k, c] - the indices are (j,k) for V
        # So for equation (i,j,k), variable (j',k',c):
        # A = U[i*m+j, c] * W[k*n+i, c] if (j,k) == (j',k') else 0
        # This means A is block-diagonal in (j,k) pairs
        
        # Actually let me think more carefully.
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # For fixed (i,j,k), this is: sum_c [U[i*m+j,c] * W[k*n+i,c]] * V[j*p+k, c] = 1
        # So the variable is V[j*p+k, c] for c=0..r-1
        # The coefficient for V[j*p+k, c] is U[i*m+j, c] * W[k*n+i, c]
        
        # Number of equations: n*m*p = 56
        # Number of variables: m*p*r = 28*r
        # For r >= 2, we have more variables than equations (28r >= 56)
        
        # Build the system: for each (i,j,k), we have equation:
        # sum_{c=0}^{r-1} U[i*m+j, c] * W[k*n+i, c] * V[j*p+k, c] = 1
        # The variable V[j*p+k, c] only appears in equations where the (j,k) pair matches.
        # So for each (j,k) pair, we have n equations (one for each i) and r variables (V[j*p+k, 0..r-1])
        # Coefficient matrix for (j,k): B[i, c] = U[i*m+j, c] * W[k*n+i, c]
        
        # This is a system of n=2 equations in r unknowns per (j,k) block.
        # We can solve each block independently!
        
        # Strategy: choose U and W with half-integer entries, then for each (j,k) block,
        # solve B * v = 1 where B is 2x r, v is the column of V for that (j,k).
        
        # For the system to have a solution, we need rank(B) = 2 (full row rank) or 
        # if rank < 2, the RHS must be in the column space.
        
        # Let's try small r values. For r=1: each block has 1 variable, 2 equations.
        # B[i,0] * V = 1 for both i, so B[0,0] = B[1,0]. This is very constrained.
        
        # For r=2: each block has 2 variables, 2 equations. If B is invertible, unique solution.
        
        # Let's try r from 3 up to 55.
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize U and W with small half-integer values
            # Use random values from {-1, -1/2, 0, 1/2, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            
            if attempt < 10:
                # Try structured initializations
                if attempt == 0:
                    # All ones (like schoolbook but with fewer columns)
                    U = np.ones((nm, r), dtype=float)
                    W = np.ones((pn, r), dtype=float)
                elif attempt == 1:
                    U = np.array([[1 if (i+c) % 2 == 0 else -1 for c in range(r)] for i in range(nm)], dtype=float)
                    W = np.array([[1 if (i+c) % 2 == 0 else -1 for c in range(r)] for i in range(pn)], dtype=float)
                else:
                    U = np.random.choice(vals, size=(nm, r))
                    W = np.random.choice(vals, size=(pn, r))
            else:
                # Random half-integer initialization
                U = np.random.choice(vals, size=(nm, r))
                W = np.random.choice(vals, size=(pn, r))
            
            # For each (j,k) block, solve for V[j*p+k, :]
            V = np.zeros((mp, r), dtype=float)
            valid = True
            
            for j in range(m):
                for k in range(p):
                    # B[i, c] = U[i*m+j, c] * W[k*n+i, c] for i=0..n-1, c=0..r-1
                    B = np.array([
                        [U[i*m+j, c] * W[k*n+i, c] for c in range(r)]
                        for i in range(n)
                    ], dtype=float)
                    
                    rhs = np.ones(n)
                    
                    # Solve B * v = rhs
                    # If B has full row rank (rank n=2), there may be solutions
                    # Use least squares or check consistency
                    try:
                        # Check if system is consistent
                        if r <= n:
                            # Square or underdetermined
                            if r == n:
                                if abs(np.linalg.det(B)) < 1e-12:
                                    # Singular, try least squares
                                    v, residuals, rank_B, sv = np.linalg.lstsq(B, rhs, rcond=None)
                                    if rank_B < n:
                                        valid = False
                                        break
                                    # Verify
                                    if np.linalg.norm(B @ v - rhs) > 1e-9:
                                        valid = False
                                        break
                                else:
                                    v = np.linalg.solve(B, rhs)
                            else:  # r < n, underdetermined
                                v, residuals, rank_B, sv = np.linalg.lstsq(B, rhs, rcond=None)
                                if rank_B < n:
                                    valid = False
                                    break
                                if np.linalg.norm(B @ v - rhs) > 1e-9:
                                    valid = False
                                    break
                        else:  # r > n, overdetermined in variables, underdetermined
                            # Use pseudoinverse to get minimum norm solution
                            v = np.linalg.pinv(B) @ rhs
                            if np.linalg.norm(B @ v - rhs) > 1e-9:
                                valid = False
                                break
                    
                    except:
                        valid = False
                        break
                    
                    V[j*p+k, :] = v
            
            if not valid:
                continue
            
            # Verify the full decomposition
            # Compute C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            C = np.zeros((n, p), dtype=float)
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for c in range(r):
                            C[i, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            if np.allclose(C, np.ones((n, p)), atol=1e-9):
                # Found a valid solution!
                # Convert to half-integer representation if possible
                # Check if entries are close to half-integers
                def to_half_int_matrix(M):
                    """Convert matrix to half-integer if entries are close to half-integers."""
                    M_check = np.round(M * 2) / 2
                    if np.allclose(M, M_check, atol=1e-8):
                        return M_check
                    return None
                
                U_h = to_half_int_matrix(U)
                V_h = to_half_int_matrix(V)
                W_h = to_half_int_matrix(W)
                
                if U_h is not None and V_h is not None and W_h is not None:
                    # Great, we have half-integer entries
                    return (U_h, V_h, W_h)
                else:
                    # Try to find nearby half-integer solution
                    # Round each entry to nearest half-integer and re-verify
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    C_r = np.zeros((n, p), dtype=float)
                    for i in range(n):
                        for k in range(p):
                            for j in range(m):
                                for c in range(r):
                                    C_r[i, k] += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    
                    if np.allclose(C_r, np.ones((n, p)), atol=1e-8):
                        return (U_r, V_r, W_r)
                    else:
                        # Try another approach: keep the real solution but represent as fractions
                        # For the output, we can use the real values (they should be rational)
                        return (U, V, W)
        
        return None
    
    # Try ranks from low to high
    best_result = None
    
    for r in range(3, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        result = try_rank(r, max_attempts=30)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        rank = U.shape[1]
        # Convert to list format
        U_list = [[float(U[i, j]) for j in range(rank)] for i in range(n*m)]
        V_list = [[float(V[i, j]) for j in range(rank)] for i in range(m*p)]
        W_list = [[float(W[i, j]) for j in range(rank)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
