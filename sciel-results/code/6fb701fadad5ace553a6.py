"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition
    # Strategy: Exploit the structure by fixing W patterns and solving linear systems
    # For each rank r, try various structured W matrices and solve for U, V
    
    # The constraint: for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Let's try ranks from 28 upward (heuristic: at least n*m + m*p + p*n - something)
    # Actually, let's be smarter. The tensor has rank at most 42.
    # Known lower bounds suggest rank >= 21 for 2x3x7 (roughly n*m*p/2)
    
    # Approach: Use the fact that for fixed k, the sub-tensor over (i,j) is rank at most min(n,m)
    # This suggests we can group terms by k
    
    # Let me try a direct algebraic approach:
    # For each (i,j,k), we need sum_c U_ij,c * V_jk,c * W_ki,c = 1
    # 
    # Key insight: If we fix W, then for each k, we have a system in U and V.
    # Let's try W with a specific structure: W[k*n+i, c] depends only on k and i in a simple way.
    
    # Try rank r decompositions for r = 28, 29, ..., 41
    best_result = None
    best_rank = rank_sb
    
    for r in range(28, rank_sb):  # Try ranks 28 to 41
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations of W
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random W with rational entries (use small integers for simplicity)
            np.random.seed(r * 1000 + trial)
            W = np.random.randint(-2, 3, size=(p * n, r))
            
            # For each column c, W gives weights for each (k,i) pair
            # We need to find U and V such that the tensor equation holds
            
            # Build the linear system:
            # For each (i,j,k): sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
            #
            # This is bilinear in U and V. Let's fix V and solve for U, then iterate.
            # Or: fix W and V, solve for U as a linear system.
            
            # Actually, let's use a different approach:
            # Fix W and V, then U is determined by a linear system.
            # Then check if the result is consistent.
            
            # Initialize V randomly
            V = np.random.randint(-2, 3, size=(m * p, r))
            
            # For each (i,j), we have: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
            # This is a system of p equations in r unknowns (U[i*m+j, 0..r-1])
            # If r >= p, we can solve this.
            
            U = np.zeros((n * m, r))
            valid = True
            
            for i in range(n):
                for j in range(m):
                    # Build system: A * u = b where u = U[i*m+j, :]
                    # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    # b[k] = 1
                    A = np.zeros((p, r))
                    for k in range(p):
                        for c in range(r):
                            A[k, c] = V[j * p + k, c] * W[k * n + i, c]
                    
                    # Solve A * u = ones(p)
                    # Use least squares if overdetermined, or exact if square
                    try:
                        if r >= p:
                            u, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            # Check if solution is exact
                            if np.allclose(A @ u, np.ones(p), atol=1e-10):
                                U[i * m + j, :] = u
                            else:
                                valid = False
                                break
                        else:
                            # Underdetermined, use lstsq
                            u, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            if np.allclose(A @ u, np.ones(p), atol=1e-10):
                                U[i * m + j, :] = u
                            else:
                                valid = False
                                break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition
            is_valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if abs(s - 1.0) > 1e-9:
                            is_valid = False
                            break
                    if not is_valid:
                        break
                if not is_valid:
                    break
            
            if is_valid:
                # Found a valid decomposition of rank r
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                # Try to convert to exact fractions if possible
                try:
                    U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_list]
                    V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_list]
                    W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_list]
                    
                    # Verify with fractions
                    frac_valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = Fraction(0)
                                for c in range(r):
                                    s += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                                if s != 1:
                                    frac_valid = False
                                    break
                            if not frac_valid:
                                break
                        if not frac_valid:
                            break
                    
                    if frac_valid:
                        U_list = [[float(x) if x.denominator == 1 else x.numerator / x.denominator for x in row] for row in U_frac]
                        V_list = [[float(x) if x.denominator == 1 else x.numerator / x.denominator for x in row] for row in V_frac]
                        W_list = [[float(x) if x.denominator == 1 else x.numerator / x.denominator for x in row] for row in W_frac]
                except:
                    pass
                
                if best_result is None or r < best_rank:
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                
                # If we found rank 28, that's likely optimal, stop early
                if r == 28:
                    break
        
        if best_rank == 28:
            break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
