import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s limit to be safe within 150s
    
    # Schoolbook decomposition (Rank 105)
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
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Search for lower rank using random sampling
    # Theoretical lower bound
    lower_bound = max(n * m, m * p, p * n)
    
    # Build the tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor equation: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Strategy: For a given rank R, sample U and V with small entries,
    # then for each (i,j,k), we need W[k*n+i, r] such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a system of linear equations in W entries.
    
    # Actually, let's think differently. The tensor has 3*5*7 = 105 entries.
    # For rank R, we have R * (n*m + m*p + p*n) = R * (15 + 35 + 21) = 71R unknowns.
    # We need 71R >= 105, so R >= 2. But the constraints are nonlinear.
    
    # Let's try a different approach: fix W to have simple structure and solve for U, V.
    # Or use the fact that for small matrices, we can enumerate some structures.
    
    # For 3x5x7, let's try rank values from lower_bound upward
    # and do random searches with small integer/half-integer entries.
    
    np.random.seed(42)
    
    # Try ranks from lower bound upward
    for R in range(lower_bound, best_rank):
        if time.time() > deadline - 15:
            break
        
        # For each rank, try several random initializations
        max_iters = 50 if R < 50 else 20
        found = False
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Generate random U and V with small integer/half-integer entries
            # Use values from {-1, -0.5, 0, 0.5, 1}
            entries = np.array([-1, -0.5, 0, 0.5, 1])
            
            U_rand = np.random.choice(entries, size=(n * m, R))
            V_rand = np.random.choice(entries, size=(m * p, R))
            
            # For each (i, k), we need to solve for W[k*n+i, :] such that:
            # For all j in range(m): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a system of m equations in R unknowns (the W row for (k,i))
            
            # Build the system: A[i,k,:] * W[k*n+i, :] = 1 for all j
            # where A[i,k,j, r] = U[i*m+j, r] * V[j*p+k, r]
            
            W_sol = np.zeros((p * n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build m x R matrix
                    A_mat = np.zeros((m, R))
                    for j in range(m):
                        A_mat[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                    
                    # Solve A_mat * W_row = ones(m)
                    # This is an underdetermined or overdetermined system
                    try:
                        # Use least squares
                        W_row, residuals, rank, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=None)
                        
                        # Check if solution is close to integer/half-integer
                        # Round to nearest half-integer
                        W_row_rounded = np.round(W_row * 2) / 2
                        
                        # Check if rounded solution satisfies the equations
                        residual = np.max(np.abs(A_mat @ W_row_rounded - np.ones(m)))
                        
                        if residual < 1e-9:
                            W_sol[k * n + i, :] = W_row_rounded
                        else:
                            # Try the unrounded solution check
                            residual_orig = np.max(np.abs(A_mat @ W_row - np.ones(m)))
                            if residual_orig < 1e-9:
                                # Check if we can round
                                W_sol[k * n + i, :] = W_row_rounded
                                # Re-verify
                                residual2 = np.max(np.abs(A_mat @ W_sol[k * n + i, :] - np.ones(m)))
                                if residual2 > 1e-9:
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                    except:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Verify the full solution
            # Check all 105 tensor entries
            all_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_rand[i * m + j, :] * V_rand[j * p + k, :], W_sol[k * n + i, :])
                        if abs(val - 1) > 1e-9:
                            all_ok = False
                            break
                    if not all_ok:
                        break
                if not all_ok:
                    break
            
            if all_ok:
                # Found a valid solution!
                best_rank = R
                best_solution = {
                    "U": U_rand.tolist(),
                    "V": V_rand.tolist(),
                    "W": W_sol.tolist()
                }
                found = True
                break
        
        if found:
            break
    
    return best_solution
# EVOLVE-BLOCK-END
