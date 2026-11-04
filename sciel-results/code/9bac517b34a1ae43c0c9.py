"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Build the target tensor T[i,j,k] = 1 for all i,j,k
    # T is n x m x p
    T = np.ones((n, m, p))
    
    # Try to find a low-rank decomposition
    # Start with schoolbook and try to reduce
    best_rank = n * m * p
    best_decomp = None
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try lower ranks
    # For 3x3x8, try ranks from 24 to 71
    for rank in range(24, best_rank):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(5):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1
            
            # Iterative refinement: ALS-style
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each (i,j), U[i*m+j, :] = sum_{k} T[i,j,k] * V[j*p+k, :] * W[k*n+i, :] / (V*W norm)
                # This is a least squares problem
                for i in range(n):
                    for j in range(m):
                        # U[i*m+j, :] should satisfy: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k] for all k
                        # This is: U[i*m+j, :] * (V_row_k outer W_row_k) = T[i,j,:]
                        # Let M[k,c] = V[j*p+k,c] * W[k*n+i,c]
                        # Then U[i*m+j, :] * M = T[i,j,:]
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve U[i*m+j, :] * M = T[i,j,:]
                        # This is: M^T * U[i*m+j, :]^T = T[i,j,:]^T
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M.T, T[i, j, :].T, rcond=None)[0].T
                        except:
                            pass
                
                # Update V similarly
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(M.T, T[:, j, k].T, rcond=None)[0].T
                        except:
                            pass
                
                # Update W similarly
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(M.T, T[i, :, k].T, rcond=None)[0].T
                        except:
                            pass
            
            # Round to half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            # Verify exactness
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(rank):
                            val += U_round[i * m + j, c] * V_round[j * p + k, c] * W_round[k * n + i, c]
                        error += abs(val - T[i, j, k])
            
            if error < 1e-6:
                best_rank = rank
                best_decomp = (U_round, V_round, W_round)
                break  # Found a valid decomposition for this rank
        
        if best_rank == rank:
            break  # Found the minimal rank
    
    # Convert to lists
    U, V, W = best_decomp
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Clean up: convert to int if possible
    def clean_matrix(mat):
        result = []
        for row in mat:
            clean_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    clean_row.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-9:
                    clean_row.append(val)  # keep as float
                else:
                    clean_row.append(val)
            result.append(clean_row)
        return result
    
    U_list = clean_matrix(U_list)
    V_list = clean_matrix(V_list)
    W_list = clean_matrix(W_list)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
