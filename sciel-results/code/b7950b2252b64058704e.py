"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    def verify_exact(U, V, W, rank):
        """Verify that U @ diag(1) @ V^T reconstructed via W matches matrix mult tensor."""
        # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T[i,j,k] = 1 if the (i,k) entry of A*B equals sum_j A[i,j]*B[j,k]
        # Actually the matrix multiplication tensor has T[i,j,k] = 1 for all (i,j,k) in the product
        # The decomposition requires: for all (i,j,k): T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Since T[i,j,k] = 1 for all valid (i,j,k), we need:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check: for each (i,j,k), sum over c of U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_random_search(max_rank, min_rank, max_iter=50000):
        """Try to find a decomposition with rank between min_rank and max_rank."""
        for R in range(max_rank, min_rank - 1, -1):
            if time.time() > deadline:
                return None
            for _ in range(max_iter // (max_rank - min_rank + 1)):
                if time.time() > deadline:
                    return None
                # Generate random sparse decomposition
                U = np.zeros((n*m, R), dtype=np.float64)
                V = np.zeros((m*p, R), dtype=np.float64)
                W = np.zeros((p*n, R), dtype=np.float64)
                
                # Initialize with random values
                for c in range(R):
                    # Randomly select which entries are non-zero
                    nnz_u = random.randint(1, 3)
                    nnz_v = random.randint(1, 3)
                    nnz_w = random.randint(1, 3)
                    
                    u_idx = random.sample(range(n*m), min(nnz_u, n*m))
                    v_idx = random.sample(range(m*p), min(nnz_v, m*p))
                    w_idx = random.sample(range(p*n), min(nnz_w, p*n))
                    
                    for idx in u_idx:
                        U[idx, c] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
                    for idx in v_idx:
                        V[idx, c] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
                    for idx in w_idx:
                        W[idx, c] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
                
                # Check if this works
                if verify_exact(U.tolist(), V.tolist(), W.tolist(), R):
                    return (U.tolist(), V.tolist(), W.tolist(), R)
        return None
    
    # Try to find a better decomposition
    result = try_random_search(best_rank - 1, max(30, best_rank // 2), max_iter=30000)
    if result is not None:
        U_r, V_r, W_r, R_r = result
        best = {"U": U_r, "V": V_r, "W": W_r}
        best_rank = R_r
        print(f"Found decomposition with rank {R_r}")
    
    return best
# EVOLVE-BLOCK-END
