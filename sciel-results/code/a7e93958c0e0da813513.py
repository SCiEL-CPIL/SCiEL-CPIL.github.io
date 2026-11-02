import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p  # 90
    
    # Build the matrix multiplication tensor
    # T[i, j, k] = 1 where (i,j) indexes U, (j,k) indexes V, (k,i) indexes W
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index: U[i*m + j], V[j*p + k], W[k*n + i]
    
    # The tensor can be viewed as a 3D array T[i, j, k]
    # For a rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Strategy: Try to find a rank-r decomposition using the known structure.
    # For matrix multiplication, the minimum rank is known to be less than n*m*p for many cases.
    # For 3x5x6, let's try to find a better decomposition.
    
    # Approach: Use the fact that matrix multiplication tensor has structure.
    # We can try to exploit the separability in the j index.
    
    # For each j, the sub-tensor T[:, j, :] is an n x p matrix (all ones for this case).
    # So T[i, j, k] = 1 for all i, j, k.
    
    # This means the tensor is fully separable: T[i,j,k] = 1 * 1 * 1
    # So a rank-1 decomposition exists:
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1*1*1 = 1 = T[i,j,k]
    
    # Wait, let me verify this is correct for matrix multiplication.
    # The matrix multiplication tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
    # In standard matrix multiplication, C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all i,j,k.
    # Yes! The tensor is indeed all ones.
    
    # Therefore, a rank-1 decomposition works:
    rank = 1
    U = np.ones((n*m, rank))
    V = np.ones((m*p, rank))
    W = np.ones((p*n, rank))
    
    # Verify the decomposition
    # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal T[i,j,k] = 1
    verified = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                prod = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                if abs(prod - 1.0) > 1e-10:
                    verified = False
                    break
            if not verified:
                break
        if not verified:
            break
    
    if verified:
        # Convert to lists for return (or keep as arrays - check what's expected)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if rank-1 doesn't work (shouldn't happen)
    rank = rank_schoolbook
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

