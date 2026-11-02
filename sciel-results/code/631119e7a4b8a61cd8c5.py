import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    # Rank 56 construction using Strassen on 2x2 blocks
    # 8 blocks of 2x2 multiplication
    # Each block has rank 7
    total_rank = 56
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Indices for A (2x4): i in 0..1, j in 0..3
    # Indices for B (4x8): j in 0..3, k in 0..7
    # Indices for C (2x8): i in 0..1, k in 0..7
    # W index: k*n + i -> k*2 + i
    
    # u: block index for middle dimension (0, 1) -> rows 2u, 2u+1 of B, cols 2u, 2u+1 of A
    # v: block index for last dimension (0, 1, 2, 3) -> cols 2v, 2v+1 of B, cols 2v, 2v+1 of C
    
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A block indices
            # A[0, 2u], A[0, 2u+1], A[1, 2u], A[1, 2u+1]
            a11_idx = 0 * 4 + 2 * u
            a12_idx = 0 * 4 + 2 * u + 1
            a21_idx = 1 * 4 + 2 * u
            a22_idx = 1 * 4 + 2 * u + 1
            
            # B block indices
            # B[2u, 2v], B[2u, 2v+1], B[2u+1, 2v], B[2u+1, 2v+1]
            b11_idx = 2 * u * 8 + 2 * v
            b12_idx = 2 * u * 8 + 2 * v + 1
            b21_idx = (2 * u + 1) * 8 + 2 * v
            b22_idx = (2 * u + 1) * 8 + 2 * v + 1
            
            # C block indices (W indices)
            # C[0, 2v], C[0, 2v+1], C[1, 2v], C[1, 2v+1]
            # W index = k * n + i = k * 2 + i
            c11_idx = 2 * v * 2 + 0
            c12_idx = (2 * v + 1) * 2 + 0
            c21_idx = 2 * v * 2 + 1
            c22_idx = (2 * v + 1) * 2 + 1
            
            # s=0: M1 = (A11+A22)(B11+B22)
            U[a11_idx][base_col] += 1
            U[a22_idx][base_col] += 1
            V[b11_idx][base_col] += 1
            V[b22_idx][base_col] += 1
            W[c11_idx][base_col] += 1
            W[c22_idx][base_col] += 1
            
            # s=1: M2 = (A21+A22)B11
            U[a21_idx][base_col + 1] += 1
            U[a22_idx][base_col + 1] += 1
            V[b11_idx][base_col + 1] += 1
            W[c21_idx][base_col + 1] += 1
            W[c22_idx][base_col + 1] -= 1
            
            # s=2: M3 = A11(B12-B22)
            U[a11_idx][base_col + 2] += 1
            V[b12_idx][base_col + 2] += 1
            V[b22_idx][base_col + 2] -= 1
            W[c12_idx][base_col + 2] += 1
            W[c22_idx][base_col + 2] += 1
            
            # s=3: M4 = A22(B21-B11)
            U[a22_idx][base_col + 3] += 1
            V[b21_idx][base_col + 3] += 1
            V[b11_idx][base_col + 3] -= 1
            W[c11_idx][base_col + 3] += 1
            W[c21_idx][base_col + 3] += 1
            
            # s=4: M5 = (A11+A12)B22
            U[a11_idx][base_col + 4] += 1
            U[a12_idx][base_col + 4] += 1
            V[b22_idx][base_col + 4] += 1
            W[c11_idx][base_col + 4] -= 1
            W[c12_idx][base_col + 4] += 1
            
            # s=5: M6 = (A21-A11)(B11+B12)
            U[a21_idx][base_col + 5] += 1
            U[a11_idx][base_col + 5] -= 1
            V[b11_idx][base_col + 5] += 1
            V[b12_idx][base_col + 5] += 1
            W[c22_idx][base_col + 5] += 1
            
            # s=6: M7 = (A12-A22)(B21+B22)
            U[a12_idx][base_col + 6] += 1
            U[a22_idx][base_col + 6] -= 1
            V[b21_idx][base_col + 6] += 1
            V[b22_idx][base_col + 6] += 1
            W[c11_idx][base_col + 6] += 1
            
    # Verify decomposition correctness
    if time.time() < deadline:
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check: for all i,j,k, sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1 if valid, else 0
        # Actually the decomposition is: T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
        # where T[i,j,k] = 1 for all valid (i,j,k) in the multiplication tensor
        # Wait - the tensor for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for the bilinear form)
        # No wait - the multiplication tensor has rank r if there exist U,V,W such that:
        # T[i,j,k] = sum_s U[i,j,s] * V[j,k,s] * W[k,i,s]
        # where T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
        # For standard matrix multiplication, T[i,j,k] = 1 for ALL valid (i,j,k)
        
        # Let's verify: reconstruct the tensor and check it's all ones
        # Tensor shape: (n, m, p) = (2, 4, 8)
        # T[i,j,k] should equal 1 for all i in 0..1, j in 0..3, k in 0..7
        
        # U index: i*m + j
        # V index: j*p + k  
        # W index: k*n + i
        
        reconstructed = np.zeros((n, m, p), dtype=np.float64)
        for s in range(total_rank):
            # U[:, s] has shape (n*m,)
            # V[:, s] has shape (m*p,)
            # W[:, s] has shape (p*n,)
            u_col = U_arr[:, s]  # (n*m,)
            v_col = V_arr[:, s]  # (m*p,)
            w_col = W_arr[:, s]  # (p*n,)
            
            # Reshape and do outer product
            u_reshaped = u_col.reshape(n, m)
            v_reshaped = v_col.reshape(m, p)
            w_reshaped = w_col.reshape(p, n)
            
            # For each (i,j,k): contribution = U[i,j,s] * V[j,k,s] * W[k,i,s]
            # We need to compute this efficiently
            # Sum over j: for fixed i,k, sum_j U[i,j,s]*V[j,k,s] * W[k,i,s]
            # = W[k,i,s] * (U_reshaped[:,s] @ V_reshaped[:,s])[i,k] ... no
            
            # Actually: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            # For fixed s: sum_j U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            # = W[k*n+i, s] * sum_j U[i*m+j, s] * V[j*p+k, s]
            # = W[k*n+i, s] * (U_s_row_i @ V_s_col_k) where U_s_row_i = U[i*m:(i+1)*m, s], V_s_col_k = V[0*p+k:m*p+k:p, s]
            
            # Let me just do it directly for verification
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        reconstructed[i, j, k] += U_arr[i * m + j, s] * V_arr[j * p + k, s] * W_arr[k * n + i, s]
        
        # Check that reconstructed is all ones
        if not np.allclose(reconstructed, 1.0, atol=1e-10):
            # Decomposition is wrong - this shouldn't happen
            print("WARNING: Decomposition verification failed")
            print("Max error:", np.max(np.abs(reconstructed - 1.0)))
    
    return {"U": U, "V": V, "W": W}
