import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    n_parts = [1, 2]
    m_parts = [2, 1, 2]
    p_parts = [2, 1, 2]
    
    n_offsets = [0]
    for s in n_parts:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_parts:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_parts:
        p_offsets.append(p_offsets[-1] + s)
    
    total_rank = 0
    for ni in n_parts:
        for mj in m_parts:
            for pk in p_parts:
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for ai, ni in enumerate(n_parts):
        for aj, mj in enumerate(m_parts):
            for ak, pk in enumerate(p_parts):
                i0 = n_offsets[ai]
                j0 = m_offsets[aj]
                k0 = p_offsets[ak]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_s = [[1, 1, 1, 0, 1, -1, 1],
                           [-1, 1, 0, 1, 0, 1, -1]]
                    V_s = [[-1, 1, 0, 1, -1, 1, 1],
                           [1, 0, 1, -1, 1, 1, 1]]
                    W_s = [[1, 0, 1, 1, 1, 0, 1],
                           [1, -1, 1, 1, 0, 1, 0]]
                    
                    for li in range(2):
                        for lj in range(2):
                            u_idx = (i0 + li) * m + (j0 + lj)
                            for t in range(7):
                                U[u_idx][col + t] = U_s[li][t]
                    
                    for lq in range(2):
                        for lk in range(2):
                            v_idx = (j0 + lq) * p + (k0 + lk)
                            for t in range(7):
                                V[v_idx][col + t] = V_s[lq][t]
                    
                    for lp in range(2):
                        for lr in range(2):
                            w_idx = (k0 + lp) * n + (i0 + lr)
                            for t in range(7):
                                W[w_idx][col + t] = W_s[lp][t]
                    
                    col += 7
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                u_idx = (i0 + li) * m + (j0 + lj)
                                v_idx = (j0 + lj) * p + (k0 + lk)
                                w_idx = (k0 + lk) * n + (i0 + li)
                                U[u_idx][col] = 1
                                V[v_idx][col] = 1
                                W[w_idx][col] = 1
                                col += 1
    
    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Test with random matrices
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C from decomposition
    # C[k, i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually, the tensor decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = A[i,j]*B[j,k]
    
    # So C[k,i] = sum_j A[i,j]*B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    C_computed = np.zeros((p, n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(total_rank):
                    C_computed[k, i] += A[i, j] * B[j, k] * 0  # This isn't right
    
    # Actually, let me verify differently. The decomposition should satisfy:
    # For all i,j,k: A[i,j]*B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(total_rank))
                expected = 1  # Since we're checking the tensor, not actual matrix product
                # Wait, the tensor T[i,j,k] = 1 if this is a valid multiplication term
                # Actually T[i,j,k] should be 1 for all i,j,k in the multiplication tensor
                # No, the multiplication tensor has T[i,j,k] = 1 for the "identity" mapping
                # Let me think again.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    # (it's the tensor that maps A and B to C where C[k,i] = sum_j A[i,j]B[j,k])
    # So the decomposition should give T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(total_rank))
                if abs(val - 1) > 1e-10:
                    max_err = max(max_err, abs(val - 1))
    
    if max_err > 1e-10:
        print(f"Verification failed! Max error: {max_err}")
    
    return {"U": U, "V": V, "W": W}
