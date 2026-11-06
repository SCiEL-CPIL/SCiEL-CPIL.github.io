import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 5, 5, 5

    # Strassen decomposition for 2x2x2
    # Index convention for the 2x2 blocks: [a11, a12, a21, a22]
    # U: rows indexed by (i,j) -> i*m+j  (m=2)
    # V: rows indexed by (j,k) -> j*p+k  (p=2)
    # W: rows indexed by (k,i) -> k*n+i  (n=2)  [official trace convention]
    def get_strassen_factors():
        factors = []
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    strassen_factors = get_strassen_factors()

    # Verify Strassen factors against the 2x2 matmul tensor (trace convention W[k*n+i, r])
    def verify_strassen(factors, nn=2, mm=2, pp=2):
        # Build the target tensor: T[i,j,k] = 1 if this is the (i,j) of A, (j,k) of B, (k,i) of C
        # We check: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == delta_{i,k} (since C[i,k] = sum_j A[i,j]*B[j,k])
        # Actually the matmul tensor is T[i,j,k] where i indexes rows of A, j columns of A / rows of B, k columns of B.
        # C[i,k] = sum_j A[i,j] B[j,k], so the tensor has T[i,j,k] = 1 for all i,j,k (it's the identity in the "multiplication" sense).
        # Wait: the matrix multiplication tensor M has M[i,j,k] = 1 for all i,j,k (it's the all-ones tensor of size n*m*p).
        # No: M[i,j,k] = delta_{j,j} = 1 always. The constraint is:
        #   C[i,k] = sum_j A[i,j] B[j,k]
        # So the tensor T[i,j,k] = 1 for all valid i,j,k.
        # The decomposition requires: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
        # For 2x2: n=m=p=2, so U is 4x7, V is 4x7, W is 4x7.
        # U index: i*2 + j (row i, col j of A)
        # V index: j*2 + k (row j, col k of B)
        # W index: k*2 + i (row k, col i of C)  [trace convention: W[k*n+i, r]]
        U_arr = np.zeros((nn * mm, len(factors)))
        V_arr = np.zeros((mm * pp, len(factors)))
        W_arr = np.zeros((pp * nn, len(factors)))
        for r, (uf, vf, wf) in enumerate(factors):
            for idx in range(4):
                U_arr[idx, r] += uf[idx]
                V_arr[idx, r] += vf[idx]
                W_arr[idx, r] += wf[idx]
        # Check: for all i, j, k: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] == 1
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    u_val = U_arr[i * mm + j]
                    v_val = V_arr[j * pp + k]
                    w_val = W_arr[k * nn + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    strassen_ok = verify_strassen(strassen_factors)

    # Use [2,2,1] block partition
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

    # Compute total rank: use Strassen for 2x2x2 blocks if verified, else schoolbook
    use_strassen = strassen_ok
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if use_strassen and ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    break
                
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if use_strassen and ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        local_to_global_U = [
                            (offsets[i] + 0) * m + (offsets[j] + 0),
                            (offsets[i] + 0) * m + (offsets[j] + 1),
                            (offsets[i] + 1) * m + (offsets[j] + 0),
                            (offsets[i] + 1) * m + (offsets[j] + 1),
                        ]
                        local_to_global_V = [
                            (offsets[j] + 0) * p + (offsets[k] + 0),
                            (offsets[j] + 0) * p + (offsets[k] + 1),
                            (offsets[j] + 1) * p + (offsets[k] + 0),
                            (offsets[j] + 1) * p + (offsets[k] + 1),
                        ]
                        local_to_global_W = [
                            (offsets[k] + 0) * n + (offsets[i] + 0),
                            (offsets[k] + 1) * n + (offsets[i] + 0),
                            (offsets[k] + 0) * n + (offsets[i] + 1),
                            (offsets[k] + 1) * n + (offsets[i] + 1),
                        ]
                        
                        for idx in range(4):
                            if uf[idx] != 0:
                                U[local_to_global_U[idx]][col] += uf[idx]
                            if vf[idx] != 0:
                                V[local_to_global_V[idx]][col] += vf[idx]
                            if wf[idx] != 0:
                                W[local_to_global_W[idx]][col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U_idx = (offsets[i] + li) * m + (offsets[j] + lj)
                                V_idx = (offsets[j] + lj) * p + (offsets[k] + lk)
                                W_idx = (offsets[k] + lk) * n + (offsets[i] + li)
                                U[U_idx][col] = 1.0
                                V[V_idx][col] = 1.0
                                W[W_idx][col] = 1.0
                                col += 1
    
    return {"U": U, "V": V, "W": W}
