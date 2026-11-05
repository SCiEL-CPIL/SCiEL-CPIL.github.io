import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 5, 5, 5

    # Strassen decomposition for 2x2x2
    # Standard Strassen: M1..M7
    # U factors (row,col of A), V factors (row,col of B), W factors (row,col of C)
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    def get_strassen_factors():
        factors = []
        # M1 = (A00+A11)(B00+B11)
        factors.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
        # M2 = (A10+A11)B00
        factors.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]))
        # M3 = A00(B10-B11)
        factors.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
        # M4 = A11(B01-B00)
        factors.append(([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]))
        # M5 = (A00+A01)B11
        factors.append(([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]))
        # M6 = (A10-A00)(B00+B01)
        factors.append(([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]))
        # M7 = (A01-A11)(B10+B11)
        factors.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]))
        return factors

    strassen_factors = get_strassen_factors()

    # Block partition: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)

    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
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
                
                if ni == 2 and mj == 2 and pk == 2:
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
    
    # Verify correctness with a small test
    if time.time() < deadline - 5.0:
        try:
            import numpy as np
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            # Test with random matrices
            np.random.seed(42)
            A_test = np.random.randint(-2, 3, (n, m)).astype(float)
            B_test = np.random.randint(-2, 3, (m, p)).astype(float)
            C_expected = A_test @ B_test
            
            # Compute via tensor decomposition
            C_computed = np.zeros((n, p), dtype=float)
            for r in range(total_rank):
                u_vec = U_arr[:, r]  # length n*m
                v_vec = V_arr[:, r]  # length m*p
                w_vec = W_arr[:, r]  # length p*n
                
                # u_vec[i*m+j] is coefficient for A[i,j]
                # v_vec[j*p+k] is coefficient for B[j,k]
                # w_vec[k*n+i] is coefficient for C[i,k]
                a_scaled = np.zeros((n, m))
                b_scaled = np.zeros((m, p))
                for ii in range(n):
                    for jj in range(m):
                        a_scaled[ii, jj] = u_vec[ii * m + jj]
                for jj in range(m):
                    for kk in range(p):
                        b_scaled[jj, kk] = v_vec[jj * p + kk]
                
                m_val = np.sum(a_scaled * A_test) * np.sum(b_scaled * B_test)
                
                for ii in range(n):
                    for kk in range(p):
                        C_computed[ii, kk] += m_val * w_vec[kk * n + ii]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # If verification fails, fall back to pure schoolbook
                U_fallback = [[0.0] * (n*m*p) for _ in range(n * m)]
                V_fallback = [[0.0] * (n*m*p) for _ in range(m * p)]
                W_fallback = [[0.0] * (n*m*p) for _ in range(p * n)]
                col_fb = 0
                for ii in range(n):
                    for jj in range(m):
                        for kk in range(p):
                            U_fallback[ii * m + jj][col_fb] = 1.0
                            V_fallback[jj * p + kk][col_fb] = 1.0
                            W_fallback[kk * n + ii][col_fb] = 1.0
                            col_fb += 1
                return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
