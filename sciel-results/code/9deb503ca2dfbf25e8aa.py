import time
import numpy as np

def construct():
    t0 = time.time()
    LIMIT = t0 + 138
    N, M, P = 3, 4, 7
    dim_u, dim_v, dim_w = N*M, M*P, P*N

    terms = []  # list of (u_sparse, v_sparse, w_sparse)

    def _add(u_sp, v_sp, w_sp):
        terms.append((u_sp, v_sp, w_sp))

    # 2x2x2 Strassen block (7 rank-1 terms)
    def strassen222(i0, j0, k0):
        # Term 1: (i0,j0)+(i1,j1) * (j0,k0)+(j1,k1) -> (k0,i0)+(k1,i1)
        i1, j1, k1 = i0+1, j0+1, k0+1
        _add({i0*M+j0:1, i1*M+j1:1}, {j0*P+k0:1, j1*P+k1:1}, {k0*N+i0:1, k1*N+i1:1})
        # Term 2: (i1,j0)+(i1,j1) * (j0,k0) -> (k0,i1)+(k1,i1)
        _add({i1*M+j0:1, i1*M+j1:1}, {j0*P+k0:1}, {k0*N+i1:1, k1*N+i1:-1})
        # Term 3: (i0,j0) * (j0,k1)+(j1,k1) -> (k1,i0)+(k1,i1)
        _add({i0*M+j0:1}, {j0*P+k1:1, j1*P+k1:-1}, {k1*N+i0:1, k1*N+i1:1})
        # Term 4: (i1,j1) * (j1,k0)+(j0,k0) -> (k0,i0)+(k0,i1)
        _add({i1*M+j1:1}, {j1*P+k0:1, j0*P+k0:-1}, {k0*N+i0:1, k0*N+i1:1})
        # Term 5: (i0,j0)+(i0,j1) * (j1,k1) -> (k0,i0)+(k1,i0)
        _add({i0*M+j0:1, i0*M+j1:1}, {j1*P+k1:1}, {k0*N+i0:-1, k1*N+i0:1})
        # Term 6: (i1,j0)+(i0,j0) * (j0,k0)+(j0,k1) -> (k1,i1)
        _add({i1*M+j0:1, i0*M+j0:-1}, {j0*P+k0:1, j0*P+k1:1}, {k1*N+i1:1})
        # Term 7: (i0,j1)+(i1,j1) * (j1,k0)+(j1,k1) -> (k0,i0)
        _add({i0*M+j1:1, i1*M+j1:-1}, {j1*P+k0:1, j1*P+k1:1}, {k0*N+i0:1})

    # 2x2x1 block (4 terms, can't use Strassen effectively)
    def block221(i0, j0, k0):
        i1, j1 = i0+1, j0+1
        # Try to save: (i0,j0)*(j0,k0) + (i0,j1)*(j1,k0) + (i1,j0)*(j0,k0) + (i1,j1)*(j1,k0)
        # That's 4 terms, no savings possible for 2x2x1
        _add({i0*M+j0:1}, {j0*P+k0:1}, {k0*N+i0:1})
        _add({i0*M+j1:1}, {j1*P+k0:1}, {k0*N+i0:1})
        _add({i1*M+j0:1}, {j0*P+k0:1}, {k0*N+i1:1})
        _add({i1*M+j1:1}, {j1*P+k0:1}, {k0*N+i1:1})

    # 1x2x2 block (4 terms)
    def block122(i0, j0, k0):
        j1, k1 = j0+1, k0+1
        _add({i0*M+j0:1}, {j0*P+k0:1}, {k0*N+i0:1})
        _add({i0*M+j0:1}, {j0*P+k1:1}, {k1*N+i0:1})
        _add({i0*M+j1:1}, {j1*P+k0:1}, {k0*N+i0:1})
        _add({i0*M+j1:1}, {j1*P+k1:1}, {k1*N+i0:1})

    # 1x2x1 block (1 term)
    def block121(i0, j0, k0):
        _add({i0*M+j0:1}, {j0*P+k0:1}, {k0*N+i0:1})

    # 2x1x2 block (4 terms)
    def block212(i0, j0, k0):
        i1, k1 = i0+1, k0+1
        _add({i0*M+j0:1}, {j0*P+k0:1}, {k0*N+i0:1})
        _add({i0*M+j0:1}, {j0*P+k1:1}, {k1*N+i0:1})
        _add({i1*M+j0:1}, {j0*P+k0:1}, {k0*N+i1:1})
        _add({i1*M+j0:1}, {j0*P+k1:1}, {k1*N+i1:1})

    # 2x1x1 block (2 terms)
    def block211(i0, j0, k0):
        i1 = i0+1
        _add({i0*M+j0:1}, {j0*P+k0:1}, {k0*N+i0:1})
        _add({i1*M+j0:1}, {j0*P+k0:1}, {k0*N+i1:1})

    # 1x1x2 block (2 terms)
    def block112(i0, j0, k0):
        k1 = k0+1
        _add({i0*M+j0:1}, {j0*P+k0:1}, {k0*N+i0:1})
        _add({i0*M+j0:1}, {j0*P+k1:1}, {k1*N+i0:1})

    # 1x1x1 block (1 term)
    def block111(i0, j0, k0):
        _add({i0*M+j0:1}, {j0*P+k0:1}, {k0*N+i0:1})

    # 2x2x3: use 2x2x2 + 2x2x1 = 7 + 4 = 11
    def block223(i0, j0, k0):
        strassen222(i0, j0, k0)
        block221(i0, j0, k0+2)

    # 2x2x4: use 2x2x2 + 2x2x2 = 14
    def block224(i0, j0, k0):
        strassen222(i0, j0, k0)
        strassen222(i0, j0, k0+2)

    # 2x2x5: 2x2x2 + 2x2x2 + 2x2x1 = 18
    def block225(i0, j0, k0):
        strassen222(i0, j0, k0)
        strassen222(i0, j0, k0+2)
        block221(i0, j0, k0+4)

    # 2x2x6: 3 * 2x2x2 = 21
    def block226(i0, j0, k0):
        strassen222(i0, j0, k0)
        strassen222(i0, j0, k0+2)
        strassen222(i0, j0, k0+4)

    # 2x2x7: 3 * 2x2x2 + 1 * 2x2x1 = 25
    def block227(i0, j0, k0):
        strassen222(i0, j0, k0)
        strassen222(i0, j0, k0+2)
        strassen222(i0, j0, k0+4)
        block221(i0, j0, k0+6)

    # 1x2x7: 3 * 1x2x2 + 1x2x1 = 13
    def block127(i0, j0, k0):
        block122(i0, j0, k0)
        block122(i0, j0, k0+2)
        block122(i0, j0, k0+4)
        block121(i0, j0, k0+6)

    # 1x4x2: 2 * 1x2x2 = 8
    def block142(i0, j0, k0):
        block122(i0, j0, k0)
        block122(i0, j0+2, k0)

    # 1x4x4: 4 * 1x2x2 = 16
    def block144(i0, j0, k0):
        block122(i0, j0, k0)
        block122(i0, j0+2, k0)
        block122(i0, j0, k0+2)
        block122(i0, j0+2, k0+2)

    # 1x4x7: 3 * 1x2x2 + 1x2x1 for each pair of j... 
    # Actually: 1x4x7 = 1x2x7 + 1x2x7 (split j) = 13 + 13 = 26
    # Or: 1x4x7 = 1x4x4 + 1x4x3 = 16 + 12 = 28
    # Or: 1x4x7 = 1x4x2 + 1x4x2 + 1x4x2 + 1x4x1 = 8+8+8+4 = 28
    # Best: split into two 1x2x7: 13+13 = 26
    def block147(i0, j0, k0):
        block127(i0, j0, k0)
        block127(i0, j0+2, k0)

    # 2x4x7: split j into two 2x2x7: 25+25 = 50
    def block247(i0, j0, k0):
        block227(i0, j0, k0)
        block227(i0, j0+2, k0)

    # 3x4x7: split i: 2x4x7 + 1x4x7 = 50 + 26 = 76
    def block347(i0, j0, k0):
        block247(i0, j0, k0)
        block147(i0+2, j0, k0)

    # Alternative decomposition: 3x4x7 = 2x4x7 + 1x4x7
    # 2x4x7 = 2x2x7 + 2x2x7 = 25 + 25 = 50
    # 1x4x7 = 1x2x7 + 1x2x7 = 13 + 13 = 26
    # Total: 76

    # Another alternative: 3x4x7 split differently
    # 3x4x7 = 3x2x4 + 3x2x3? No, let's stick with what we have.
    
    # Let's try: 3x4x7 = 3x4x4 + 3x4x3
    # 3x4x4 = 2x4x4 + 1x4x4 = (2x2x4 + 2x2x4) + 1x4x4 = (14+14) + 16 = 44
    # 3x4x3 = 2x4x3 + 1x4x3 = (2x2x3 + 2x2x3) + (1x2x3 + 1x2x3) = (11+11) + (6+6) = 34
    # Total: 44 + 34 = 78. Worse.

    # Another: 3x4x7 = 3x2x7 + 3x2x7 (split j)
    # 3x2x7 = 2x2x7 + 1x2x7 = 25 + 13 = 38
    # Total: 38 + 38 = 76. Same.

    # 3x4x7 = 3x4x2 + 3x4x2 + 3x4x2 + 3x4x1
    # 3x4x2 = 2x4x2 + 1x4x2 = (2x2x2 + 2x2x2) + 1x4x2 = (7+7) + 8 = 22
    # 3x4x1 = 2x4x1 + 1x4x1 = (2x2x1 + 2x2x1) + 1x4x1 = (4+4) + 4 = 12
    # Total: 22+22+22+12 = 78. Worse.

    # So 76 is our best with this decomposition strategy.

    block347(0, 0, 0)

    # Convert sparse terms to dense vectors
    rank = len(terms)
    U_mat = np.zeros((dim_u, rank), dtype=np.int64)
    V_mat = np.zeros((dim_v, rank), dtype=np.int64)
    W_mat = np.zeros((dim_w, rank), dtype=np.int64)

    for r, (u_sp, v_sp, w_sp) in enumerate(terms):
        for idx, val in u_sp.items():
            U_mat[idx, r] = val
        for idx, val in v_sp.items():
            V_mat[idx, r] = val
        for idx, val in w_sp.items():
            W_mat[idx, r] = val

    # Quick verification
    if time.time() < LIMIT:
        # Verify: sum_r U[:,r] * V[:,r] * W[:,r] should equal the matmul tensor
        # For a random test: A (3x4), B (4x7)
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T[i,j,k] = delta (coefficient)
        # Our decomposition: T[i,j,k] = sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
        
        # Test with specific values
        A_test = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=float)
        B_test = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18,19,20,21],[22,23,24,25,26,27,28]], dtype=float)
        C_test = A_test @ B_test
        
        C_recon = np.zeros((N, P), dtype=float)
        for r in range(rank):
            u_vec = U_mat[:, r]
            v_vec = V_mat[:, r]
            w_vec = W_mat[:, r]
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        C_recon[i, k] += u_vec[i*M+j] * v_vec[j*P+k] * w_vec[k*N+i]
        
        if not np.allclose(C_recon, C_test):
            raise RuntimeError("Verification failed")

    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
