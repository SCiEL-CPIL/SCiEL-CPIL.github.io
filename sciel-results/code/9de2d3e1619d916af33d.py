import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit

    n, m, p = 4, 4, 4

    # Strassen 2x2 terms
    # Each term: (u_vec, v_vec, w_vec) for 2x2 matrices
    # Indices 0..3 for 2x2 matrices (row-major: a=0, b=1, c=2, d=3)
    def strassen_2x2_terms():
        terms = []
        # M0 = (a+d)(e+h)
        u0 = [1, 0, 0, 1]
        v0 = [1, 0, 0, 1]
        w0 = [1, 0, 0, 1]
        terms.append((u0, v0, w0))

        # M1 = (c+d)e
        u1 = [0, 0, 1, 1]
        v1 = [1, 0, 0, 0]
        w1 = [0, 0, 1, -1]
        terms.append((u1, v1, v1) if False else (u1, v1, w1))

        # M2 = a(f-h)
        u2 = [1, 0, 0, 0]
        v2 = [0, 1, 0, -1]
        w2 = [0, 1, 0, 1]
        terms.append((u2, v2, w2))

        # M3 = d(g-f)
        u3 = [0, 0, 0, 1]
        v3 = [0, -1, 1, 0]
        w3 = [-1, 1, 0, 0]
        terms.append((u3, v3, w3))

        # M4 = (a+b)h
        u4 = [1, 1, 0, 0]
        v4 = [0, 0, 0, 1]
        w4 = [1, 0, 1, 0]
        terms.append((u4, v4, w4))

        # M5 = (c-a)(e+f)
        u5 = [-1, 0, 1, 0]
        v5 = [1, 1, 0, 0]
        w5 = [0, 0, 0, 1]
        terms.append((u5, v5, w5))

        # M6 = (b-d)(g+h)
        u6 = [0, 1, 0, -1]
        v6 = [0, 0, 1, 1]
        w6 = [1, 0, 0, 0]
        terms.append((u6, v6, w6))

        return terms

    base_terms = strassen_2x2_terms()

    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Get block maps for each of the 7 Strassen terms
    def get_block_maps(m_idx):
        def block_map(br, bc, coeff=1):
            res = []
            for dr in range(2):
                for dc in range(2):
                    gi = (2 * br + dr) * 4 + (2 * bc + dc)
                    res.append((gi, coeff))
            return res

        X_map = [[None for _ in range(2)] for _ in range(2)]
        Y_map = [[None for _ in range(2)] for _ in range(2)]

        if m_idx == 0:
            # X = A00 + A11, Y = B00 + B11
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = [((2*0+r)*4 + (2*0+c), 1), ((2*1+r)*4 + (2*1+c), 1)]
                    Y_map[r][c] = [((2*0+r)*4 + (2*0+c), 1), ((2*1+r)*4 + (2*1+c), 1)]
            W_pos = [(0, 0, 1), (1, 1, 1)]
        elif m_idx == 1:
            # X = A10 + A11, Y = B00
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = [((2*1+r)*4 + (2*0+c), 1), ((2*1+r)*4 + (2*1+c), 1)]
                    Y_map[r][c] = [((2*0+r)*4 + (2*0+c), 1)]
            W_pos = [(1, 0, 1), (1, 1, -1)]
        elif m_idx == 2:
            # X = A00, Y = B01 - B11
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = [((2*0+r)*4 + (2*0+c), 1)]
                    Y_map[r][c] = [((2*0+r)*4 + (2*1+c), 1), ((2*1+r)*4 + (2*1+c), -1)]
            W_pos = [(0, 1, 1), (1, 1, 1)]
        elif m_idx == 3:
            # X = A11, Y = B10 - B00
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = [((2*1+r)*4 + (2*1+c), 1)]
                    Y_map[r][c] = [((2*1+r)*4 + (2*0+c), 1), ((2*0+r)*4 + (2*0+c), -1)]
            W_pos = [(0, 0, -1), (0, 1, 1)]
        elif m_idx == 4:
            # X = A00 + A01, Y = B11
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = [((2*0+r)*4 + (2*0+c), 1), ((2*0+r)*4 + (2*1+c), 1)]
                    Y_map[r][c] = [((2*1+r)*4 + (2*1+c), 1)]
            W_pos = [(0, 0, 1), (1, 0, 1)]
        elif m_idx == 5:
            # X = A10 - A00, Y = B00 + B01
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = [((2*1+r)*4 + (2*0+c), 1), ((2*0+r)*4 + (2*0+c), -1)]
                    Y_map[r][c] = [((2*0+r)*4 + (2*0+c), 1), ((2*0+r)*4 + (2*1+c), 1)]
            W_pos = [(1, 1, 1)]
        elif m_idx == 6:
            # X = A01 - A11, Y = B10 + B11
            for r in range(2):
                for c in range(2):
                    X_map[r][c] = [((2*0+r)*4 + (2*1+c), 1), ((2*1+r)*4 + (2*1+c), -1)]
                    Y_map[r][c] = [((2*1+r)*4 + (2*0+c), 1), ((2*1+r)*4 + (2*1+c), 1)]
            W_pos = [(0, 0, 1)]

        return X_map, Y_map, W_pos

    # Build the decomposition
    for m_idx in range(7):
        X_map, Y_map, W_pos = get_block_maps(m_idx)

        for s_idx in range(7):
            term_idx = m_idx * 7 + s_idx
            u_local, v_local, w_local = base_terms[s_idx]

            # Expand U
            for k in range(4):
                if u_local[k] != 0:
                    r = k // 2
                    c = k % 2
                    coeff = u_local[k]
                    for gi, gc in X_map[r][c]:
                        U[gi][term_idx] += coeff * gc

            # Expand V
            for k in range(4):
                if v_local[k] != 0:
                    r = k // 2
                    c = k % 2
                    coeff = v_local[k]
                    for gi, gc in Y_map[r][c]:
                        V[gi][term_idx] += coeff * gc

            # Expand W
            for k in range(4):
                if w_local[k] != 0:
                    r = k // 2
                    c = k % 2
                    coeff = w_local[k]
                    for br, bc, bc_coeff in W_pos:
                        wi = (2 * bc + c) * 4 + (2 * br + r)
                        W[wi][term_idx] += coeff * bc_coeff

    # Verify correctness with a random test
    if time.time() < deadline:
        rng = np.random.RandomState(42)
        A = rng.randint(-5, 5, size=(n, m))
        B = rng.randint(-5, 5, size=(m, p))
        C_expected = A @ B

        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)

        A_flat = A.flatten()
        B_flat = B.flatten()

        # C[i,j] = sum_k W[k*p + i, j] * (U @ A_flat)[k] * (V @ B_flat)[k]
        # Actually: C[i,j] = sum_t W[j*p + i, t] * (U[i*m + j, t]) * (V[j*p + k, t])... 
        # Let me think about the index convention again.
        # W[k*n + i, r] where k is col, i is row of output
        # So W index = k*n + i for output C[i, k]
        
        # The tensor decomposition says:
        # C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  -- no this isn't right either
        
        # Standard: C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r]... 
        # Wait, for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # Tensor: T[i,k,j] = A[i,k]*B[k,j] contributes to C[i,j]
        # Decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
        # where u_r is over (i,k) pairs, v_r is over (k,j) pairs, w_r is over (i,j) pairs
        # C[i,j] = sum_r w_r[i,j] * (sum_k u_r[i,k]*A[i,k]) * (sum_k v_r[k,j]*B[k,j])
        # No that's not right. 
        # C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * ... 
        
        # Let me just verify by computing:
        # For each output position (i, j):
        # C[i,j] should equal sum_r W[j*n + i, r] * (U[i*m + j, r]) * ... 
        # Hmm, I need to be more careful.
        
        # The convention is:
        # U has shape (n*m, rank): indexed by (i, k) flattened as i*m+k or i*n+m... 
        # Actually U[i*m+j, r] doesn't make sense for n=m=p=4.
        # U shape is (n*m, rank) = (16, 49). Index is i*m+k where i in [0,n), k in [0,m)
        # V shape is (m*p, rank) = (16, 49). Index is k*p+j where k in [0,m), j in [0,p)
        # W shape is (p*n, rank) = (16, 49). Index is j*n+i where j in [0,p), i in [0,n)
        
        # C[i,j] = sum_r W[j*n+i, r] * (sum_k U[i*m+k, r] * A[i,k]) * (sum_k V[k*p+j, r] * B[k,j])
        # No! That's wrong too. The correct formula is:
        # C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] summed over k and r
        # Which is: C[i,j] = sum_k sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        
        # Let's verify this way
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    for r in range(rank):
                        C_computed[i, j] += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
        
        if not np.allclose(C_computed, C_expected):
            # If verification fails, fall back to schoolbook rank-64
            U2 = [[0] * 64 for _ in range(n * m)]
            V2 = [[0] * 64 for _ in range(m * p)]
            W2 = [[0] * 64 for _ in range(p * n)]
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        t = i * m * p + k * p + j
                        U2[i*m+k][t] = 1
                        V2[k*p+j][t] = 1
                        W2[j*n+i][t] = 1
            return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
