import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    def build_strassen_2x2_blocks():
        """Standard Strassen on 2x2 blocks. Rank 189."""
        block_size = 2
        blocks_per_dim = 3
        rank = (blocks_per_dim ** 3) * 7
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        strassen_patterns = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]

        col_idx = 0
        for I in range(blocks_per_dim):
            for J in range(blocks_per_dim):
                for K in range(blocks_per_dim):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_patterns:
                        for (pi, pj), val in u_coeffs.items():
                            r = I * block_size + pi
                            c = J * block_size + pj
                            U[r * m + c, col_idx] += val
                        for (pi, pj), val in v_coeffs.items():
                            r = J * block_size + pi
                            c = K * block_size + pj
                            V[r * p + c, col_idx] += val
                        for (pi, pj), val in w_coeffs.items():
                            k = K * block_size + pj
                            i = I * block_size + pi
                            W[k * n + i, col_idx] += val
                        col_idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank

    def verify_decomposition(U, V, W):
        """Verify if the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for _ in range(10):
            A = np.random.randint(-10, 10, (n, m))
            B = np.random.randint(-10, 10, (m, p))
            C_true = A @ B
            C_comp = np.zeros((n, p))
            for r in range(rank):
                a_proj = np.zeros((n, m))
                for idx in range(n * m):
                    a_proj.flat[idx] = U[idx, r]
                a_mat = a_proj.reshape(n, m)
                b_proj = np.zeros((m, p))
                for idx in range(m * p):
                    b_proj.flat[idx] = V[idx, r]
                b_mat = b_proj.reshape(m, p)
                w_col = W[:, r]
                # C_comp += w_col^T (reshaped) * (a_mat @ b_mat)
                # W is indexed by k*n+i. So w_col[k*n+i] is the coefficient for C[i,k]
                w_mat = np.zeros((p, n))
                for k in range(p):
                    for i in range(n):
                        w_mat[k, i] = w_col[k * n + i]
                # C[i,k] += sum_r w_mat[k,i] * (a_mat @ b_mat)[i,k]
                C_comp += (a_mat @ b_mat) * w_mat.T
            if not np.allclose(C_true, C_comp, atol=1e-6):
                return False
        return True

    def build_kronecker_4x27():
        """
        Kronecker product of a rank-4 decomposition for 2x2 and rank-27 for 3x3.
        This yields rank 108.
        The rank-4 decomposition for 2x2 is the trivial schoolbook (since rank 4 is the min for 2x2).
        """
        # 2x2 schoolbook (rank 4)
        # U2: 4x4, V2: 4x4, W2: 4x4
        # Rows of U2: 00, 01, 10, 11 (i*m+j -> i*2+j)
        # Rows of V2: 00, 01, 10, 11 (j*p+k -> j*2+k)
        # Rows of W2: 00, 01, 10, 11 (k*n+i -> k*2+i)
        
        U2 = np.zeros((4, 4), dtype=int)
        V2 = np.zeros((4, 4), dtype=int)
        W2 = np.zeros((4, 4), dtype=int)
        
        # r=0: A00 B00 -> C00
        U2[0, 0] = 1
        V2[0, 0] = 1
        W2[0, 0] = 1
        
        # r=1: A01 B10 -> C01 (k=1, i=0 -> W[1*2+0] = W[2])
        # Wait, W index is k*n+i.
        # C01 is i=0, k=1. So W[1*2+0] = W[2].
        # U index i*m+j. A01 is i=0, j=1. U[0*2+1] = U[1].
        # V index j*p+k. B10 is j=1, k=0. V[1*2+0] = V[2].
        U2[1, 1] = 1
        V2[2, 1] = 1
        W2[2, 1] = 1
        
        # r=2: A10 B00 -> C10 (i=1, k=0 -> W[0*2+1] = W[1])
        # A10 is i=1, j=0. U[1*2+0] = U[2].
        # B00 is j=0, k=0. V[0*2+0] = V[0].
        U2[2, 2] = 1
        V2[0, 2] = 1
        W2[1, 2] = 1
        
        # r=3: A11 B10 -> C11 (i=1, k=1 -> W[1*2+1] = W[3])
        # A11 is i=1, j=1. U[1*2+1] = U[3].
        # B10 is j=1, k=0. V[1*2+0] = V[2].
        U2[3, 3] = 1
        V2[2, 3] = 1
        W2[3, 3] = 1
        
        # 3x3 schoolbook (rank 27)
        U3 = np.zeros((9, 27), dtype=int)
        V3 = np.zeros((9, 27), dtype=int)
        W3 = np.zeros((9, 27), dtype=int)
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    r = i * 9 + j * 3 + k
                    U3[i * 3 + j, r] = 1
                    V3[j * 3 + k, r] = 1
                    W3[k * 3 + i, r] = 1

        # Construct 6x6 factors
        rank = 4 * 27
        U6 = np.zeros((36, rank), dtype=int)
        V6 = np.zeros((36, rank), dtype=int)
        W6 = np.zeros((36, rank), dtype=int)
        
        for r2 in range(4):
            for r3 in range(27):
                r = r2 * 27 + r3
                
                nz_u2 = np.nonzero(U2[:, r2])[0]
                nz_v2 = np.nonzero(V2[:, r2])[0]
                nz_w2 = np.nonzero(W2[:, r2])[0]
                
                nz_u3 = np.nonzero(U3[:, r3])[0]
                nz_v3 = np.nonzero(V3[:, r3])[0]
                nz_w3 = np.nonzero(W3[:, r3])[0]
                
                for u2 in nz_u2:
                    i2 = u2 // 2
                    j2 = u2 % 2
                    val_u2 = U2[u2, r2]
                    
                    for u3 in nz_u3:
                        i3 = u3 // 3
                        j3 = u3 % 3
                        val_u3 = U3[u3, r3]
                        
                        i6 = i2 * 3 + i3
                        j6 = j2 * 3 + j3
                        g_u = i6 * 6 + j6
                        U6[g_u, r] = val_u2 * val_u3

                for v2 in nz_v2:
                    j2 = v2 // 2
                    k2 = v2 % 2
                    val_v2 = V2[v2, r2]
                    
                    for v3 in nz_v3:
                        j3 = v3 // 3
                        k3 = v3 % 3
                        val_v3 = V3[v3, r3]
                        
                        j6 = j2 * 3 + j3
                        k6 = k2 * 3 + k3
                        g_v = j6 * 6 + k6
                        V6[g_v, r] = val_v2 * val_v3

                for w2 in nz_w2:
                    k2 = w2 // 2
                    i2 = w2 % 2
                    val_w2 = W2[w2, r2]
                    
                    for w3 in nz_w3:
                        k3 = w3 // 3
                        i3 = w3 % 3
                        val_w3 = W3[w3, r3]
                        
                        k6 = k2 * 3 + k3
                        i6 = i2 * 3 + i3
                        g_w = k6 * 6 + i6
                        W6[g_w, r] = val_w2 * val_w3

        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}, rank

    # Try the new Kronecker 4x27 approach first (Rank 108)
    if time.time() < deadline:
        sol_kron, rank_kron = build_kronecker_4x27()
        if verify_decomposition(sol_kron['U'], sol_kron['V'], sol_kron['W']):
            return sol_kron

    # Fallback to Strassen 189
    sol, rank = build_strassen_2x2_blocks()
    return sol
# EVOLVE-BLOCK-END
