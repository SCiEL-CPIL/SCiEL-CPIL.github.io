import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140

    def strassen_2x2_terms():
        terms = []
        terms.append((np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])))
        terms.append((np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0])))
        terms.append((np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1]), np.array([0, 1, 0, 0])))
        terms.append((np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0]), np.array([0, 1, 0, -1])))
        terms.append((np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1])))
        terms.append((np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([-1, 0, 0, 0])))
        terms.append((np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0])))
        return terms

    def build_decomposition():
        s2 = strassen_2x2_terms()
        rank = 53

        U_full = np.zeros((n * m, rank))
        V_full = np.zeros((m * p, rank))
        W_full = np.zeros((p * n, rank))

        # First 49 columns: double Strassen for B[:, 0:4] -> C[:, 0:4]
        for oi in range(7):
            for ii in range(7):
                col = oi * 7 + ii
                u_o, v_o, w_o = s2[oi]
                u_i, v_i, w_i = s2[ii]
                
                Uo = u_o.reshape(2, 2)
                Ui = u_i.reshape(2, 2)
                Vo = v_o.reshape(2, 2)
                Vi = v_i.reshape(2, 2)
                Wo = w_o.reshape(2, 2)
                Wi = w_i.reshape(2, 2)
                
                for a in range(4):
                    for b in range(4):
                        ar, ai = a // 2, a % 2
                        bc, bi = b // 2, b % 2
                        U_full[a * m + b, col] = Uo[ar, bc] * Ui[ai, bi]
                
                for j in range(4):
                    for k in range(4):
                        jr, ji = j // 2, j % 2
                        kc, ki = k // 2, k % 2
                        V_full[j * p + k, col] = Vo[jr, kc] * Vi[ji, ki]
                
                for k in range(4):
                    for i in range(4):
                        kr, ki_ = k // 2, k % 2
                        ic, ii_ = i // 2, i % 2
                        W_full[k * n + i, col] = Wo[kr, ic] * Wi[ki_, ii_]

        # Next 4 columns: rank-4 decomposition for the 5th column (B[:, 4] -> C[:, 4])
        for j in range(m):
            col = 49 + j
            # U: column j of A has 1s at positions (i*m + j) for i=0..3
            for i in range(n):
                U_full[i * m + j, col] = 1
            # V: B_j4 is at position j*p + 4
            V_full[j * p + 4, col] = 1
            # W: C_i4 is at position 4*n + i for i=0..3
            for i in range(n):
                W_full[4 * n + i, col] = 1

        return {'U': U_full.tolist(), 'V': V_full.tolist(), 'W': W_full.tolist()}

    def verify(U, V, W, num_tests=10):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            C_decomp = np.zeros((n, p))
            for r in range(rank):
                u_r = U[:, r].reshape(n, m)
                v_r = V[:, r].reshape(m, p)
                w_r = W[:, r].reshape(p, n)
                C_decomp += (A * u_r) @ (B * v_r) * w_r.T
            
            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                return False
        return True

    result = build_decomposition()
    if verify(result['U'], result['V'], result['W']):
        return result
    
    # Fallback: Schoolbook (rank 80)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
