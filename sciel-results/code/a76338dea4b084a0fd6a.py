import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    n, m, p = 2, 2, 11
    rank = 39  # 5 Strassen blocks (7 each) + 1 schoolbook column (4)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    r = 0

    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]

        # Strassen terms for 2x2 * 2x2
        # m1 = (a+d)(e+h)
        U[0][r] += 1; U[3][r] += 1
        V[v_e][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1; W[w_c11][r] += 1
        r += 1

        # m2 = (c+d)e
        U[2][r] += 1; U[3][r] += 1
        V[v_e][r] += 1
        W[w_c10][r] += 1; W[w_c11][r] += -1
        r += 1

        # m3 = a(f-h)
        U[0][r] += 1
        V[v_f][r] += 1; V[v_h][r] += -1
        W[w_c01][r] += 1; W[w_c11][r] += 1
        r += 1

        # m4 = d(g-e)
        U[3][r] += 1
        V[v_g][r] += 1; V[v_e][r] += -1
        W[w_c00][r] += 1; W[w_c10][r] += 1
        r += 1

        # m5 = (a+b)h
        U[0][r] += 1; U[1][r] += 1
        V[v_h][r] += 1
        W[w_c00][r] += -1; W[w_c01][r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2][r] += 1; U[0][r] += -1
        V[v_e][r] += 1; V[v_f][r] += 1
        W[w_c11][r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1][r] += 1; U[3][r] += -1
        V[v_g][r] += 1; V[v_h][r] += 1
        W[w_c00][r] += 1
        r += 1

        if time.time() > deadline:
            break

    # Last column k=10: schoolbook
    if r < rank:
        k2 = 10
        v_e2 = k2      # B[0, 10]
        v_g2 = 11 + k2 # B[1, 10]
        w_c02 = 2 * k2      # C[0, 10]
        w_c12 = 2 * k2 + 1  # C[1, 10]

        # a * B[0, 10]
        U[0][r] += 1
        V[v_e2][r] += 1
        W[w_c02][r] += 1
        r += 1

        # b * B[1, 10]
        U[1][r] += 1
        V[v_g2][r] += 1
        W[w_c02][r] += 1
        r += 1

        # c * B[0, 10]
        U[2][r] += 1
        V[v_e2][r] += 1
        W[w_c12][r] += 1
        r += 1

        # d * B[1, 10]
        U[3][r] += 1
        V[v_g2][r] += 1
        W[w_c12][r] += 1
        r += 1

    # Fast verification using numpy (vectorized)
    if time.time() < deadline:
        try:
            Ua = np.array(U, dtype=np.float64)  # (4, rank)
            Va = np.array(V, dtype=np.float64)  # (22, rank)
            Wa = np.array(W, dtype=np.float64)  # (22, rank)
            
            # Verify: for random A (2x2), B (2x11), check A@B matches tensor decomposition
            # C[i,k] = sum_j sum_r U[i*2+j, r] * V[j*11+k, r] * W[k*2+i, r]
            np.random.seed(123)
            A_test = np.random.randn(2, 2)
            B_test = np.random.randn(2, 11)
            C_expected = A_test @ B_test
            
            # Vectorized computation:
            # For each rank term r: outer product structure
            # C[i,k] = sum_r Wa[k*2+i, r] * (sum_j Ua[i*2+j, r] * Va[j*11+k, r])
            # Let M[i,k,r] = sum_j Ua[i*2+j, r] * Va[j*11+k, r]
            # Then C[i,k] = sum_r Wa[k*2+i, r] * M[i,k,r]
            
            # M[i,k,r]: for each i (2), k (11), r (rank), sum over j (2)
            # Ua reshaped: Ua[i*2+j, r] -> shape (2, 2, rank) where dims are (i, j, r)
            U_reshaped = Ua.reshape(2, 2, rank)  # (i, j, r)
            # Va reshaped: Va[j*11+k, r] -> shape (2, 11, rank) where dims are (j, k, r)
            V_reshaped = Va.reshape(2, 11, rank)  # (j, k, r)
            
            # M[i,k,r] = sum_j U_reshaped[i,j,r] * V_reshaped[j,k,r]
            # = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)
            M = np.einsum('ijr,jkr->ikr', U_reshaped, V_reshaped)  # (2, 11, rank)
            
            # Wa reshaped: Wa[k*2+i, r] -> shape (11, 2, rank) where dims are (k, i, r)
            W_reshaped = Wa.reshape(11, 2, rank)  # (k, i, r)
            
            # C[i,k] = sum_r M[i,k,r] * W_reshaped[k,i,r]
            C_decomp = np.einsum('ikr,kir->ik', M, W_reshaped)  # (2, 11)
            
            if np.max(np.abs(C_expected - C_decomp)) > 1e-10:
                # Fallback: return schoolbook rank-44 decomposition (always valid)
                U_fb = [[0] * 44 for _ in range(4)]
                V_fb = [[0] * 44 for _ in range(22)]
                W_fb = [[0] * 44 for _ in range(22)]
                r_fb = 0
                for i in range(2):
                    for j in range(2):
                        for k in range(11):
                            U_fb[i*2+j][r_fb] = 1
                            V_fb[j*11+k][r_fb] = 1
                            W_fb[k*2+i][r_fb] = 1
                            r_fb += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
