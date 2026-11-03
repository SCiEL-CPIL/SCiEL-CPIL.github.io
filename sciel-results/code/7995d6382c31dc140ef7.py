import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # 1. Build 2x2 Strassen factors (U2, V2, W2) of shape (4, 7)
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # Term 0: S0 = (P00+P11)(Q00+Q11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: S1 = (P10+P11)Q00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # Term 2: S2 = P00(Q01-Q11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1
    
    # Term 3: S3 = P11(Q10-Q00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 4: S4 = (P00+P01)Q11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 5: S5 = (P10-P00)(Q00+Q01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6: S6 = (P01-P11)(Q10+Q11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = -1
    
    # 2. Define Block-Level Strassen terms
    # Each term t has P_blocks, Q_blocks, R_blocks
    # Blocks are identified by (u, v) in {0,1}x{0,1}
    # P_blocks: list of ((u,v), coeff)
    
    block_terms = [
        # t=0: M0 = (A00+A11)(B00+B11)
        (
            [((0,0), 1), ((1,1), 1)],
            [((0,0), 1), ((1,1), 1)],
            [((0,0), 1), ((1,1), 1)]
        ),
        # t=1: M1 = (A10+A11)B00
        (
            [((1,0), 1), ((1,1), 1)],
            [((0,0), 1)],
            [((1,0), 1)]
        ),
        # t=2: M2 = A00(B01-B11)
        (
            [((0,0), 1)],
            [((0,1), 1), ((1,1), -1)],
            [((0,1), 1), ((1,1), 1)]
        ),
        # t=3: M3 = A11(B10-B00)
        (
            [((1,1), 1)],
            [((1,0), 1), ((0,0), -1)],
            [((0,0), 1), ((1,0), 1)]
        ),
        # t=4: M4 = (A00+A01)B11
        (
            [((0,0), 1), ((0,1), 1)],
            [((1,1), 1)],
            [((0,0), -1), ((0,1), 1)]
        ),
        # t=5: M5 = (A10-A00)(B00+B01)
        (
            [((1,0), 1), ((0,0), -1)],
            [((0,0), 1), ((0,1), 1)],
            [((1,1), 1)]
        ),
        # t=6: M6 = (A01-A11)(B10+B11)
        (
            [((0,1), 1), ((1,1), -1)],
            [((1,0), 1), ((1,1), 1)],
            [((0,0), 1), ((1,1), -1)]
        ),
    ]
    
    # 3. Construct U, V, W
    for t in range(7):
        P_blocks, Q_blocks, R_blocks = block_terms[t]
        for s in range(7):
            r = t * 7 + s
            
            # U contributions
            for i in range(4):
                coeff_U2 = U2[i, s]
                if coeff_U2 == 0:
                    continue
                e = i // 2
                b = i % 2
                for (u, v, alpha) in P_blocks:
                    I = (2*u + e) * 4 + (2*v + b)
                    U[I, r] += coeff_U2 * alpha
            
            # V contributions
            for j in range(4):
                coeff_V2 = V2[j, s]
                if coeff_V2 == 0:
                    continue
                b = j // 2
                f = j % 2
                for (u, v, beta) in Q_blocks:
                    J = (2*u + b) * 4 + (2*v + f)
                    V[J, r] += coeff_V2 * beta
            
            # W contributions
            for k in range(4):
                coeff_W2 = W2[k, s]
                if coeff_W2 == 0:
                    continue
                e = k // 2
                f = k % 2
                for (u, v, gamma) in R_blocks:
                    K = (2*u + e) * 4 + (2*v + f)
                    W[K, r] += coeff_W2 * gamma
    
    # 4. Verify
    # Check: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] == 1
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U[i*4+j, :] * V[j*4+k, :] * W[k*4+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails (should not happen)
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*4+k, col] = 1
                    W_fb[k*4+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
