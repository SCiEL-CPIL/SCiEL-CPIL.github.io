"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    # A is n x m = 2x4, B is m x p = 4x4, C is n x p = 2x4
    # Output W follows trace convention: W[k*n+i, r]
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Strassen 2x2x2 terms
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # terms[s] = (U_coeffs, V_coeffs, W_coeffs)
    terms = [
        # 0: M1 = (x00+x11)(y00+y11) -> z00 += M1, z11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # 1: M2 = (x10+x11)y00 -> z10 += M2, z11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # 2: M3 = x00(y01-y11) -> z01 += M3, z11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # 3: M4 = x11(y10-y00) -> z00 += M4, z10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # 4: M5 = (x00+x01)y11 -> z00 -= M5, z01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # 5: M6 = (x10-x00)(y00+y01) -> z11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # 6: M7 = (x01-x11)(y10+y11) -> z00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block definitions:
    # Block 0: A_L (i=0,1, j=0,1) * B_LL (j=0,1, r=0,1) -> C_L (i=0,1, r=0,1)
    # Block 1: A_R (i=0,1, j=2,3) * B_RL (j=2,3, r=0,1) -> C_L (i=0,1, r=0,1)
    # Block 2: A_L (i=0,1, j=0,1) * B_LR (j=0,1, r=2,3) -> C_R (i=0,1, r=2,3)
    # Block 3: A_R (i=0,1, j=2,3) * B_RR (j=2,3, r=2,3) -> C_R (i=0,1, r=2,3)
    
    def get_global_A(block, li, lj):
        """Map local (li, lj) in 2x2 block to global A index (i*4+j)"""
        i = li
        if block in (0, 2):  # A_L: j = lj
            j = lj
        else:  # block 1, 3: A_R: j = 2 + lj
            j = 2 + lj
        return i * 4 + j
        
    def get_global_B(block, li, lj):
        """Map local (li, lj) in 2x2 block to global B index (j*4+r)"""
        # li is j_local, lj is r_local
        j_local = li
        r_local = lj
        if block in (0, 1):  # B_?L: r = r_local
            j = j_local
            r = r_local
        else:  # block 2, 3: B_?R: r = 2 + r_local
            j = j_local
            r = 2 + r_local
        if block in (1, 3):  # A_R blocks: j = 2 + j_local
            j = 2 + j_local
        return j * 4 + r
        
    def get_global_C(block, li, lj):
        """Map local (li, lj) in 2x2 block to global C index (r*2+i)"""
        # li is i_local, lj is r_local
        i_local = li
        r_local = lj
        if block in (0, 1):  # C_L: r = r_local
            i = i_local
            r = r_local
        else:  # block 2, 3: C_R: r = 2 + r_local
            i = i_local
            r = 2 + r_local
        return r * 2 + i

    # Iterate blocks
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_A(b, li, lj)
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_B(b, li, lj)
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_C(b, li, lj)
                W[g_idx][col] += coeff

    # Verify correctness with random test
    if time.time() < deadline - 5.0:
        try:
            rng = np.random.RandomState(42)
            A_test = rng.randn(n, m)
            B_test = rng.randn(m, p)
            C_expected = A_test @ B_test
            
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            
            C_computed = np.zeros(p * n)
            for r in range(rank):
                u_sum = np.dot(U_arr[:, r], A_test.flatten())
                v_sum = np.dot(V_arr[:, r], B_test.flatten())
                for c in range(p * n):
                    C_computed[c] += W_arr[c, r] * u_sum * v_sum
            
            max_err = 0.0
            for i in range(n):
                for k in range(p):
                    err = abs(C_expected[i, k] - C_computed[k * n + i])
                    max_err = max(max_err, err)
            
            if max_err > 1e-10:
                # Fallback: return trivial rank-32 decomposition
                rank = 32
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = i * m * p + j * p + k
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
