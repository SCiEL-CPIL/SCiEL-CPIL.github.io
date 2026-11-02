"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t_start = time.time()
    deadline = t_start + 110  # safety margin before 120s
    
    rank_terms = []  # list of (u_dict, v_dict, w_dict)
    
    # Part 1: 4x4 * 4x4 for columns k=0,1,2,3 using Strassen on 2x2 blocks (rank 28)
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B_first4 = [[B00, B01], [B10, B11]] each 2x2 (columns 0-3)
    # C_first4 = [[C00, C01], [C10, C11]] each 2x2
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):  # k1 = 0,1 for columns 0-3
                k_base = k1 * 2  # columns k_base, k_base+1
                
                terms = [
                    # M1: (X00+X11)(Y00+Y11)
                    ({(i1*2, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k_base): 1, (j1*2+1, k_base+1): 1},
                     {(i1*2, k_base): 1, (i1*2+1, k_base+1): 1}),
                    # M2: (X10+X11)Y00
                    ({(i1*2+1, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k_base): 1},
                     {(i1*2+1, k_base): 1, (i1*2+1, k_base+1): -1}),
                    # M3: X00(Y01-Y11)
                    ({(i1*2, j1*2): 1},
                     {(j1*2, k_base+1): 1, (j1*2+1, k_base+1): -1},
                     {(i1*2, k_base+1): 1, (i1*2+1, k_base+1): 1}),
                    # M4: X11(Y10-Y00)
                    ({(i1*2+1, j1*2+1): 1},
                     {(j1*2+1, k_base): 1, (j1*2, k_base): -1},
                     {(i1*2, k_base): 1, (i1*2+1, k_base): 1}),
                    # M5: (X00+X01)Y11
                    ({(i1*2, j1*2): 1, (i1*2, j1*2+1): 1},
                     {(j1*2+1, k_base+1): 1},
                     {(i1*2, k_base): -1, (i1*2, k_base+1): 1}),
                    # M6: (X10-X00)(Y00+Y01)
                    ({(i1*2+1, j1*2): 1, (i1*2, j1*2): -1},
                     {(j1*2, k_base): 1, (j1*2, k_base+1): 1},
                     {(i1*2+1, k_base+1): 1}),
                    # M7: (X01-X11)(Y10+Y11)
                    ({(i1*2, j1*2+1): 1, (i1*2+1, j1*2+1): -1},
                     {(j1*2+1, k_base): 1, (j1*2+1, k_base+1): 1},
                     {(i1*2, k_base): 1}),
                ]
                
                for u_dict, v_dict, w_dict in terms:
                    rank_terms.append((u_dict, v_dict, w_dict))
    
    # Part 2: 4x4 * 4x2 for columns k=4,5 using schoolbook (rank 32)
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                rank_terms.append(
                    ({(i, j): 1},
                     {(j, k): 1},
                     {(i, k): 1})
                )
    
    # Now build U, V, W matrices
    rank = len(rank_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][r] = coeff
    
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    for r in range(rank):
        u_vec = U_arr[:, r].reshape(n, m)
        v_vec = V_arr[:, r].reshape(m, p)
        w_vec = W_arr[:, r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
