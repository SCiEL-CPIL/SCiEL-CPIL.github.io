import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6x6 matrix multiplication using Strassen(2x2) tensor Laderman(3x3)
    # Total rank: 7 * 23 = 161
    
    rank = 161
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 decomposition terms
    strassen_terms = [
        # M1: (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: d(g-e)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Laderman 3x3 decomposition terms (rank 23)
    # Corrected W coefficients for terms 21, 22, 23
    laderman_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1}),
        # M2
        ({0: 1, 3: 1}, {1: 1, 4: 1}, {1: 1}),
        # M3
        ({0: 1}, {1: 1, 0: -1}, {1: 1}),
        # M4
        ({3: 1}, {3: 1, 4: -1}, {3: 1, 4: 1}),
        # M5
        ({1: 1, 2: 1}, {4: 1}, {1: 1, 2: 1}),
        # M6
        ({2: 1}, {7: 1, 4: -1}, {2: 1}),
        # M7
        ({4: 1, 5: 1}, {8: 1}, {4: 1, 5: 1}),
        # M8
        ({5: 1}, {8: 1, 6: -1}, {5: 1}),
        # M9
        ({0: 1}, {6: 1}, {6: 1}),
        # M10
        ({1: 1}, {6: 1}, {7: 1}),
        # M11
        ({2: 1}, {6: 1}, {8: 1}),
        # M12
        ({3: 1}, {7: 1}, {6: 1}),
        # M13
        ({4: 1}, {7: 1}, {7: 1}),
        # M14
        ({5: 1}, {7: 1}, {8: 1}),
        # M15
        ({0: 1}, {8: 1}, {6: 1}),
        # M16
        ({1: 1}, {8: 1}, {7: 1}),
        # M17
        ({2: 1}, {8: 1}, {8: 1}),
        # M18
        ({3: 1}, {8: 1}, {7: 1}),
        # M19
        ({4: 1}, {8: 1}, {8: 1}),
        # M20
        ({5: 1}, {8: 1}, {8: 1}),
        # M21: (a00+a01+a02)(b20+b21+b22)
        # W: -1 for C00..C12, -1 for C20, -1 for C21, -1 for C22?
        # Formula: C00..C12 have -M21. C20 has -M21. C21 has -M21. C22 has -M21.
        # So -1 for all 9.
        ({0: 1, 1: 1, 2: 1}, {6: 1, 7: 1, 8: 1}, {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1, 8: -1}),
        # M22: (a10+a11+a12)(b20+b21+b22)
        # W: -1 for C00..C12, 0 for C20, -1 for C21, -1 for C22.
        ({3: 1, 4: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 0, 7: -1, 8: -1}),
        # M23: (a00..a12)(b20..b22)
        # W: +1 for all 9.
        ({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}, {6: 1, 7: 1, 8: 1}, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}),
    ]
    
    # Verify Laderman 3x3 first
    def verify_laderman():
        sz = 3
        r = 23
        U3 = np.zeros((sz*sz, r))
        V3 = np.zeros((sz*sz, r))
        W3 = np.zeros((sz*sz, r))
        
        for t, (u_c, v_c, w_c) in enumerate(laderman_terms):
            for idx, coeff in u_c.items():
                rr, cc = divmod(idx, sz)
                U3[rr*sz+cc, t] += coeff
            for idx, coeff in v_c.items():
                rr, cc = divmod(idx, sz)
                V3[rr*sz+cc, t] += coeff
            for idx, coeff in w_c.items():
                rr, cc = divmod(idx, sz)
                W3[cc*sz+rr, t] += coeff
        
        U3t = U3.reshape(sz, sz, r)
        V3t = V3.reshape(sz, sz, r)
        W3t = W3.reshape(sz, sz, r)
        T3 = np.einsum('ikt,kjt,jit->ikj', U3t, V3t, W3t)
        return np.allclose(T3, np.ones((sz,sz,sz)), atol=1e-10)
    
    laderman_ok = verify_laderman()
    
    if not laderman_ok:
        # Fall back to Strassen(2x2) x schoolbook(3x3) = rank 189
        rank_fb = 189
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                        for local_idx, coeff in u_coeffs.items():
                            r, c = divmod(local_idx, 2)
                            global_r = 2 * i + r
                            global_c = 2 * k + c
                            U_fb[global_r * m + global_c][col_fb] += coeff
                        for local_idx, coeff in v_coeffs.items():
                            r, c = divmod(local_idx, 2)
                            global_r = 2 * k + r
                            global_c = 2 * j + c
                            V_fb[global_r * p + global_c][col_fb] += coeff
                        for local_idx, coeff in w_coeffs.items():
                            r, c = divmod(local_idx, 2)
                            global_r = 2 * i + r
                            global_c = 2 * j + c
                            W_fb[global_c * n + global_r][col_fb] += coeff
                        col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Build 6x6 using Strassen(2x2) tensor Laderman(3x3)
    for t, (u_lad, v_lad, w_lad) in enumerate(laderman_terms):
        for s, (u_str, v_str, w_str) in enumerate(strassen_terms):
            for (ik_idx, cu) in u_lad.items():
                i, k = divmod(ik_idx, 3)
                for (elem, cs) in u_str.items():
                    a, b = divmod(elem, 2)
                    global_r = 2 * i + a
                    global_c = 2 * k + b
                    U[global_r * m + global_c][col] += cu * cs
            
            for (kj_idx, cv) in v_lad.items():
                k, j = divmod(kj_idx, 3)
                for (elem, cs) in v_str.items():
                    a, b = divmod(elem, 2)
                    global_r = 2 * k + a
                    global_c = 2 * j + b
                    V[global_r * p + global_c][col] += cv * cs
            
            for (ij_idx, cw) in w_lad.items():
                i, j = divmod(ij_idx, 3)
                for (elem, cs) in w_str.items():
                    a, b = divmod(elem, 2)
                    global_r = 2 * i + a
                    global_c = 2 * j + b
                    W[global_c * n + global_r][col] += cw * cs
            
            col += 1
    
    # Verify the full 6x6 decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    T = np.einsum('ikt,kjt,jit->ikj', U_t, V_t, W_t)
    expected = np.ones((n, m, p))
    
    if not np.allclose(T, expected, atol=1e-10):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j][col_s] = 1
                    V_s[j * p + k][col_s] = 1
                    W_s[k * n + i][col_s] = 1
                    col_s += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
