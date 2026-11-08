import numpy as np
import time

def construct():
    """
    Construct rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Strategy:
    - Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    - Use Strassen (rank 49) for A*B1 and A*B2
    - Use Schoolbook (rank 16) for A*B3
    - Total rank: 49 + 49 + 16 = 114
    """
    n, m, p = 4, 4, 9
    U_parts = []
    V_parts = []
    W_parts = []

    def add_term(u, v, w):
        U_parts.append(u)
        V_parts.append(v)
        W_parts.append(w)

    # Strassen 2x2 terms
    # Each term: (A_indices, B_indices, C_indices) with coefficients
    s2x2_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def generate_strassen_4x4_terms(a_base_i, a_base_j, b_base_j, c_base_i, c_base_k):
        terms = []
        
        for meta_a, meta_b, meta_c in s2x2_terms:
            for inner_a, inner_b, inner_c in s2x2_terms:
                u = np.zeros(16, dtype=int)
                v = np.zeros(36, dtype=int)
                w = np.zeros(36, dtype=int)
                
                # Map inner_a (2x2 on 2x2 blocks) to global A (4x4)
                for (br, bc), coeff_a in meta_a.items():
                    for (ir, ic), coeff_inner in inner_a.items():
                        gi = a_base_i + br * 2 + ir
                        gj = a_base_j + bc * 2 + ic
                        u[gi * 4 + gj] += coeff_a * coeff_inner

                for (br, bc), coeff_b in meta_b.items():
                    for (ir, ic), coeff_inner in inner_b.items():
                        bi = br * 2 + ir
                        bj = b_base_j + bc * 2 + ic
                        v[bi * 9 + bj] += coeff_b * coeff_inner

                for (br, bc), coeff_c in meta_c.items():
                    for (ir, ic), coeff_inner in inner_c.items():
                        ci = c_base_i + br * 2 + ir
                        cj = c_base_k + bc * 2 + ic
                        w[cj * 4 + ci] += coeff_c * coeff_inner
                        
                terms.append((u.tolist(), v.tolist(), w.tolist()))
        return terms

    # Add Strassen terms for first 4x4 block of B (cols 0-3)
    for u, v, w in generate_strassen_4x4_terms(0, 0, 0, 0, 0):
        add_term(u, v, w)

    # Add Strassen terms for second 4x4 block of B (cols 4-7)
    for u, v, w in generate_strassen_4x4_terms(0, 0, 4, 0, 4):
        add_term(u, v, w)

    # Add schoolbook terms for third block of B (col 8)
    for i in range(4):
        for j in range(4):
            u = np.zeros(16, dtype=int)
            v = np.zeros(36, dtype=int)
            w = np.zeros(36, dtype=int)
            u[i * 4 + j] = 1
            v[j * 9 + 8] = 1
            w[8 * 4 + i] = 1
            add_term(u.tolist(), v.tolist(), w.tolist())

    rank = len(U_parts)
    
    # Construct final matrices
    U = np.array(U_parts).T if U_parts else np.array([])
    V = np.array(V_parts).T if V_parts else np.array([])
    W = np.array(W_parts).T if W_parts else np.array([])
    
    U_out = U.tolist() if U.size > 0 else []
    V_out = V.tolist() if V.size > 0 else []
    W_out = W.tolist() if W.size > 0 else []
    
    return {"U": U_out, "V": V_out, "W": W_out}
