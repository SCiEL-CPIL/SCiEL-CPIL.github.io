import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 4
    rank = 49

    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    # Convention: W[k*n + i, r] where C[i,k] = sum_r U[i*m+j? No, U is for A.
    # Task says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # A is n x m. Index in U: i*m + j.
    # B is m x p. Index in V: j*p + k.
    # C is p x n? No, C is n x p?
    # Standard matrix mult: C = A * B. A is n x m, B is m x p, C is n x p.
    # Task says: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies C is indexed by (i, k) where i is row (0..n-1) and k is col (0..p-1)?
    # Wait, W has shape (p*n, rank).
    # If W index is k*n + i, and k is 0..p-1, i is 0..n-1.
    # This suggests C[i, k] is stored at W[k*n + i].
    # So C is n x p.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Strassen 2x2 terms definition
    # Returns list of 7 tuples: (A_coeffs, B_coeffs, C_coeffs)
    # Each coeff is a dict: {(row, col): value}
    
    def get_strassen_2x2_terms():
        terms = []
        # t0: (A00+A11) * (B00+B11)
        terms.append({
            'A': {(0,0): 1, (1,1): 1},
            'B': {(0,0): 1, (1,1): 1},
            'C': {(0,0): 1, (1,1): 1}
        })
        # t1: (A10+A11) * B00
        terms.append({
            'A': {(1,0): 1, (1,1): 1},
            'B': {(0,0): 1},
            'C': {(1,0): 1, (1,1): -1}
        })
        # t2: A00 * (B01-B11)
        terms.append({
            'A': {(0,0): 1},
            'B': {(0,1): 1, (1,1): -1},
            'C': {(0,1): 1, (1,1): 1}
        })
        # t3: A11 * (B10-B00)
        terms.append({
            'A': {(1,1): 1},
            'B': {(1,0): 1, (0,0): -1},
            'C': {(0,0): 1, (1,0): 1}
        })
        # t4: (A00+A01) * B11
        terms.append({
            'A': {(0,0): 1, (0,1): 1},
            'B': {(1,1): 1},
            'C': {(0,0): -1, (0,1): 1}
        })
        # t5: (A10-A00) * (B00+B01)
        terms.append({
            'A': {(1,0): 1, (0,0): -1},
            'B': {(0,0): 1, (0,1): 1},
            'C': {(1,1): 1}
        })
        # t6: (A01-A11) * (B10+B11)
        terms.append({
            'A': {(0,1): 1, (1,1): -1},
            'B': {(1,0): 1, (1,1): 1},
            'C': {(0,0): 1}
        })
        return terms

    strassen_terms = get_strassen_2x2_terms()

    # Block-level Strassen for 4x4
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # We define the 7 block-level multiplications P_k
    # P_k = A_comb * B_comb
    # And how P_k contributes to C blocks.
    
    block_mults = [
        # P0 = (A00 + A11) * (B00 + B11)
        {
            'A_blocks': { (0,0): 1, (1,1): 1 },
            'B_blocks': { (0,0): 1, (1,1): 1 },
            'C_contrib': { (0,0): 1, (1,1): 1 }
        },
        # P1 = (A10 + A11) * B00
        {
            'A_blocks': { (1,0): 1, (1,1): 1 },
            'B_blocks': { (0,0): 1 },
            'C_contrib': { (1,0): 1, (1,1): -1 }
        },
        # P2 = A00 * (B01 - B11)
        {
            'A_blocks': { (0,0): 1 },
            'B_blocks': { (0,1): 1, (1,1): -1 },
            'C_contrib': { (0,1): 1, (1,1): 1 }
        },
        # P3 = A11 * (B10 - B00)
        {
            'A_blocks': { (1,1): 1 },
            'B_blocks': { (1,0): 1, (0,0): -1 },
            'C_contrib': { (0,0): 1, (1,0): 1 }
        },
        # P4 = (A00 + A01) * B11
        {
            'A_blocks': { (0,0): 1, (0,1): 1 },
            'B_blocks': { (1,1): 1 },
            'C_contrib': { (0,0): -1, (0,1): 1 }
        },
        # P5 = (A10 - A00) * (B00 + B01)
        {
            'A_blocks': { (1,0): 1, (0,0): -1 },
            'B_blocks': { (0,0): 1, (0,1): 1 },
            'C_contrib': { (1,1): 1 }
        },
        # P6 = (A01 - A11) * (B10 + B11)
        {
            'A_blocks': { (0,1): 1, (1,1): -1 },
            'B_blocks': { (1,0): 1, (1,1): 1 },
            'C_contrib': { (0,0): 1 }
        }
    ]

    rank_idx = 0
    for b_mult in block_mults:
        for s_term in strassen_terms:
            if time.time() - start_time > DEADLINE:
                return {"U": U, "V": V, "W": W}
            
            # Combine block and local terms
            # U elements
            U_elems = {}
            for (bi, bj), c_b in b_mult['A_blocks'].items():
                for (li, lj), c_l in s_term['A'].items():
                    gi = 2 * bi + li
                    gj = 2 * bj + lj
                    idx = u_idx(gi, gj)
                    U_elems[idx] = U_elems.get(idx, 0) + c_b * c_l
            
            # V elements
            V_elems = {}
            for (bi, bj), c_b in b_mult['B_blocks'].items():
                for (li, lj), c_l in s_term['B'].items():
                    gj = 2 * bi + li
                    gk = 2 * bj + lj
                    idx = v_idx(gj, gk)
                    V_elems[idx] = V_elems.get(idx, 0) + c_b * c_l
            
            # W elements
            W_elems = {}
            for (ci, cj), c_b in b_mult['C_contrib'].items():
                for (li, lj), c_l in s_term['C'].items():
                    gi = 2 * ci + li
                    gj = 2 * cj + lj
                    # W index is k*n + i. Here gj is col (k), gi is row (i).
                    idx = w_idx(gj, gi)
                    W_elems[idx] = W_elems.get(idx, 0) + c_b * c_l
            
            for idx, val in U_elems.items():
                if val != 0:
                    U[idx][rank_idx] = val
            for idx, val in V_elems.items():
                if val != 0:
                    V[idx][rank_idx] = val
            for idx, val in W_elems.items():
                if val != 0:
                    W[idx][rank_idx] = val
            
            rank_idx += 1

    return {"U": U, "V": V, "W": W}
