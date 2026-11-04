import numpy as np

def construct():
    n, m, p = 2, 5, 5
    
    # Define Strassen rank-7 decomposition for 2x2x2
    # Inputs: A[2x2], B[2x2], Output: C[2x2]
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A10)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For a 2x2x2 block with local indices (i, j, k) where i,j,k in {0,1}
    # U[i*2+j, col], V[j*2+k, col], W[k*2+i, col]
    
    def get_strassen_terms(i_base, j_base, k_base):
        """Get the 7 Strassen terms for a 2x2x2 block starting at (i_base, j_base, k_base)."""
        terms = []
        # Each term is (u_coeff, v_coeff, w_coeff) where coeff is a dict from local (idx) to value
        # M1 = (A00+A11)(B00+B11)
        terms.append({
            'u': {(0,0): 1, (1,1): 1},
            'v': {(0,0): 1, (1,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        })
        # M2 = (A10+A11)(B00)
        terms.append({
            'u': {(1,0): 1, (1,1): 1},
            'v': {(0,0): 1},
            'w': {(0,0): 1, (1,0): 1}
        })
        # M3 = (A00)(B01-B11)
        terms.append({
            'u': {(0,0): 1},
            'v': {(0,1): 1, (1,1): -1},
            'w': {(0,1): 1, (1,1): 1}
        })
        # M4 = (A11)(B10-B00)
        terms.append({
            'u': {(1,1): 1},
            'v': {(1,0): 1, (0,0): -1},
            'w': {(1,0): 1, (0,0): 1}
        })
        # M5 = (A00+A10)(B11)
        terms.append({
            'u': {(0,0): 1, (1,0): 1},
            'v': {(1,1): 1},
            'w': {(1,1): 1, (0,1): 1}
        })
        # M6 = (A10-A00)(B00+B01)
        terms.append({
            'u': {(1,0): 1, (0,0): -1},
            'v': {(0,0): 1, (0,1): 1},
            'w': {(0,0): 1, (1,1): 1}
        })
        # M7 = (A01-A11)(B10+B11)
        terms.append({
            'u': {(0,1): 1, (1,1): -1},
            'v': {(1,0): 1, (1,1): 1},
            'w': {(1,1): 1, (0,0): 1}
        })
        
        # Convert to global indices
        global_terms = []
        for term in terms:
            u_global = {}
            v_global = {}
            w_global = {}
            for (li, lj), val in term['u'].items():
                gi, gj = i_base + li, j_base + lj
                u_global[gi * m + gj] = val
            for (lj, lk), val in term['v'].items():
                gj, gk = j_base + lj, k_base + lk
                v_global[gj * p + gk] = val
            for (lk, li), val in term['w'].items():
                gk, gi = k_base + lk, i_base + li
                w_global[gk * n + gi] = val
            global_terms.append((u_global, v_global, w_global))
        return global_terms
    
    # Define trivial 1x1x1 term
    def get_trivial_term(i_idx, j_idx, k_idx):
        return [({i_idx * m + j_idx: 1}, {j_idx * p + k_idx: 1}, {k_idx * n + i_idx: 1})]
    
    # Define trivial 2x1x1 terms (for j-group of size 2, k-group of size 1)
    def get_trivial_2x1x1(i_base, j_base, k_idx):
        """2x1x1 block: i in {i_base, i_base+1}, j in {j_base, j_base+1}, k = k_idx"""
        terms = []
        for di in range(2):
            for dj in range(2):
                i_idx = i_base + di
                j_idx = j_base + dj
                terms.append((
                    {i_idx * m + j_idx: 1},
                    {j_idx * p + k_idx: 1},
                    {k_idx * n + i_idx: 1}
                ))
        return terms
    
    # Define trivial 1x2x1 terms (for j-group of size 1, k-group of size 2)
    def get_trivial_1x2x1(i_base, j_idx, k_base):
        """1x2x1 block: i in {i_base, i_base+1}, j = j_idx, k in {k_base, k_base+1}"""
        terms = []
        for di in range(2):
            for dk in range(2):
                i_idx = i_base + di
                k_idx = k_base + dk
                terms.append((
                    {i_idx * m + j_idx: 1},
                    {j_idx * p + k_idx: 1},
                    {k_idx * n + i_idx: 1}
                ))
        return terms
    
    # Collect all terms
    all_terms = []
    
    # 4 Strassen blocks (2x2x2)
    # (J0, K0): j in {0,1}, k in {0,1}
    all_terms.extend(get_strassen_terms(0, 0, 0))
    # (J0, K1): j in {0,1}, k in {2,3}
    all_terms.extend(get_strassen_terms(0, 0, 2))
    # (J1, K0): j in {2,3}, k in {0,1}
    all_terms.extend(get_strassen_terms(0, 2, 0))
    # (J1, K1): j in {2,3}, k in {2,3}
    all_terms.extend(get_strassen_terms(0, 2, 2))
    
    # 4 trivial 2x2x1 blocks (j-group size 2, k-group size 1)
    # (J0, K2): j in {0,1}, k = 4
    all_terms.extend(get_trivial_2x1x1(0, 0, 4))
    # (J1, K2): j in {2,3}, k = 4
    all_terms.extend(get_trivial_2x1x1(0, 2, 4))
    
    # 4 trivial 2x1x2 blocks (j-group size 1, k-group size 2)
    # (J2, K0): j = 4, k in {0,1}
    all_terms.extend(get_trivial_1x2x1(0, 4, 0))
    # (J2, K1): j = 4, k in {2,3}
    all_terms.extend(get_trivial_1x2x1(0, 4, 2))
    
    # 1 trivial 2x1x1 block
    # (J2, K2): j = 4, k = 4
    all_terms.extend(get_trivial_term(0, 4, 4))
    all_terms.extend(get_trivial_term(1, 4, 4))
    
    rank = len(all_terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(all_terms):
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
