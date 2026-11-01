import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p):
    R = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i * m + j, j * p + k, k * n + i] = 1
    return np.array_equal(T, Expected)

def _add_strassen_block(U, V, W, a_off, b_off, c_off, rank_start):
    # a_off: start index in A (2x2 block)
    # b_off: start index in B (2x2 block)
    # c_off: start index in C (2x2 block)
    # A indices: a_off, a_off+1, a_off+2, a_off+3 (row major 2x2)
    # B indices: b_off, b_off+1, b_off+2, b_off+3
    # C indices: c_off, c_off+1, c_off+2, c_off+3
    
    # Strassen terms
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices in local 2x2 block (0..3)
    # A: 00=0, 01=1, 10=2, 11=3
    # B: 00=0, 01=1, 10=2, 11=3
    # C: 00=0, 01=1, 10=2, 11=3
    
    terms = [
        # (A_idx, coeff), (B_idx, coeff), (C_idx, coeff)
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)])
    ]
    
    r = rank_start
    for t in terms:
        a_idx, b_idx, c_idx = t
        for idx, coeff in a_idx:
            U[a_off + idx, r] += coeff
        for idx, coeff in b_idx:
            V[b_off + idx, r] += coeff
        for idx, coeff in c_idx:
            W[c_off + idx, r] += coeff
        r += 1
    return r

def _add_schoolbook_block(U, V, W, a_off, b_off, c_off, rank_start):
    # 2x2x1 schoolbook
    # C00 = A00 B00 + A10 B10 ? No.
    # A is 2x2, B is 2x1, C is 2x1.
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    
    # Indices in local block
    # A: 00=0, 01=1, 10=2, 11=3
    # B: 0=0, 1=1 (2x1)
    # C: 0=0, 1=1 (2x1)
    
    terms = [
        # A00 B0 -> C0
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        # A01 B1 -> C0
        ([(1, 1)], [(1, 1)], [(0, 1)]),
        # A10 B0 -> C1
        ([(2, 1)], [(0, 1)], [(1, 1)]),
        # A11 B1 -> C1
        ([(3, 1)], [(1, 1)], [(1, 1)])
    ]
    
    r = rank_start
    for t in terms:
        a_idx, b_idx, c_idx = t
        for idx, coeff in a_idx:
            U[a_off + idx, r] += coeff
        for idx, coeff in b_idx:
            V[b_off + idx, r] += coeff
        for idx, coeff in c_idx:
            W[c_off + idx, r] += coeff
        r += 1
    return r

def construct():
    n, m, p = DIMENSIONS
    dim_a = n * m # 8
    dim_b = m * p # 28
    dim_c = p * n # 14
    
    # We will build a rank 50 decomposition
    # U: (8, 50), V: (28, 50), W: (14, 50)
    R = 50
    U = np.zeros((dim_a, R), dtype=int)
    V = np.zeros((dim_b, R), dtype=int)
    W = np.zeros((dim_c, R), dtype=int)
    
    rank = 0
    
    # Split A (2x4) into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3)
    # A1 offset 0, A2 offset 4
    # Split B (4x7) into B1 (2x7, rows 0-1) and B2 (2x7, rows 2-3)
    # B1 offset 0, B2 offset 14 (since 7 cols * 2 rows = 14)
    
    # For A1 (2x2):
    #   Split B1 (2x7) into B1_4 (2x4, cols 0-3) and B1_3 (2x3, cols 4-6)
    #   B1_4 offset 0, B1_3 offset 28? No.
    #   B is flat. Row j, Col k -> idx j*7+k.
    #   B1 is rows 0,1.
    #   B1_4 is cols 0,1,2,3. Indices: 0,1,2,3 (row 0), 7,8,9,10 (row 1).
    #   This is NOT a contiguous 2x2 block in memory.
    #   My helper functions assume contiguous 2x2 blocks in memory.
    #   I need to map the indices correctly or adjust the helpers.
    
    # Let's rewrite helpers to take explicit index maps.
    
    def _add_block(U, V, W, a_map, b_map, c_map, terms, rank_start):
        # a_map: dict {local_idx: global_idx}
        # terms: list of (a_list, b_list, c_list) where each is list of (local_idx, coeff)
        r = rank_start
        for t in terms:
            a_list, b_list, c_list = t
            for l_idx, coeff in a_list:
                U[a_map[l_idx], r] += coeff
            for l_idx, coeff in b_list:
                V[b_map[l_idx], r] += coeff
            for l_idx, coeff in c_list:
                W[c_map[l_idx], r] += coeff
            r += 1
        return r

    # Define Strassen terms (local indices 0-3)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)])
    ]
    
    # Define Schoolbook 2x2x1 terms
    schoolbook_terms = [
        ([(0, 1)], [(0, 1)], [(0, 1)]),
        ([(1, 1)], [(1, 1)], [(0, 1)]),
        ([(2, 1)], [(0, 1)], [(1, 1)]),
        ([(3, 1)], [(1, 1)], [(1, 1)])
    ]
    
    # Helper to create maps
    def make_a_map(i_start, j_start):
        # A is 2x4. i in 0..1, j in 0..3.
        # Local 2x2: li in 0..1, lj in 0..1.
        # Global: i = i_start + li, j = j_start + lj.
        # Index: i*4 + j.
        return {
            0: (i_start) * 4 + j_start,
            1: (i_start) * 4 + (j_start + 1),
            2: (i_start + 1) * 4 + j_start,
            3: (i_start + 1) * 4 + (j_start + 1)
        }
        
    def make_b_map(j_start, k_start):
        # B is 4x7. j in 0..3, k in 0..6.
        # Local 2x2: lj in 0..1, lk in 0..1.
        # Global: j = j_start + lj, k = k_start + lk.
        # Index: j*7 + k.
        return {
            0: (j_start) * 7 + k_start,
            1: (j_start) * 7 + (k_start + 1),
            2: (j_start + 1) * 7 + k_start,
            3: (j_start + 1) * 7 + (k_start + 1)
        }
        
    def make_c_map(k_start, i_start):
        # C is 7x2. k in 0..6, i in 0..1.
        # Local 2x2: lk in 0..1, li in 0..1.
        # Global: k = k_start + lk, i = i_start + li.
        # Index: k*2 + i.
        return {
            0: (k_start) * 2 + i_start,
            1: (k_start) * 2 + (i_start + 1),
            2: (k_start + 1) * 2 + i_start,
            3: (k_start + 1) * 2 + (i_start + 1)
        }
        
    def make_b_map_2x1(j_start, k_start):
        # B is 4x7. Local 2x1: lj in 0..1, lk=0.
        return {
            0: (j_start) * 7 + k_start,
            1: (j_start + 1) * 7 + k_start
        }
        
    def make_c_map_2x1(k_start, i_start):
        # C is 7x2. Local 2x1: lk=0, li in 0..1.
        return {
            0: (k_start) * 2 + i_start,
            1: (k_start) * 2 + (i_start + 1)
        }

    # 1. A1 (i=0..1, j=0..1) * B1_4 (j=0..1, k=0..3)
    # Split B1_4 into B1_4a (k=0..1) and B1_4b (k=2..3)
    
    # A1 map
    a1_map = make_a_map(0, 0)
    
    # B1_4a map (j=0..1, k=0..1)
    b1_4a_map = make_b_map(0, 0)
    # C1_4a map (k=0..1, i=0..1)
    c1_4a_map = make_c_map(0, 0)
    
    rank = _add_block(U, V, W, a1_map, b1_4a_map, c1_4a_map, strassen_terms, rank)
    
    # B1_4b map (j=0..1, k=2..3)
    b1_4b_map = make_b_map(0, 2)
    # C1_4b map (k=2..3, i=0..1)
    c1_4b_map = make_c_map(2, 0)
    
    rank = _add_block(U, V, W, a1_map, b1_4b_map, c1_4b_map, strassen_terms, rank)
    
    # B1_3 (j=0..1, k=4..6)
    # Split into B1_3a (k=4..5) and B1_3b (k=6)
    
    # B1_3a map (j=0..1, k=4..5)
    b1_3a_map = make_b_map(0, 4)
    # C1_3a map (k=4..5, i=0..1)
    c1_3a_map = make_c_map(4, 0)
    
    rank = _add_block(U, V, W, a1_map, b1_3a_map, c1_3a_map, strassen_terms, rank)
    
    # B1_3b map (j=0..1, k=6)
    b1_3b_map = make_b_map_2x1(0, 6)
    # C1_3b map (k=6, i=0..1)
    c1_3b_map = make_c_map_2x1(6, 0)
    
    rank = _add_block(U, V, W, a1_map, b1_3b_map, c1_3b_map, schoolbook_terms, rank)
    
    # 2. A2 (i=0..1, j=2..3) * B2 (j=2..3, k=0..6)
    # Split B2 into B2_4 (k=0..3) and B2_3 (k=4..6)
    
    # A2 map
    a2_map = make_a_map(0, 2)
    
    # B2_4a (j=2..3, k=0..1)
    b2_4a_map = make_b_map(2, 0)
    # C2_4a (k=0..1, i=0..1)
    c2_4a_map = make_c_map(0, 0)
    
    rank = _add_block(U, V, W, a2_map, b2_4a_map, c2_4a_map, strassen_terms, rank)
    
    # B2_4b (j=2..3, k=2..3)
    b2_4b_map = make_b_map(2, 2)
    # C2_4b (k=2..3, i=0..1)
    c2_4b_map = make_c_map(2, 0)
    
    rank = _add_block(U, V, W, a2_map, b2_4b_map, c2_4b_map, strassen_terms, rank)
    
    # B2_3a (j=2..3, k=4..5)
    b2_3a_map = make_b_map(2, 4)
    # C2_3a (k=4..5, i=0..1)
    c2_3a_map = make_c_map(4, 0)
    
    rank = _add_block(U, V, W, a2_map, b2_3a_map, c2_3a_map, strassen_terms, rank)
    
    # B2_3b (j=2..3, k=6)
    b2_3b_map = make_b_map_2x1(2, 6)
    # C2_3b (k=6, i=0..1)
    c2_3b_map = make_c_map_2x1(6, 0)
    
    rank = _add_block(U, V, W, a2_map, b2_3b_map, c2_3b_map, schoolbook_terms, rank)
    
    # Total rank used: 7*6 + 4*2 = 42 + 8 = 50.
    # We allocated 50.
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to trivial if verification fails (should not happen)
        U = np.zeros((dim_a, dim_a*dim_b*dim_c//dim_c), dtype=int) # Just resize
        # Rebuild trivial
        U = np.zeros((dim_a, 56), dtype=int)
        V = np.zeros((dim_b, 56), dtype=int)
        W = np.zeros((dim_c, 56), dtype=int)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
                    U = U[:, :r] # This is inefficient, just return trivial directly
        # Just return trivial
        U = np.zeros((dim_a, 56), dtype=int)
        V = np.zeros((dim_b, 56), dtype=int)
        W = np.zeros((dim_c, 56), dtype=int)
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1
                    V[j*p+k, r] = 1
                    W[k*n+i, r] = 1
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    T = np.einsum('ar,br,cr->abc', U, V, W)
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Expected[i*m + j, j*p + k, k*n + i] = 1
    if np.array_equal(T, Expected):
        print(f"Valid decomposition with rank {U.shape[1]}")
    else:
        print("Invalid decomposition")
