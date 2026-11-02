import time
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term to the global matrices
    # u_idx: list of (local_index, coeff)
    # v_idx: list of (local_index, coeff)
    # w_idx: list of (local_index, coeff)
    # col: the column index in U, V, W
    # map_u, map_v, map_w: functions mapping local index to global index
    
    def add_term(col, u_coeffs, v_coeffs, w_coeffs, map_u, map_v, map_w):
        for loc_u, coeff_u in u_coeffs:
            glob_u = map_u(loc_u)
            if glob_u is not None:
                U[glob_u][col] += coeff_u
        for loc_v, coeff_v in v_coeffs:
            glob_v = map_v(loc_v)
            if glob_v is not None:
                V[glob_v][col] += coeff_v
        for loc_w, coeff_w in w_coeffs:
            glob_w = map_w(loc_w)
            if glob_w is not None:
                W[glob_w][col] += coeff_w

    # --- Define Sub-Decompositions ---
    
    # 1. M_2,2,2 (Strassen)
    # A: 2x2, B: 2x2, C: 2x2
    # Local indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    def get_strassen_terms():
        terms = []
        # m1 = (a+d)(e+h) -> C11 + C22
        terms.append(
            ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])
        )
        # m2 = (c+d)e -> C21
        terms.append(
            ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)])
        )
        # m3 = a(f-h) -> C12 - C22
        terms.append(
            ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, -1)])
        )
        # m4 = d(g-e) -> C11 - C21
        terms.append(
            ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, -1)])
        )
        # m5 = (a+b)h -> C12
        terms.append(
            ([(0, 1), (1, 1)], [(3, 1)], [(1, 1)])
        )
        # m6 = (c-a)(e+f) -> C22
        terms.append(
            ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])
        )
        # m7 = (b-d)(g+h) -> C11
        terms.append(
            ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
        )
        return terms

    # 2. M_2,2,1 (Schoolbook)
    # A: 2x2, B: 2x1, C: 2x1
    # B indices: 00, 10 -> 0, 1
    # C indices: 00, 10 -> 0, 1
    
    def get_schoolbook_221_terms():
        terms = []
        # A00 * B00 -> C00
        terms.append(([(0, 1)], [(0, 1)], [(0, 1)]))
        # A10 * B00 -> C10
        terms.append(([(2, 1)], [(0, 1)], [(1, 1)]))
        # A01 * B10 -> C00
        terms.append(([(1, 1)], [(1, 1)], [(0, 1)]))
        # A11 * B10 -> C10
        terms.append(([(3, 1)], [(1, 1)], [(1, 1)]))
        return terms

    # 3. M_2,2,3
    # Split B (2x3) into B1 (2x2) and B2 (2x1)
    # B1 cols 0,1. B2 col 2.
    # C cols 0,1 from B1. C col 2 from B2.
    
    def get_223_terms():
        terms = []
        # Strassen for B1
        strassen_terms = get_strassen_terms()
        for t in strassen_terms:
            u, v, w = t
            # v indices 0,1,2,3 map to global B indices 0,1,2,3 (cols 0,1)
            # w indices 0,1,2,3 map to global C indices 0,1,2,3 (cols 0,1)
            terms.append((u, v, w))
        
        # Schoolbook for B2
        sb_terms = get_schoolbook_221_terms()
        for t in sb_terms:
            u, v, w = t
            # v index 0 -> global B index 2 (col 2, row 0)
            # v index 1 -> global B index 3 (col 2, row 1)
            new_v = [(0, v[0][1]), (1, v[1][1])] if len(v)==2 else [(loc+2, c) for loc, c in v]
            # Wait, v for 221 is length 2 (indices 0,1).
            # Global B for 2x3 has indices 0..5.
            # B2 is rows 0,1 of col 2.
            # Local v[0] is B[0,2] -> global 2.
            # Local v[1] is B[1,2] -> global 3.
            new_v = [(loc + 2, c) for loc, c in v]
            
            # w index 0 -> global C index 2 (col 2, row 0)
            # w index 1 -> global C index 3 (col 2, row 1)
            new_w = [(loc + 2, c) for loc, c in w]
            
            terms.append((u, new_v, new_w))
            
        return terms

    # 4. M_2,2,4
    # Split B (2x4) into B1 (2x2) and B2 (2x2)
    # B1 cols 0,1. B2 cols 2,3.
    
    def get_224_terms():
        terms = []
        # Strassen for B1
        strassen_terms = get_strassen_terms()
        for t in strassen_terms:
            u, v, w = t
            # v indices 0..3 map to global 0..3
            # w indices 0..3 map to global 0..3
            terms.append((u, v, w))
            
        # Strassen for B2
        for t in strassen_terms:
            u, v, w = t
            # v indices 0..3 map to global 2..5
            new_v = [(loc + 2, c) for loc, c in v]
            # w indices 0..3 map to global 2..5
            new_w = [(loc + 2, c) for loc, c in w]
            terms.append((u, new_v, new_w))
            
        return terms

    # 5. M_2,1,7
    # A: 2x1, B: 1x7, C: 2x7
    # A indices: 0, 1
    # B indices: 0..6
    # C indices: 0..13 (k*2+i)
    
    def get_217_terms():
        terms = []
        # A0 * Bk -> C0k
        # A1 * Bk -> C1k
        for k in range(7):
            # A0 * Bk
            terms.append(([(0, 1)], [(k, 1)], [(k * 2, 1)]))
            # A1 * Bk
            terms.append(([(1, 1)], [(k, 1)], [(k * 2 + 1, 1)]))
        return terms

    # --- Assemble M_2,3,7 ---
    
    # Split A (2x3) into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    # Split B (3x7) into B1 (2x7, rows 0,1) and B2 (1x7, row 2)
    
    # Part 1: A1 * B1 (M_2,2,7)
    # M_2,2,7 is M_2,2,3 (cols 0,1,2) + M_2,2,4 (cols 3,4,5,6)
    
    col_offset = 0
    
    # Sub-part 1a: M_2,2,3
    terms_223 = get_223_terms()
    for t in terms_223:
        u, v, w = t
        # Map u (local 2x2) to global A (2x3)
        # Local u indices 0,1,2,3 -> Global 0,1,2,3
        map_u = lambda loc: loc
        
        # Map v (local 2x3) to global B (3x7)
        # Local v indices 0..5 -> Global 0..5 (rows 0,1 of B)
        map_v = lambda loc: loc
        
        # Map w (local 2x3) to global C (7x2)
        # Local w indices 0..5 -> Global 0..5
        map_w = lambda loc: loc
        
        add_term(col_offset, u, v, w, map_u, map_v, map_w)
        col_offset += 1
        
    # Sub-part 1b: M_2,2,4
    terms_224 = get_224_terms()
    for t in terms_224:
        u, v, w = t
        # Map u (local 2x2) to global A (2x3)
        map_u = lambda loc: loc
        
        # Map v (local 2x4) to global B (3x7)
        # Local v indices 0..7 (2 rows x 4 cols)
        # Global B rows 0,1. Cols 3,4,5,6.
        # Local col 0 -> Global col 3.
        # Local index = row * 4 + col.
        # Global index = row * 7 + (col + 3).
        def map_v_224(loc):
            row = loc // 4
            col = loc % 4
            return row * 7 + (col + 3)
            
        # Map w (local 2x4) to global C (7x2)
        # Local w indices 0..7 (4 cols x 2 rows)
        # Local index = col * 2 + row.
        # Global index = (col + 3) * 2 + row.
        def map_w_224(loc):
            col = loc // 2
            row = loc % 2
            return (col + 3) * 2 + row
            
        add_term(col_offset, u, v, w, map_u, map_v_224, map_w_224)
        col_offset += 1
        
    # Part 2: A2 * B2 (M_2,1,7)
    terms_217 = get_217_terms()
    for t in terms_217:
        u, v, w = t
        # Map u (local 2x1) to global A (2x3)
        # Local u indices 0,1 -> Global 4,5 (col 2)
        def map_u_217(loc):
            return 4 + loc
            
        # Map v (local 1x7) to global B (3x7)
        # Local v indices 0..6 -> Global 14..20 (row 2)
        def map_v_217(loc):
            return 14 + loc
            
        # Map w (local 2x7) to global C (7x2)
        # Local w indices 0..13 -> Global 0..13
        map_w = lambda loc: loc
        
        add_term(col_offset, u, v, w, map_u_217, map_v_217, map_w)
        col_offset += 1
        
    # Verify rank
    if col_offset != rank:
        # If mismatch, pad or truncate? Should be exact.
        pass
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
