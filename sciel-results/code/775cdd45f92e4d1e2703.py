"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 39.
    # Strategy:
    # Split m=3 into m1=2 and m2=1.
    # Split p=7 into p1=2, p2=2, p3=2, p4=1 for the m1 part.
    # 
    # A (2x3) = [A1 (2x2), A2 (2x1)]
    # B (3x7) = [B1 (2x7); B2 (1x7)]
    # C (2x7) = A1*B1 + A2*B2
    #
    # A1*B1 (2x2x7):
    # Split B1 (2x7) into B11 (2x2), B12 (2x2), B13 (2x2), B14 (2x1)
    # C1 = A1*B11 + A1*B12 + A1*B13 + A1*B14
    #
    # Ranks:
    # M(2,2,2) = 7 (Strassen)
    # M(2,2,1) = 4
    # M(2,1,7) = 14
    # Total = 7 + 7 + 7 + 4 + 14 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)] # 6 x 39
    V = [[0] * rank for _ in range(m * p)] # 21 x 39
    W = [[0] * rank for _ in range(p * n)] # 14 x 39
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val_u=1, val_v=1, val_w=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w
        col += 1

    # Strassen decomposition for M(2,2,2)
    # Indices for 2x2x2:
    # A: 0,1 (row 0), 2,3 (row 1) -> i*2+j
    # B: 0,1 (col 0), 2,3 (col 1) -> j*2+k
    # C: 0,1 (col 0), 2,3 (col 1) -> k*2+i
    
    def add_strassen(base_u, base_v, base_w):
        # base_u: index of A[0,0] in U
        # base_v: index of B[0,0] in V
        # base_w: index of C[0,0] in W
        
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        # M2 = (A10+A11)B00
        # C10 += M2
        # M3 = A00(B01-B11)
        # C01 += M3
        # M4 = A11(B10-B00)
        # C10 += M4, C11 -= M4
        # M5 = (A00+A01)B11
        # C01 += M5
        # M6 = (A10-A00)(B00+B01)
        # C00 -= M6, C10 += M6
        # M7 = (A01-A11)(B10+B11)
        # C01 -= M7, C11 += M7
        
        # Term 1: M1
        # A00, A11, B00, B11, C00, C11
        add_term(base_u + 0, base_v + 0, base_w + 0, 1, 1, 1) # A00 B00 -> C00
        add_term(base_u + 1, base_v + 2, base_w + 0, 1, 1, 1) # A01 B11 -> C00 (Wait, Strassen M1 is (A00+A11)(B00+B11))
        # My add_term adds single products. Strassen combines them.
        # I need to implement Strassen as 7 terms where each term is a product of sums.
        # The interface requires U, V, W to be matrices such that C = U * V^T * W^T?
        # No, the standard tensor decomposition is:
        # C_{i,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # This corresponds to sum_r (U_r^T * A * V_r) * W_r?
        # Actually, U is (n*m, r), V is (m*p, r), W is (p*n, r).
        # The product is sum_r U[:,r] * V[:,r] * W[:,r]?
        # The tensor is T = sum_r u_r \otimes v_r \otimes w_r.
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        
        # So for Strassen M1 = (A00+A11)(B00+B11):
        # This is not a single rank-1 term. It is a sum of 4 rank-1 terms if expanded.
        # But Strassen's algorithm uses 7 multiplications of *sums*.
        # The tensor decomposition requires rank-1 terms (products of scalars).
        # Strassen's 7 "multiplications" are not rank-1 tensor terms in the sense of single indices.
        # They are linear combinations.
        # To convert Strassen to tensor decomposition, we must expand the sums.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This would increase the rank.
        # The rank of M(2,2,2) is 7. This means there exist 7 vectors u, v, w such that T = sum u_r v_r w_r.
        # These vectors contain the coefficients of the linear combinations.
        
        # So, for Strassen, the vectors are:
        # r=0: u=(1,0,1,0), v=(1,0,0,1), w=(1,0,0,-1) ? No.
        
        # Let's use the explicit rank-7 decomposition vectors for M(2,2,2).
        # u_r, v_r, w_r are vectors of length 4 (indices 0..3).
        
        # Standard Rank-7 Decomposition:
        # 1. u=(1,0,1,0), v=(1,0,0,1), w=(1,0,0,-1) -> (A00+A11)(B00+B11) -> C00-C11?
        
        # I will use a known list of 7 triples for 2x2x2.
        # Indices: A: 00,01,10,11 -> 0,1,2,3
        # B: 00,10,01,11 -> 0,2,1,3 (j*2+k)
        # C: 00,10,01,11 -> 0,2,1,3 (k*2+i)
        
        # Triple 1:
        # u: 1, 0, 1, 0  (A00 + A11)
        # v: 1, 0, 0, 1  (B00 + B11)
        # w: 1, 0, 0, -1 (C00 - C11)
        
        # Triple 2:
        # u: 0, 0, 1, 1  (A10 + A11)
        # v: 1, 0, 0, 0  (B00)
        # w: 0, 0, 1, 0  (C10)
        
        # Triple 3:
        # u: 1, 0, 0, 0  (A00)
        # v: 0, 0, 1, -1 (B01 - B11)
        # w: 0, 0, 1, 0  (C01)
        
        # Triple 4:
        # u: 0, 0, 0, 1  (A11)
        # v: 0, 1, -1, 0 (B10 - B00)
        # w: 0, 1, 0, -1 (C10 - C11)
        
        # Triple 5:
        # u: 1, 1, 0, 0  (A00 + A01)
        # v: 0, 0, 0, 1  (B11)
        # w: 0, 0, 1, 0  (C01)
        
        # Triple 6:
        # u: -1, 0, 1, 0 (A10 - A00)
        # v: 1, 0, 1, 0  (B00 + B01)
        # w: 1, 0, 0, 0  (C00)
        
        # Triple 7:
        # u: 0, 1, 0, -1 (A01 - A11)
        # v: 0, 1, 0, 1  (B10 + B11)
        # w: 0, 1, 0, 0  (C11)
        
        # I will hardcode these 7 triples.
        
        # Mapping indices for 2x2x2 block:
        # u_idx_offset, v_idx_offset, w_idx_offset
        
        triples = [
            # u, v, w
            ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, -1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 0]),
            ([0, 0, 0, 1], [0, 1, -1, 0], [0, 1, 0, -1]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0]),
            ([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, 0]),
        ]
        
        for u_vec, v_vec, w_vec in triples:
            for ju, uu in enumerate(u_vec):
                if uu != 0:
                    U[base_u + ju][col] += uu
            for jv, vv in enumerate(v_vec):
                if vv != 0:
                    V[base_v + jv][col] += vv
            for jw, ww in enumerate(w_vec):
                if ww != 0:
                    W[base_w + jw][col] += ww
            col += 1

    # --- Part 1: A1 * B11 (2x2x2) ---
    # A1 is A[:, 0:2]. Indices in U: 0,1,2,3.
    # B11 is B[0:2, 0:2]. Indices in V: 0,2,1,3 (j*2+k).
    #   j=0, k=0 -> 0
    #   j=0, k=1 -> 1
    #   j=1, k=0 -> 2
    #   j=1, k=1 -> 3
    # Wait, V index is j*p + k. Here p=2 for the sub-problem.
    # But in the global V, the indices are different.
    # Global V indices: j*7 + k.
    # B11 corresponds to j in {0,1}, k in {0,1}.
    # V indices: 0*7+0=0, 0*7+1=1, 1*7+0=7, 1*7+1=8.
    
    # Global W indices: k*n + i.
    # C11 corresponds to k in {0,1}, i in {0,1}.
    # W indices: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3.
    
    # Strassen indices for u (A): 0,1,2,3 map to A[0,0], A[0,1], A[1,0], A[1,1].
    # Global U indices: i*3 + j.
    # A[0,0]->0, A[0,1]->1, A[1,0]->3, A[1,1]->4.
    # So base_u is not contiguous.
    
    # I will create a mapping function.
    
    def get_u_idx(i, j):
        return i * m + j
        
    def get_v_idx(j, k):
        return j * p + k
        
    def get_w_idx(k, i):
        return k * n + i

    # --- Add Strassen for A1*B11 ---
    # u_map: [A00, A01, A10, A11] -> [0, 1, 3, 4]
    # v_map: [B00, B10, B01, B11] -> [0, 7, 1, 8] (j*7+k)
    #   Strassen v indices: 0->B00, 1->B10, 2->B01, 3->B11
    # w_map: [C00, C10, C01, C11] -> [0, 2, 1, 3] (k*2+i)
    #   Strassen w indices: 0->C00, 1->C10, 2->C01, 3->C11
    
    u_map_1 = [get_u_idx(0,0), get_u_idx(0,1), get_u_idx(1,0), get_u_idx(1,1)]
    v_map_1 = [get_v_idx(0,0), get_v_idx(1,0), get_v_idx(0,1), get_v_idx(1,1)]
    w_map_1 = [get_w_idx(0,0), get_w_idx(1,0), get_w_idx(0,1), get_w_idx(1,1)]
    
    # Strassen Triples (u, v, w)
    strassen_triples = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, -1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 0]),
        ([0, 0, 0, 1], [0, 1, -1, 0], [0, 1, 0, -1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [0, 1, 0, 0]),
    ]
    
    for u_vec, v_vec, w_vec in strassen_triples:
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[u_map_1[idx]][col] += val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[v_map_1[idx]][col] += val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[w_map_1[idx]][col] += val
        col += 1

    # --- Add Strassen for A1*B12 ---
    # B12 is B[0:2, 2:4]. k in {2,3}.
    # v_map_2: [B02, B12, B03, B13] -> [get_v_idx(0,2), get_v_idx(1,2), get_v_idx(0,3), get_v_idx(1,3)]
    # w_map_2: [C02, C12, C03, C13] -> [get_w_idx(2,0), get_w_idx(3,0), get_w_idx(2,1), get_w_idx(3,1)]
    
    u_map_2 = u_map_1
    v_map_2 = [get_v_idx(0,2), get_v_idx(1,2), get_v_idx(0,3), get_v_idx(1,3)]
    w_map_2 = [get_w_idx(2,0), get_w_idx(3,0), get_w_idx(2,1), get_w_idx(3,1)]
    
    for u_vec, v_vec, w_vec in strassen_triples:
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[u_map_2[idx]][col] += val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[v_map_2[idx]][col] += val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[w_map_2[idx]][col] += val
        col += 1

    # --- Add Strassen for A1*B13 ---
    # B13 is B[0:2, 4:6]. k in {4,5}.
    
    u_map_3 = u_map_1
    v_map_3 = [get_v_idx(0,4), get_v_idx(1,4), get_v_idx(0,5), get_v_idx(1,5)]
    w_map_3 = [get_w_idx(4,0), get_w_idx(5,0), get_w_idx(4,1), get_w_idx(5,1)]
    
    for u_vec, v_vec, w_vec in strassen_triples:
        for idx, val in enumerate(u_vec):
            if val != 0:
                U[u_map_3[idx]][col] += val
        for idx, val in enumerate(v_vec):
            if val != 0:
                V[v_map_3[idx]][col] += val
        for idx, val in enumerate(w_vec):
            if val != 0:
                W[w_map_3[idx]][col] += val
        col += 1

    # --- Add Schoolbook for A1*B14 ---
    # B14 is B[0:2, 6:7]. k in {6}.
    # M(2,2,1) Rank 4.
    # Terms: A00B06->C06, A01B16->C06, A10B06->C16, A11B16->C16
    
    for i in range(2):
        for j in range(2):
            k = 6
            u_idx = get_u_idx(i, j)
            v_idx = get_v_idx(j, k)
            w_idx = get_w_idx(k, i)
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    # --- Add Schoolbook for A2*B2 ---
    # A2 is A[:, 2]. j=2.
    # B2 is B[2, :]. j=2, k in 0..6.
    # M(2,1,7) Rank 14.
    # Terms: A_i2 * B_2k -> C_ik
    
    for i in range(2):
        for k in range(7):
            j = 2
            u_idx = get_u_idx(i, j)
            v_idx = get_v_idx(j, k)
            w_idx = get_w_idx(k, i)
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
