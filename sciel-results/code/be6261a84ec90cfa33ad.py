"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition by splitting:
    # n=3 into 2+1, m=4 into 2+2, p=8 into 4+4
    # Sub-problems:
    #   4 of size (2,2,4) -> each split into 2x(2,2,2) with Strassen rank 7, total 14 each
    #   4 of size (1,2,4) -> schoolbook rank 8 each
    # Total rank: 4*14 + 4*8 = 56 + 32 = 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]  # 3*4 = 12 rows
    V = [[0] * rank for _ in range(m * p)]  # 4*8 = 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 8*3 = 24 rows
    
    col = 0
    
    # Strassen 2x2x2: returns list of (U_entries, V_entries, W_entries) for 7 terms
    # U is indexed by (i, j) for i in 0..1, j in 0..1 -> 4 entries
    # V is indexed by (j, k) for j in 0..1, k in 0..1 -> 4 entries  
    # W is indexed by (k, i) for k in 0..1, i in 0..1 -> 4 entries
    # 
    # The 7 Strassen terms with coefficients:
    def strassen_2x2x2():
        """Return 7 tuples of (u_coeffs, v_coeffs, w_coeffs) for 2x2x2 Strassen.
        u_coeffs: list of (idx, coeff) for U (4 entries: (0,0),(0,1),(1,0),(1,1))
        v_coeffs: list of (idx, coeff) for V (4 entries: (0,0),(0,1),(1,0),(1,1))
        w_coeffs: list of (idx, coeff) for W (4 entries: (0,0),(0,1),(1,0),(1,1))
        """
        # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
        # Strassen formulas:
        # m1 = (a+d)(e+h): U: a=U[0], d=U[3]; V: e=V[0], h=V[3]; W: C[0,0] and C[1,1]
        # C[0,0] gets +m1, C[1,1] gets +m1
        terms = []
        # m1 = (A00+A11)(B00+B11)
        terms.append(([ (0,1),(3,1) ], [ (0,1),(3,1) ], [ (0,1),(3,1) ]))  # W[0]=C[0,0], W[3]=C[1,1]
        # m2 = (A10+A11)B00
        terms.append(([ (2,1),(3,1) ], [ (0,1) ], [ (2,1) ]))  # W[2]=C[1,0]
        # m3 = A00(B11-B10)
        terms.append(([ (0,1) ], [ (3,1),(2,-1) ], [ (1,1) ]))  # W[1]=C[0,1]
        # m4 = A11(B10-A00)
        terms.append(([ (3,1) ], [ (2,1),(0,-1) ], [ (2,1) ]))  # W[2]=C[1,0]
        # m5 = (A00+A01)B11
        terms.append(([ (0,1),(1,1) ], [ (3,1) ], [ (3,-1) ]))  # W[3]=C[1,1] with -1
        # m6 = (A10-A00)(B00+B01)
        terms.append(([ (2,1),(0,-1) ], [ (0,1),(1,1) ], [ (3,1) ]))  # W[3]=C[1,1]
        # m7 = (A01-A11)(B10+B11)
        terms.append(([ (1,1),(3,-1) ], [ (2,1),(3,1) ], [ (0,1) ]))  # W[0]=C[0,0]
        
        return terms
    
    # For a (2,2,4) sub-problem at offsets (i0, j0, k0):
    # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1, k0+2, k0+3}
    # Split k into two halves of 2: {k0, k0+1} and {k0+2, k0+3}
    # Each half is a 2x2x2 Strassen
    def add_2x2x4(i0, j0, k0):
        nonlocal col
        for khalf in range(2):
            k_start = k0 + 2 * khalf
            for (u_list, v_list, w_list) in strassen_2x2x2():
                for (ui, uc) in u_list:
                    U[i0 * 4 + j0 + ui][col] += uc
                for (vi, vc) in v_list:
                    V[(j0 + vi) * 8 + k_start + vi % 2][col] += vc  # Wait, need to be careful
                for (wi, wc) in w_list:
                    W[(k_start + wi % 2) * 3 + i0 + wi // 2][col] += wc  # Need to fix indexing
                col += 1
    
    # Hmm, I need to be more careful with the index mappings.
    # Let me restart with clearer indexing.
    
    # Reset
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U row index: i*m + j, where i in [0,n), j in [0,m)
    # V row index: j*p + k, where j in [0,m), k in [0,p)
    # W row index: k*n + i, where k in [0,p), i in [0,n)
    
    # For 2x2x2 Strassen with local indices (li, lj, lk) in {0,1}:
    # Global: i = i0 + li, j = j0 + lj, k = k0 + lk
    # U local index: li*2 + lj -> global: (i0+li)*m + (j0+lj)
    # V local index: lj*2 + lk -> global: (j0+lj)*p + (k0+lk)
    # W local index: lk*2 + li -> global: (k0+lk)*n + (i0+li)
    
    def strassen_2x2x2_terms():
        """7 terms, each is (u_coeffs, v_coeffs, w_coeffs)
        Each coeff list is [(local_idx, coefficient), ...]
        U local: li*2+lj
        V local: lj*2+lk
        W local: lk*2+li
        """
        terms = []
        # m1 = (A00+A11)(B00+B11)
        # U: A00=0, A11=3; V: B00=0, B11=3; W: C00=0, C11=3
        terms.append(([ (0,1),(3,1) ], [ (0,1),(3,1) ], [ (0,1),(3,1) ]))
        # m2 = (A10+A11)B00
        # U: A10=2, A11=3; V: B00=0; W: C10=2
        terms.append(([ (2,1),(3,1) ], [ (0,1) ], [ (2,1) ]))
        # m3 = A00(B11-B10)
        # U: A00=0; V: B11=3, B10=2; W: C01=1
        terms.append(([ (0,1) ], [ (3,1),(2,-1) ], [ (1,1) ]))
        # m4 = A11(B10-A00)
        # U: A11=3; V: B10=2, B00=0; W: C10=2
        terms.append(([ (3,1) ], [ (2,1),(0,-1) ], [ (2,1) ]))
        # m5 = (A00+A01)B11
        # U: A00=0, A01=1; V: B11=3; W: C11=3 with -1
        terms.append(([ (0,1),(1,1) ], [ (3,1) ], [ (3,-1) ]))
        # m6 = (A10-A00)(B00+B01)
        # U: A10=2, A00=0; V: B00=0, B01=1; W: C11=3
        terms.append(([ (2,1),(0,-1) ], [ (0,1),(1,1) ], [ (3,1) ]))
        # m7 = (A01-A11)(B10+B11)
        # U: A01=1, A11=3; V: B10=2, B11=3; W: C00=0
        terms.append(([ (1,1),(3,-1) ], [ (2,1),(3,1) ], [ (0,1) ]))
        return terms
    
    strassen_terms = strassen_2x2x2_terms()
    
    # Add a 2x2x2 sub-problem at global offsets (i0, j0, k0)
    def add_2x2x2(i0, j0, k0):
        nonlocal col
        for (u_list, v_list, w_list) in strassen_terms:
            for (li_lj, uc) in u_list:
                li = li_lj // 2
                lj = li_lj % 2
                gi = i0 + li
                gj = j0 + lj
                U[gi * m + gj][col] += uc
            for (lj_lk, vc) in v_list:
                lj = lj_lk // 2
                lk = lj_lk % 2
                gj = j0 + lj
                gk = k0 + lk
                V[gj * p + gk][col] += vc
            for (lk_li, wc) in w_list:
                lk = lk_li // 2
                li = lk_li % 2
                gk = k0 + lk
                gi = i0 + li
                W[gk * n + gi][col] += wc
            col += 1
    
    # Add a 1x2x4 sub-problem at global offsets (i0, j0, k0)
    # i is fixed (size 1), j in {j0, j0+1}, k in {k0..k0+3}
    # Schoolbook: 2*4 = 8 terms
    def add_1x2x4(i0, j0, k0):
        nonlocal col
        for lj in range(2):
            for lk in range(4):
                gi = i0
                gj = j0 + lj
                gk = k0 + lk
                U[gi * m + gj][col] = 1
                V[gj * p + gk][col] = 1
                W[gk * n + gi][col] = 1
                col += 1
    
    # Now compose:
    # n=3 = 2+1, m=4 = 2+2, p=8 = 4+4
    # i_blocks: [(0,2), (2,1)]
    # j_blocks: [(0,2), (2,2)]
    # k_blocks: [(0,4), (4,4)]
    
    for (i0, isz) in [(0,2), (2,1)]:
        for (j0, jsz) in [(0,2), (2,2)]:
            for (k0, ksz) in [(0,4), (4,4)]:
                if isz == 2 and jsz == 2:
                    # 2x2x4: split k into two 2x2x2
                    for khalf in range(2):
                        add_2x2x2(i0, j0, k0 + 2*khalf)
                elif isz == 1 and jsz == 2:
                    # 1x2x4: schoolbook
                    add_1x2x4(i0, j0, k0)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
