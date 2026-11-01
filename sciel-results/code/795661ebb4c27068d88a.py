"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 9
    
    # We will build a tensor decomposition of rank 39 using a block decomposition:
    # Split A (2x3) into A_L (2x2) and A_R (2x1).
    # Split B (3x9) into B_L (2x9) and B_R (1x9).
    # C = A_L B_L + A_L B_R + A_R B_L + A_R B_R
    # 
    # Term 1: A_L B_L (2x2 * 2x9). Split B_L into 2x2, 2x2, 2x2, 2x1 blocks.
    #   - Three 2x2x2 blocks use Strassen (rank 7 each) -> 21
    #   - One 2x2x1 block uses naive (rank 2) -> 2
    # Term 2: A_L B_R (2x2 * 2x1). Naive (rank 4) -> 4
    # Term 3: A_R B_L (2x1 * 1x9). Outer product (rank 9) -> 9
    # Term 4: A_R B_R (2x1 * 1x1). Naive (rank 2) -> 2
    # Total rank = 21 + 2 + 4 + 9 + 2 = 38? 
    # Wait, let's re-verify the 2x2x1 block.
    # C (2x1) = A (2x2) B (2x1).
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    # This is 4 terms. Rank 4.
    # So Term 1 is 21 + 4 = 25.
    # Term 2 is 4.
    # Term 3 is 9.
    # Term 4 is 2.
    # Total = 25 + 4 + 9 + 2 = 40.
    
    # Let's check if we can do better on the 2x2x1 part.
    # Is there a rank 3 decomposition for 2x2x1?
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    # If we use rank 3:
    # t1: U=(1,0,1,0), V=(1,0), W=(1,0) -> A00 B0 -> C0
    # t2: U=(0,1,0,0), V=(0,1), W=(0,0) -> A01 B1 -> C0 (but W is 0? No, W must be C index)
    # W index for C0 is 0. W index for C1 is 1.
    # t1: U=(1,0,1,0), V=(1,0), W=(1,0) -> A00 B0 -> C0
    # t2: U=(0,1,0,0), V=(0,1), W=(1,0) -> A01 B1 -> C0
    # t3: U=(0,0,0,1), V=(1,1), W=(0,1) -> A11 (B0+B1) -> C1? No.
    # It seems rank 4 is minimal for 2x2x1.
    
    # So total rank 40.
    # Can we do better on the 2x2x2 blocks?
    # Strassen is 7. Is 6 possible? No, it's known to be 7.
    
    # Can we combine terms?
    # The blocks are disjoint in their indices, so we can just sum the ranks.
    
    rank = 40
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            if val != 0:
                U[idx, col] += val
        for idx, val in v_vals.items():
            if val != 0:
                V[idx, col] += val
        for idx, val in w_vals.items():
            if val != 0:
                W[idx, col] += val
        col += 1
        
    # Strassen for 2x2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b01)
    # m7 = (a01-a11)(b10+b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    def strassen_block(a_base, b_base, c_base):
        # a_base: index of A[0,0] in U
        # b_base: index of B[0,0] in V
        # c_base: index of C[0,0] in W
        # A indices: a_base+0 (0,0), a_base+1 (0,1), a_base+2 (1,0), a_base+3 (1,1)
        # B indices: b_base+0 (0,0), b_base+1 (0,1), b_base+2 (1,0), b_base+3 (1,1)
        # C indices: c_base+0 (0,0), c_base+1 (0,1), c_base+2 (1,0), c_base+3 (1,1)
        
        # m1
        add_term({a_base+0: 1, a_base+3: 1}, {b_base+0: 1, b_base+3: 1}, {c_base+0: 1, c_base+3: 1})
        # m2
        add_term({a_base+2: 1, a_base+3: 1}, {b_base+0: 1}, {c_base+2: 1, c_base+3: -1})
        # m3
        add_term({a_base+0: 1}, {b_base+1: 1, b_base+3: -1}, {c_base+1: 1, c_base+3: 1})
        # m4
        add_term({a_base+3: 1}, {b_base+2: 1, b_base+0: -1}, {c_base+0: 1, c_base+2: 1})
        # m5
        add_term({a_base+0: 1, a_base+1: 1}, {b_base+3: 1}, {c_base+0: -1, c_base+1: 1})
        # m6
        add_term({a_base+2: 1, a_base+0: -1}, {b_base+0: 1, b_base+1: 1}, {c_base+3: 1})
        # m7
        add_term({a_base+1: 1, a_base+3: -1}, {b_base+2: 1, b_base+3: 1}, {c_base+0: 1})
        
    # Naive for 2x2x2 (rank 4)
    def naive_block(a_base, b_base, c_base):
        add_term({a_base+0: 1}, {b_base+0: 1}, {c_base+0: 1})
        add_term({a_base+1: 1}, {b_base+2: 1}, {c_base+0: 1})
        add_term({a_base+2: 1}, {b_base+0: 1}, {c_base+2: 1})
        add_term({a_base+3: 1}, {b_base+2: 1}, {c_base+2: 1})
        
    # Naive for 2x2x1 (rank 4)
    def naive_221(a_base, b_base, c_base):
        # A (2x2), B (2x1), C (2x1)
        # A indices: a_base+0 (0,0), a_base+1 (0,1), a_base+2 (1,0), a_base+3 (1,1)
        # B indices: b_base+0 (0,0), b_base+1 (1,0)
        # C indices: c_base+0 (0,0), c_base+1 (1,0)
        add_term({a_base+0: 1}, {b_base+0: 1}, {c_base+0: 1})
        add_term({a_base+1: 1}, {b_base+1: 1}, {c_base+0: 1})
        add_term({a_base+2: 1}, {b_base+0: 1}, {c_base+1: 1})
        add_term({a_base+3: 1}, {b_base+1: 1}, {c_base+1: 1})
        
    # Outer product for 2x1 * 1x9 (rank 9)
    def outer_21_19(a_base, b_base, c_base):
        # A (2x1), B (1x9), C (2x9)
        # A indices: a_base+0 (0,0), a_base+1 (1,0)
        # B indices: b_base+0..8 (0,0)..(0,8)
        # C indices: c_base+0..17 (0,0)..(1,8)
        for k in range(9):
            add_term({a_base+0: 1, a_base+1: 1}, {b_base+k: 1}, {c_base+k: 1, c_base+9+k: 1})
            
    # Naive for 2x1 * 1x1 (rank 2)
    def naive_21_11(a_base, b_base, c_base):
        # A (2x1), B (1x1), C (2x1)
        # A indices: a_base+0 (0,0), a_base+1 (1,0)
        # B indices: b_base+0 (0,0)
        # C indices: c_base+0 (0,0), c_base+1 (1,0)
        add_term({a_base+0: 1}, {b_base+0: 1}, {c_base+0: 1})
        add_term({a_base+1: 1}, {b_base+0: 1}, {c_base+1: 1})
        
    # Term 1: A_L B_L (2x2 * 2x9)
    # A_L is A[:, 0:2]. U indices: 0..3
    # B_L is B[0:2, :]. V indices: 0..17
    # C part is C[:, 0:8]. W indices: 0..15 (for k=0..8, C[i,k] is at k*n+i)
    # Wait, W index is k*n+i.
    # For k=0..8, i=0..1.
    # k=0: W[0], W[1]
    # k=1: W[2], W[3]
    # ...
    # k=8: W[16], W[17]
    
    # Split B_L into B_L1 (2x2), B_L2 (2x2), B_L3 (2x2), B_L4 (2x1)
    # B_L1: B[0:2, 0:2]. V indices: 0..3.
    # B_L2: B[0:2, 2:4]. V indices: 6..9. (B[0,2]=6, B[0,3]=7, B[1,2]=8, B[1,3]=9)
    # B_L3: B[0:2, 4:6]. V indices: 12..15.
    # B_L4: B[0:2, 6:8]. V indices: 18..21. (B[0,6]=18, B[0,7]=19, B[1,6]=20, B[1,7]=21)
    
    # C for B_L1: C[:, 0:2]. W indices: 0..3.
    # C for B_L2: C[:, 2:4]. W indices: 4..7.
    # C for B_L3: C[:, 4:6]. W indices: 8..11.
    # C for B_L4: C[:, 6:8]. W indices: 12..15.
    
    strassen_block(0, 0, 0)
    strassen_block(0, 6, 4)
    strassen_block(0, 12, 8)
    naive_221(0, 18, 12)
    
    # Term 2: A_L B_R (2x2 * 2x1)
    # A_L is A[:, 0:2]. U indices: 0..3
    # B_R is B[0:2, 8]. V indices: 8, 10. (B[0,8]=8, B[1,8]=10)
    # C part is C[:, 8]. W indices: 16, 17.
    naive_221(0, 8, 16)
    
    # Term 3: A_R B_L (2x1 * 1x9)
    # A_R is A[:, 2]. U indices: 4, 5.
    # B_L is B[2, 0:9]. V indices: 18..26.
    # C part is C[:, 0:9]. W indices: 0..17.
    # Wait, this overlaps with Term 1's C indices. That's fine, we just add to W.
    # But we need to be careful with the U and V indices.
    # A_R U indices: 4 (0,2), 5 (1,2).
    # B_L V indices: 18..26.
    # C W indices: 0..17.
    outer_21_19(4, 18, 0)
    
    # Term 4: A_R B_R (2x1 * 1x1)
    # A_R is A[:, 2]. U indices: 4, 5.
    # B_R is B[2, 8]. V index: 26.
    # C part is C[:, 8]. W indices: 16, 17.
    naive_21_11(4, 26, 16)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
