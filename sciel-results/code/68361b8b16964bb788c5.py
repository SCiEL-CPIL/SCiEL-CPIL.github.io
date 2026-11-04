import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 100
    # Structure: 4 blocks of 2x2x7
    # Each 2x2x7 is decomposed into 3x (2x2x2 Strassen) + 1x (2x2x1 Schoolbook)
    # Rank per 2x2x7: 3*7 + 4 = 25
    # Total rank: 4 * 25 = 100
    
    rank = 100
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term to the decomposition
    # u_idx: list of (index_in_U, value)
    # v_idx: list of (index_in_V, value)
    # w_idx: list of (index_in_W, value)
    def add_term(col, u_terms, v_terms, w_terms):
        for idx, val in u_terms:
            U[idx, col] += val
        for idx, val in v_terms:
            V[idx, col] += val
        for idx, val in w_terms:
            W[idx, col] += val
            
    # Define the 2x2x7 decomposition logic
    # Local indices for 2x2x7:
    # U (A): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V (B): 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(0,4), 5:(0,5), 6:(0,6),
    #        7:(1,0), 8:(1,1), 9:(1,2), 10:(1,3), 11:(1,4), 12:(1,5), 13:(1,6)
    # W (C): 0:(0,0), 1:(0,1), 2:(0,2), 3:(0,3), 4:(0,4), 5:(0,5), 6:(0,6),
    #        7:(1,0), 8:(1,1), 9:(1,2), 10:(1,3), 11:(1,4), 12:(1,5), 13:(1,6)
    # Note: W index is k*n + i? No, p*n.
    # C is 2x7. Flatten: C[0,0]...C[0,6], C[1,0]...C[1,6]
    # W index (k, i) -> k*2 + i.
    # So W[0] is (0,0), W[1] is (0,1), ..., W[6] is (0,6), W[7] is (1,0), etc.
    
    def build_2x2x7(col_offset, u_map, v_map, w_map):
        # u_map: local u index -> global U index
        # v_map: local v index -> global V index
        # w_map: local w index -> global W index
        
        # Strassen for cols 0-1
        # M1 = (A00+A11)(B00+B11) -> W00 += M1, W01 += M1
        # Note: W is C. C00 gets M1. C01 gets M1?
        # Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = A11(B01-B11)
        # M3 = (A00-A11)B11
        # M4 = A00(B10-B00)
        # M5 = (A00+A10)B00
        # M6 = (A11+A10)(B10-B00)
        # M7 = (A00+A11)(B01-B11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M2 + M7
        # C10 = M3 + M6
        # C11 = M1 - M2 + M3 + M5
        
        # Term 1: M1
        u_terms = [(u_map[0], 1), (u_map[3], 1)]
        v_terms = [(v_map[0], 1), (v_map[8], 1)] # B00, B11 (local 0, 7? No, B11 is local 8? 
        # Local V indices:
        # 0: (0,0), 1: (0,1), 2: (0,2)...
        # 7: (1,0), 8: (1,1)...
        # So B00=0, B11=8.
        v_terms = [(v_map[0], 1), (v_map[8], 1)]
        w_terms = [(w_map[0], 1), (w_map[7], 1)] # C00, C11? 
        # Wait, C00 is local W 0. C11 is local W 7? 
        # W index (k, i). C11 is i=1, k=1. Index 1*2+1 = 3.
        # Let's re-verify W mapping.
        # C is 2x7.
        # C[i, k].
        # W index is k*n + i = k*2 + i.
        # C00: i=0, k=0 -> 0.
        # C01: i=0, k=1 -> 2.
        # C10: i=1, k=0 -> 1.
        # C11: i=1, k=1 -> 3.
        # So:
        # C00 -> 0
        # C01 -> 2
        # C10 -> 1
        # C11 -> 3
        # C02 -> 4
        # C03 -> 6
        # C12 -> 5
        # C13 -> 7
        # C04 -> 8
        # C05 -> 10
        # C14 -> 9
        # C15 -> 11
        # C06 -> 12
        # C16 -> 13
        
        # Correct W mapping for Strassen 2x2x2 (cols 0,1):
        # C00 -> 0
        # C01 -> 2
        # C10 -> 1
        # C11 -> 3
        
        # M1 contributes to C00 and C11.
        w_terms = [(w_map[0], 1), (w_map[3], 1)]
        add_term(col_offset, u_terms, v_terms, w_terms)
        
        # M2 = A11(B01-B11) -> C01 += M2
        u_terms = [(u_map[3], 1)]
        v_terms = [(v_map[1], 1), (v_map[9], -1)] # B01=1, B11=9? 
        # B11 is local V index 8? 
        # V indices:
        # 0: (0,0), 1: (0,1)
        # 7: (1,0), 8: (1,1)
        # So B11 is 8.
        v_terms = [(v_map[1], 1), (v_map[8], -1)]
        w_terms = [(w_map[2], 1)] # C01
        add_term(col_offset + 1, u_terms, v_terms, w_terms)
        
        # M3 = (A00-A11)B11 -> C11 += M3
        u_terms = [(u_map[0], 1), (u_map[3], -1)]
        v_terms = [(v_map[8], 1)]
        w_terms = [(w_map[3], 1)]
        add_term(col_offset + 2, u_terms, v_terms, w_terms)
        
        # M4 = A00(B10-B00) -> C10 += M4
        u_terms = [(u_map[0], 1)]
        v_terms = [(v_map[7], 1), (v_map[0], -1)] # B10=7, B00=0
        w_terms = [(w_map[1], 1)] # C10
        add_term(col_offset + 3, u_terms, v_terms, w_terms)
        
        # M5 = (A00+A10)B00 -> C00 += M5, C11 += M5
        u_terms = [(u_map[0], 1), (u_map[2], 1)]
        v_terms = [(v_map[0], 1)]
        w_terms = [(w_map[0], 1), (w_map[3], 1)]
        add_term(col_offset + 4, u_terms, v_terms, w_terms)
        
        # M6 = (A11+A10)(B10-B00) -> C10 += M6
        u_terms = [(u_map[3], 1), (u_map[2], 1)]
        v_terms = [(v_map[7], 1), (v_map[0], -1)]
        w_terms = [(w_map[1], 1)]
        add_term(col_offset + 5, u_terms, v_terms, w_terms)
        
        # M7 = (A00+A11)(B01-B11) -> C01 += M7, C00 -= M7
        u_terms = [(u_map[0], 1), (u_map[3], 1)]
        v_terms = [(v_map[1], 1), (v_map[8], -1)]
        w_terms = [(w_map[2], 1), (w_map[0], -1)]
        add_term(col_offset + 6, u_terms, v_terms, w_terms)
        
        # Strassen for cols 2-3
        # V indices: B02=2, B03=3, B12=9, B13=10
        # W indices: C02=4, C03=6, C12=5, C13=7
        
        # M1'
        u_terms = [(u_map[0], 1), (u_map[3], 1)]
        v_terms = [(v_map[2], 1), (v_map[9], 1)]
        w_terms = [(w_map[4], 1), (w_map[7], 1)]
        add_term(col_offset + 7, u_terms, v_terms, w_terms)
        
        # M2'
        u_terms = [(u_map[3], 1)]
        v_terms = [(v_map[3], 1), (v_map[9], -1)]
        w_terms = [(w_map[6], 1)]
        add_term(col_offset + 8, u_terms, v_terms, w_terms)
        
        # M3'
        u_terms = [(u_map[0], 1), (u_map[3], -1)]
        v_terms = [(v_map[9], 1)]
        w_terms = [(w_map[7], 1)]
        add_term(col_offset + 9, u_terms, v_terms, w_terms)
        
        # M4'
        u_terms = [(u_map[0], 1)]
        v_terms = [(v_map[8], 1), (v_map[2], -1)] # B12=9? No, B12 is (1,2). Local V index 9.
        # B12 is row 1, col 2. Index 7+2 = 9.
        # B02 is row 0, col 2. Index 2.
        v_terms = [(v_map[9], 1), (v_map[2], -1)]
        w_terms = [(w_map[5], 1)] # C12
        add_term(col_offset + 10, u_terms, v_terms, w_terms)
        
        # M5'
        u_terms = [(u_map[0], 1), (u_map[2], 1)]
        v_terms = [(v_map[2], 1)]
        w_terms = [(w_map[4], 1), (w_map[7], 1)]
        add_term(col_offset + 11, u_terms, v_terms, w_terms)
        
        # M6'
        u_terms = [(u_map[3], 1), (u_map[2], 1)]
        v_terms = [(v_map[9], 1), (v_map[2], -1)]
        w_terms = [(w_map[5], 1)]
        add_term(col_offset + 12, u_terms, v_terms, w_terms)
        
        # M7'
        u_terms = [(u_map[0], 1), (u_map[3], 1)]
        v_terms = [(v_map[3], 1), (v_map[9], -1)]
        w_terms = [(w_map[6], 1), (w_map[4], -1)]
        add_term(col_offset + 13, u_terms, v_terms, w_terms)
        
        # Strassen for cols 4-5
        # V indices: B04=4, B05=5, B14=11, B15=12
        # W indices: C04=8, C05=10, C14=9, C15=11
        
        # M1''
        u_terms = [(u_map[0], 1), (u_map[3], 1)]
        v_terms = [(v_map[4], 1), (v_map[11], 1)]
        w_terms = [(w_map[8], 1), (w_map[11], 1)]
        add_term(col_offset + 14, u_terms, v_terms, w_terms)
        
        # M2''
        u_terms = [(u_map[3], 1)]
        v_terms = [(v_map[5], 1), (v_map[11], -1)]
        w_terms = [(w_map[10], 1)]
        add_term(col_offset + 15, u_terms, v_terms, w_terms)
        
        # M3''
        u_terms = [(u_map[0], 1), (u_map[3], -1)]
        v_terms = [(v_map[11], 1)]
        w_terms = [(w_map[11], 1)]
        add_term(col_offset + 16, u_terms, v_terms, w_terms)
        
        # M4''
        u_terms = [(u_map[0], 1)]
        v_terms = [(v_map[10], 1), (v_map[4], -1)] # B14=11? No, B14 is (1,4). Index 7+4=11.
        # B04 is (0,4). Index 4.
        v_terms = [(v_map[11], 1), (v_map[4], -1)]
        w_terms = [(w_map[9], 1)] # C14
        add_term(col_offset + 17, u_terms, v_terms, w_terms)
        
        # M5''
        u_terms = [(u_map[0], 1), (u_map[2], 1)]
        v_terms = [(v_map[4], 1)]
        w_terms = [(w_map[8], 1), (w_map[11], 1)]
        add_term(col_offset + 18, u_terms, v_terms, w_terms)
        
        # M6''
        u_terms = [(u_map[3], 1), (u_map[2], 1)]
        v_terms = [(v_map[11], 1), (v_map[4], -1)]
        w_terms = [(w_map[9], 1)]
        add_term(col_offset + 19, u_terms, v_terms, w_terms)
        
        # M7''
        u_terms = [(u_map[0], 1), (u_map[3], 1)]
        v_terms = [(v_map[5], 1), (v_map[11], -1)]
        w_terms = [(w_map[10], 1), (w_map[8], -1)]
        add_term(col_offset + 20, u_terms, v_terms, w_terms)
        
        # Schoolbook for col 6
        # V indices: B06=6, B16=13
        # W indices: C06=12, C16=13
        
        # A00 B06 -> C06
        u_terms = [(u_map[0], 1)]
        v_terms = [(v_map[6], 1)]
        w_terms = [(w_map[12], 1)]
        add_term(col_offset + 21, u_terms, v_terms, w_terms)
        
        # A01 B16 -> C06? No, A01 B16 contributes to C06?
        # C[i,k] = sum_j A[i,j] B[j,k].
        # C06 = A00 B06 + A01 B16.
        # So A01 B16 also goes to C06.
        u_terms = [(u_map[1], 1)]
        v_terms = [(v_map[13], 1)]
        w_terms = [(w_map[12], 1)]
        add_term(col_offset + 22, u_terms, v_terms, w_terms)
        
        # A10 B06 -> C16
        u_terms = [(u_map[2], 1)]
        v_terms = [(v_map[6], 1)]
        w_terms = [(w_map[13], 1)]
        add_term(col_offset + 23, u_terms, v_terms, w_terms)
        
        # A11 B16 -> C16
        u_terms = [(u_map[3], 1)]
        v_terms = [(v_map[13], 1)]
        w_terms = [(w_map[13], 1)]
        add_term(col_offset + 24, u_terms, v_terms, w_terms)
        
    # Now map the 4 blocks
    # Block 1: A11 (rows 0-1, cols 0-1) * B_top (rows 0-1, cols 0-6) -> C_top_left (rows 0-1, cols 0-6)
    # U indices: 0,1,4,5
    # V indices: 0-13
    # W indices: 0-13 (C rows 0-1, cols 0-6)
    
    u_map_1 = {0: 0, 1: 1, 2: 4, 3: 5}
    v_map_1 = {i: i for i in range(14)}
    w_map_1 = {i: i for i in range(14)}
    build_2x2x7(0, u_map_1, v_map_1, w_map_1)
    
    # Block 2: A12 (rows 0-1, cols 2-3) * B_bot (rows 2-3, cols 0-6) -> C_top_right (rows 0-1, cols 0-6)
    # U indices: 2,3,6,7
    # V indices: 14-27
    # W indices: 0-13
    
    u_map_2 = {0: 2, 1: 3, 2: 6, 3: 7}
    v_map_2 = {i: 14 + i for i in range(14)}
    w_map_2 = {i: i for i in range(14)}
    build_2x2x7(25, u_map_2, v_map_2, w_map_2)
    
    # Block 3: A21 (rows 2-3, cols 0-1) * B_top (rows 0-1, cols 0-6) -> C_bot_left (rows 2-3, cols 0-6)
    # U indices: 8,9,12,13
    # V indices: 0-13
    # W indices: 14-27 (C rows 2-3, cols 0-6)
    # W index for C[i,k] where i is 2 or 3.
    # W index = k*4 + i.
    # For i=2, k=0..6: 2, 6, 10, 14, 18, 22, 26.
    # For i=3, k=0..6: 3, 7, 11, 15, 19, 23, 27.
    # So w_map_3 should map local 0..13 to these.
    # Local 0 (C00) -> Global 2
    # Local 1 (C10) -> Global 3
    # Local 2 (C01) -> Global 6
    # Local 3 (C11) -> Global 7
    # Local 4 (C02) -> Global 10
    # Local 5 (C12) -> Global 11
    # Local 6 (C03) -> Global 14
    # Local 7 (C13) -> Global 15
    # Local 8 (C04) -> Global 18
    # Local 9 (C14) -> Global 19
    # Local 10 (C05) -> Global 22
    # Local 11 (C15) -> Global 23
    # Local 12 (C06) -> Global 26
    # Local 13 (C16) -> Global 27
    
    u_map_3 = {0: 8, 1: 9, 2: 12, 3: 13}
    v_map_3 = {i: i for i in range(14)}
    w_map_3 = {
        0: 2, 1: 3, 2: 6, 3: 7, 4: 10, 5: 11, 6: 14, 7: 15,
        8: 18, 9: 19, 10: 22, 11: 23, 12: 26, 13: 27
    }
    build_2x2x7(50, u_map_3, v_map_3, w_map_3)
    
    # Block 4: A22 (rows 2-3, cols 2-3) * B_bot (rows 2-3, cols 0-6) -> C_bot_right (rows 2-3, cols 0-6)
    # U indices: 10,11,14,15
    # V indices: 14-27
    # W indices: 14-27
    
    u_map_4 = {0: 10, 1: 11, 2: 14, 3: 15}
    v_map_4 = {i: 14 + i for i in range(14)}
    w_map_4 = w_map_3
    build_2x2x7(75, u_map_4, v_map_4, w_map_4)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
