import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists, then convert to arrays if needed, 
    # but the interface expects lists of lists or similar. 
    # The prompt example uses lists of lists.
    
    U = []
    V = []
    W = []
    
    # Helper to add a term
    # u_indices: list of (global_idx, coeff)
    # v_indices: list of (global_idx, coeff)
    # w_indices: list of (global_idx, coeff)
    def add_term(u_idx, v_idx, w_idx):
        col_u = [0] * (n * m)
        col_v = [0] * (m * p)
        col_w = [0] * (p * n)
        
        for idx, coeff in u_idx:
            col_u[idx] += coeff
        for idx, coeff in v_idx:
            col_v[idx] += coeff
        for idx, coeff in w_idx:
            col_w[idx] += coeff
            
        U.append(col_u)
        V.append(col_v)
        W.append(col_w)

    # Strassen 2x2x2
    # Local A: 0,1 (rows), 0,1 (cols) -> global row r*2+dr, col s*2+dc
    # Local B: 0,1 (rows), 0,1 (cols) -> global row s*2+dr, col c*2+dc
    # Local C: 0,1 (rows), 0,1 (cols) -> global row c*2+dr, col r*2+dc (W index k*n+i)
    
    def strassen_222(r_base, s_base, c_base):
        # r_base, s_base, c_base are 0 or 2 for the 2x2 blocks
        # Global indices:
        # A rows: r_base, r_base+1
        # A cols: s_base, s_base+1
        # B rows: s_base, s_base+1
        # B cols: c_base, c_base+1
        # C rows: c_base, c_base+1
        # C cols: r_base, r_base+1
        
        # A[i,j] global index: i*m + j
        # B[i,j] global index: i*p + j
        # W[k*n+i] global index: k*n + i
        
        A00 = (r_base) * m + s_base
        A01 = (r_base) * m + s_base + 1
        A10 = (r_base + 1) * m + s_base
        A11 = (r_base + 1) * m + s_base + 1
        
        B00 = (s_base) * p + c_base
        B01 = (s_base) * p + c_base + 1
        B10 = (s_base + 1) * p + c_base
        B11 = (s_base + 1) * p + c_base + 1
        
        W00 = (c_base) * n + r_base
        W01 = (c_base) * n + r_base + 1
        W10 = (c_base + 1) * n + r_base
        W11 = (c_base + 1) * n + r_base + 1
        
        # M1 = (A00+A11)(B00+B11)
        add_term([(A00, 1), (A11, 1)], [(B00, 1), (B11, 1)], [(W00, 1), (W11, 1)])
        # M2 = (A10+A11)B00
        add_term([(A10, 1), (A11, 1)], [(B00, 1)], [(W10, 1), (W11, -1)])
        # M3 = A00(B01-B11)
        add_term([(A00, 1)], [(B01, 1), (B11, -1)], [(W01, 1), (W00, -1)])
        # M4 = A11(B10-B00)
        add_term([(A11, 1)], [(B10, 1), (B00, -1)], [(W10, 1), (W00, -1)])
        # M5 = (A00+A01)B11
        add_term([(A00, 1), (A01, 1)], [(B11, 1)], [(W00, -1), (W01, 1)])
        # M6 = (A10-A00)(B00+B01)
        add_term([(A10, 1), (A00, -1)], [(B00, 1), (B01, 1)], [(W11, 1)])
        # M7 = (A01-A11)(B10+B11)
        add_term([(A01, 1), (A11, -1)], [(B10, 1), (B11, 1)], [(W00, 1), (W11, -1)])

    # Standard 2x2x1
    # A 2x2, B 1x2, C 2x2
    def std_221(r_base, s_idx, c_base):
        # A rows r_base, r_base+1; cols s_idx
        # B row s_idx; cols c_base, c_base+1
        # C rows c_base, c_base+1; cols r_base, r_base+1
        
        A0 = (r_base) * m + s_idx
        A1 = (r_base + 1) * m + s_idx
        
        B0 = (s_idx) * p + c_base
        B1 = (s_idx) * p + c_base + 1
        
        W00 = (c_base) * n + r_base
        W01 = (c_base) * n + r_base + 1
        W10 = (c_base + 1) * n + r_base
        W11 = (c_base + 1) * n + r_base + 1
        
        # A0*B0 -> C00
        add_term([(A0, 1)], [(B0, 1)], [(W00, 1)])
        # A0*B1 -> C01
        add_term([(A0, 1)], [(B1, 1)], [(W10, 1)])
        # A1*B0 -> C10
        add_term([(A1, 1)], [(B0, 1)], [(W01, 1)])
        # A1*B1 -> C11
        add_term([(A1, 1)], [(B1, 1)], [(W11, 1)])

    # Standard 2x1x2
    # A 2x1, B 2x1, C 2x1
    def std_212(r_base, s_base, c_idx):
        # A rows r_base, r_base+1; col s_base..s_base+1? No, A is 2x1?
        # Wait, block is r (2), s (1), c (2).
        # A is 2x1? No, A is n x m. Block A is 2 x 1.
        # B is 1 x 2? No, B is m x p. Block B is 1 x 2.
        # C is 2 x 2? No, C is p x n. Block C is 2 x 2.
        # Wait, my previous classification:
        # s=2 (size 1), c in {0,1} (size 2).
        # A (2x1), B (1x2), C (2x2).
        # This is 2x1x2? No, A is 2x1, B is 1x2, C is 2x2.
        # Inner dim is 1.
        # So it's 2 (row) x 1 (inner) x 2 (col).
        # Rank 4.
        
        A0 = (r_base) * m + s_base
        A1 = (r_base + 1) * m + s_base
        
        B0 = (s_base) * p + c_base
        B1 = (s_base) * p + c_base + 1
        
        W00 = (c_base) * n + r_base
        W01 = (c_base) * n + r_base + 1
        W10 = (c_base + 1) * n + r_base
        W11 = (c_base + 1) * n + r_base + 1
        
        add_term([(A0, 1)], [(B0, 1)], [(W00, 1)])
        add_term([(A0, 1)], [(B1, 1)], [(W10, 1)])
        add_term([(A1, 1)], [(B0, 1)], [(W01, 1)])
        add_term([(A1, 1)], [(B1, 1)], [(W11, 1)])

    # Standard 2x1x1
    def std_211(r_base, s_idx, c_idx):
        A0 = (r_base) * m + s_idx
        A1 = (r_base + 1) * m + s_idx
        
        B0 = (s_idx) * p + c_idx
        
        W0 = (c_idx) * n + r_base
        W1 = (c_idx) * n + r_base + 1
        
        add_term([(A0, 1)], [(B0, 1)], [(W0, 1)])
        add_term([(A1, 1)], [(B0, 1)], [(W1, 1)])

    # Loop over blocks
    # r_base in [0, 2]
    # s_base in [0, 2, 4] (sizes 2, 2, 1)
    # c_base in [0, 2, 4] (sizes 2, 2, 1)
    
    for r_base in [0, 2]:
        for s_base, s_size in [(0, 2), (2, 2), (4, 1)]:
            for c_base, c_size in [(0, 2), (2, 2), (4, 1)]:
                if time.time() > deadline:
                    break
                
                if s_size == 2 and c_size == 2:
                    strassen_222(r_base, s_base, c_base)
                elif s_size == 2 and c_size == 1:
                    std_221(r_base, s_base, c_base)
                elif s_size == 1 and c_size == 2:
                    std_212(r_base, s_base, c_base)
                else:
                    std_211(r_base, s_base, c_base)
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
