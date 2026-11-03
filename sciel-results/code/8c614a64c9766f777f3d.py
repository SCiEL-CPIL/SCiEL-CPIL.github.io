import time
import numpy as np

DIMENSIONS = [4, 5, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # U shape: (n*m, rank) = (20, rank)
    # V shape: (m*p, rank) = (25, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u, v, w):
        U_terms.append(u)
        V_terms.append(v)
        W_terms.append(w)
        
    # Helper to create sparse vectors as lists of (index, value)
    # We will convert to dense at the end or keep sparse if possible.
    # The interface expects matrices.
    
    # 1. Strassen 4x4x4 part
    # A is 4x4 (indices 0..3, 0..3 in the 4x5 A, so col 0..3)
    # B is 4x4 (indices 0..3, 0..3 in the 5x5 B, so row 0..3)
    # C is 4x4 (indices 0..3, 0..3 in the 4x5 C, so row 0..3)
    
    # We need to map local 2x2 Strassen terms to global indices.
    # Local A index (li, lj) -> Global A index (li, lj) if in top-left?
    # The 4x4 A is the top-left 4x4 of the 4x5 A.
    # Global A index: i * 5 + j.
    # Local A index in 4x4: i * 4 + j.
    # But Strassen works on 2x2 blocks.
    
    # Let's define the 2x2 Strassen basis vectors.
    # For a 2x2 matrix M with indices 0,1,2,3 corresponding to (0,0),(0,1),(1,0),(1,1)
    # Strassen terms (u, v, w) where u, v, w are length 4 vectors.
    
    strassen_2x2_terms = []
    
    # P1: (A00+A11)(B00+B11) -> C00+C11? No, contributes to C00 and C11?
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Term 1: P1
    # u: A00=1, A11=1 -> [1,0,0,1]
    # v: B00=1, B11=1 -> [1,0,0,1]
    # w: C00=1, C11=1? No, P1 contributes to C00 (coeff 1) and C11 (coeff 1).
    # Wait, C00 = ... + P1 ... and C11 = ... + P1 ...
    # So w: C00=1, C11=1 -> [1,0,0,1]
    strassen_2x2_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # Term 2: P2
    # u: A10=1, A11=1 -> [0,0,1,1]
    # v: B00=1 -> [1,0,0,0]
    # w: C10=1 -> [0,0,1,0]
    strassen_2x2_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
    
    # Term 3: P3
    # u: A00=1 -> [1,0,0,0]
    # v: B01=1, B11=-1 -> [0,1,0,-1]
    # w: C01=1 -> [0,1,0,0]
    strassen_2x2_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
    
    # Term 4: P4
    # u: A11=1 -> [0,0,0,1]
    # v: B10=1, B00=-1 -> [-1,0,1,0]
    # w: C00=1, C10=1 -> [1,0,1,0]
    strassen_2x2_terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0]))
    
    # Term 5: P5
    # u: A00=1, A01=1 -> [1,1,0,0]
    # v: B11=1 -> [0,0,0,1]
    # w: C00=-1, C01=1 -> [-1,1,0,0]
    strassen_2x2_terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
    
    # Term 6: P6
    # u: A10=1, A00=-1 -> [-1,0,1,0]
    # v: B00=1, B01=1 -> [1,1,0,0]
    # w: C11=1 -> [0,0,0,1]
    strassen_2x2_terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # Term 7: P7
    # u: A01=1, A11=-1 -> [0,1,0,-1]
    # v: B10=1, B11=1 -> [0,0,1,1]
    # w: C00=1, C11=1 -> [1,0,0,1]
    strassen_2x2_terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1]))
    
    # Now, nest Strassen 4x4 using these 2x2 terms.
    # The 4x4 A is composed of 2x2 blocks:
    # A11 (0,0), A12 (0,1), A21 (1,0), A22 (1,1)
    # Each block is 2x2.
    # Global A index for element (i,j) in 4x4 is i*4+j.
    # But in the 4x5 matrix, it is i*5+j.
    
    # We iterate over the 7 outer Strassen terms.
    # Each outer term specifies linear combos of the 4 blocks.
    # e.g. Outer P1: (A11+A22)(B11+B22)
    # This means we take the 2x2 Strassen decomposition for (A11+A22) and (B11+B22).
    # Wait, no. The outer Strassen combines the *blocks*.
    # The product is a 2x2 matrix of blocks? No, the result is a 4x4 matrix.
    # The outer Strassen computes the 4x4 product using 7 products of 2x2 matrices.
    # Each "product" $P_{outer}$ is a 2x2 matrix multiplication.
    # So for each outer term, we need to perform a 2x2 matrix multiplication.
    # We use the inner Strassen (7 terms) for that.
    
    # So, for each of the 7 outer terms, we add 7 inner terms.
    # Total 49 terms.
    
    # Outer Strassen terms (coefficients for blocks A11, A12, A21, A22)
    # Blocks are indexed 0,1,2,3 corresponding to TL, TR, BL, BR.
    
    outer_terms = []
    # P1: (A0+A3)(B0+B3)
    outer_terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1])) # C0+C3? No, C blocks.
    # C00 (TL) gets P1, C11 (BR) gets P1.
    # So w_outer: [1,0,0,1]
    
    # P2: (A2+A3)B0
    outer_terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0])) # C10 (BL)
    
    # P3: A0(B1-B3)
    outer_terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0])) # C01 (TR)
    
    # P4: A3(B2-B0)
    outer_terms.append(([0,0,0,1], [-1,0,1,0], [1,0,1,0])) # C00 (TL) and C10 (BL)
    
    # P5: (A0+A1)B3
    outer_terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0])) # C00 (TL) - and C01 (TR)
    
    # P6: (A2-A0)(B0+B1)
    outer_terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1])) # C11 (BR)
    
    # P7: (A1-A3)(B2+B3)
    outer_terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,1])) # C00 (TL) and C11 (BR)
    
    # For each outer term (u_out, v_out, w_out):
    # u_out specifies which A blocks to combine.
    # v_out specifies which B blocks to combine.
    # w_out specifies which C blocks are affected.
    
    # For each of the 7 inner terms (u_in, v_in, w_in) from strassen_2x2_terms:
    # The full term is a combination.
    
    for u_out, v_out, w_out in outer_terms:
        for u_in, v_in, w_in in strassen_2x2_terms:
            # Construct global vectors
            u_global = [0] * 20
            v_global = [0] * 25
            w_global = [0] * 20
            
            # u_in has 4 entries corresponding to elements of a 2x2 block.
            # u_out has 4 entries corresponding to the 4 blocks.
            # The coefficient for element (bi, ei) in block bi is u_out[bi] * u_in[ei].
            # Block 0: A rows 0-1, cols 0-1. Global A index i*5+j.
            # Block 1: A rows 0-1, cols 2-3.
            # Block 2: A rows 2-3, cols 0-1.
            # Block 3: A rows 2-3, cols 2-3.
            
            for bi in range(4):
                if u_out[bi] == 0: continue
                # Map block index to global row/col offsets
                # bi=0: r_off=0, c_off=0
                # bi=1: r_off=0, c_off=2
                # bi=2: r_off=2, c_off=0
                # bi=3: r_off=2, c_off=2
                r_off = (bi // 2) * 2
                c_off = (bi % 2) * 2
                
                for ei in range(4):
                    if u_in[ei] == 0: continue
                    # ei=0: (0,0), ei=1: (0,1), ei=2: (1,0), ei=3: (1,1)
                    er = ei // 2
                    ec = ei % 2
                    gi = r_off + er
                    gj = c_off + ec
                    u_global[gi * 5 + gj] = u_out[bi] * u_in[ei]
            
            # v_in, v_out for B
            # B is 4x4.
            # Block 0: B rows 0-1, cols 0-1. Global B index j*5+k.
            # Block 1: B rows 0-1, cols 2-3.
            # Block 2: B rows 2-3, cols 0-1.
            # Block 3: B rows 2-3, cols 2-3.
            for bi in range(4):
                if v_out[bi] == 0: continue
                r_off = (bi // 2) * 2
                c_off = (bi % 2) * 2
                for ei in range(4):
                    if v_in[ei] == 0: continue
                    er = ei // 2
                    ec = ei % 2
                    gj = r_off + er
                    gk = c_off + ec
                    v_global[gj * 5 + gk] = v_out[bi] * v_in[ei]
            
            # w_in, w_out for C
            # C is 4x4.
            # Block 0: C rows 0-1, cols 0-1. Global C index k*4+i.
            # Block 1: C rows 0-1, cols 2-3.
            # Block 2: C rows 2-3, cols 0-1.
            # Block 3: C rows 2-3, cols 2-3.
            for bi in range(4):
                if w_out[bi] == 0: continue
                r_off = (bi // 2) * 2
                c_off = (bi % 2) * 2
                for ei in range(4):
                    if w_in[ei] == 0: continue
                    er = ei // 2
                    ec = ei % 2
                    gk = r_off + er
                    gi = c_off + ec
                    w_global[gk * 4 + gi] = w_out[bi] * w_in[ei]
            
            add_term(u_global, v_global, w_global)
            
    # 2. Schoolbook 4x4x1 part
    # A is 4x4 (cols 0..3), B is 4x1 (row 4, cols 0..4? No, B is 5x5. Row 4 is 1x5.
    # Wait, my split was:
    # A = [A_left (4x4) | A_right (4x1)]
    # B = [B_top (4x5); B_bot (1x5)]
    # Term 1: A_left * B_top. (Done above, 4x4 * 4x5? No, B_top is 4x5.
    # My Strassen part assumed B was 4x4.
    # B_top is 4x5.
    # I used Strassen for 4x4 * 4x4.
    # I need to handle the 5th column of B_top.
    # B_top cols 0..3 handled by Strassen.
    # B_top col 4 is 4x1.
    # A_left (4x4) * B_top_col4 (4x1).
    # This is 4x4x1. Schoolbook rank 16.
    
    for i in range(4):
        for j in range(4):
            u = [0] * 20
            v = [0] * 25
            w = [0] * 20
            u[i * 5 + j] = 1
            v[j * 5 + 4] = 1 # B row j, col 4
            w[4 * 4 + i] = 1 # C row 4, col i? No.
            # C index is k*n + i. k=4, i=i.
            # w[4 * 4 + i] is correct.
            add_term(u, v, w)
            
    # 3. Outer product 4x1 * 1x5
    # A_right (4x1) * B_bot (1x5).
    # A_right is col 4 of A.
    # B_bot is row 4 of B.
    u = [0] * 20
    v = [0] * 25
    w = [0] * 20
    for i in range(4):
        u[i * 5 + 4] = 1
    for k in range(5):
        v[4 * 5 + k] = 1
    for i in range(4):
        for k in range(5):
            w[k * 4 + i] = 1
    add_term(u, v, w)
    
    # Convert lists to numpy arrays
    rank = len(U_terms)
    U = np.array(U_terms).T
    V = np.array(V_terms).T
    W = np.array(W_terms).T
    
    return {"U": U, "V": V, "W": W}
