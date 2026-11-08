import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication.
    Uses the structure of 7 blocks of 2x2 Strassen.
    Returns {'U': ..., 'V': ..., 'W': ...}
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Helper to get indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Strassen 2x2 coefficients
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # The 7 terms for a 2x2 multiplication (indices 0..6)
    # We map these to the 4x4 blocks.
    # A blocks: A00 (0-1, 0-1), A01 (0-1, 2-3), A10 (2-3, 0-1), A11 (2-3, 2-3)
    # B blocks: B00, B01, B10, B11
    # C blocks: C00, C01, C10, C11
    
    # For each of the 7 "super" terms (block level), we decompose into 7 "sub" terms (element level).
    # Total 49 terms.
    
    # Define the block-level Strassen operations
    # Each operation is a linear combination of A-blocks and B-blocks.
    # A-blocks are 2x2 matrices.
    # We need to expand these into individual elements.
    
    # Let's define the sub-terms for a single 2x2 Strassen.
    # Input: 2x2 A, 2x2 B.
    # Output: 7 terms (U, V, W) for the 2x2 product.
    # We will hardcode the 7x7 expansion.
    
    # Strassen 2x2 decomposition details:
    # Term 0: U: a11, a22; V: b11, b22; W: c11, c22 (coeffs: 1, -1 for c11? No, C11 = M1+M4-M5+M7)
    # This is getting complex to hardcode manually. 
    # Better to generate it programmatically from the block definitions.
    
    # Block definitions:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # Block-level Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Each Pi is a product of two 2x2 matrices.
    # Each Pi can be decomposed into 7 rank-1 terms using Strassen.
    
    # Let's define the 7 rank-1 terms for a 2x2 matrix product.
    # For a generic 2x2 product X*Y, the 7 terms are:
    # t0: (x00+x11)(y00+y11)
    # t1: (x10+x11)y00
    # t2: x00(y01-y11)
    # t3: x11(y10-y00)
    # t4: (x00+x01)y11
    # t5: (x10-x00)(y00+y01)
    # t6: (x01-x11)(y10+y11)
    
    # We need to map the block elements to global indices.
    
    # Function to add a 2x2 Strassen decomposition for a specific block pair
    # A_block: list of (row, col) indices for the 2x2 A block
    # B_block: list of (row, col) indices for the 2x2 B block
    # C_contribs: list of (C_row, C_col, coeff) for each of the 7 Strassen terms?
    # No, the C contribution depends on which Pi it is.
    
    # Let's iterate through the 7 Pi terms.
    
    # Define A block indices
    # A00: (0,0), (0,1), (1,0), (1,1)
    # A01: (0,2), (0,3), (1,2), (1,3)
    # A10: (2,0), (2,1), (3,0), (3,1)
    # A11: (2,2), (2,3), (3,2), (3,3)
    
    A_blocks = [
        [(0,0), (0,1), (1,0), (1,1)], # A00
        [(0,2), (0,3), (1,2), (1,3)], # A01
        [(2,0), (2,1), (3,0), (3,1)], # A10
        [(2,2), (2,3), (3,2), (3,3)]  # A11
    ]
    
    # Define B block indices
    B_blocks = [
        [(0,0), (0,1), (1,0), (1,1)], # B00
        [(0,2), (0,3), (1,2), (1,3)], # B01
        [(2,0), (2,1), (3,0), (3,1)], # B10
        [(2,2), (2,3), (3,2), (3,3)]  # B11
    ]
    
    # Define C block indices (for output)
    C_blocks = [
        [(0,0), (0,1), (1,0), (1,1)], # C00
        [(0,2), (0,3), (1,2), (1,3)], # C01
        [(2,0), (2,1), (3,0), (3,1)], # C10
        [(2,2), (2,3), (3,2), (3,3)]  # C11
    ]
    
    # Strassen 2x2 term definitions (coeffs for X and Y)
    # X is A-like, Y is B-like.
    # t0: x00, x11; y00, y11
    # t1: x10, x11; y00
    # t2: x00; y01, y11 (-1)
    # t3: x11; y10, y00 (-1)
    # t4: x00, x01; y11
    # t5: x10, x00(-1); y00, y01
    # t6: x01, x11(-1); y10, y11
    
    # We represent X coeffs as dict {(r,c): coeff}
    # We represent Y coeffs as dict {(r,c): coeff}
    # We represent Z coeffs (C) as dict {(r,c): coeff}
    
    strassen_terms = [
        (
            {(0,0):1, (1,1):1}, 
            {(0,0):1, (1,1):1}, 
            None # Z is determined by which Pi this is
        ),
        (
            {(1,0):1, (1,1):1}, 
            {(0,0):1}, 
            None
        ),
        (
            {(0,0):1}, 
            {(0,1):1, (1,1):-1}, 
            None
        ),
        (
            {(1,1):1}, 
            {(1,0):1, (0,0):-1}, 
            None
        ),
        (
            {(0,0):1, (0,1):1}, 
            {(1,1):1}, 
            None
        ),
        (
            {(1,0):1, (0,0):-1}, 
            {(0,0):1, (0,1):1}, 
            None
        ),
        (
            {(0,1):1, (1,1):-1}, 
            {(1,0):1, (1,1):1}, 
            None
        )
    ]
    
    # Mapping from Pi index to C block contributions
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    # Pi indices: 0..6 corresponding to P1..P7
    
    # C_block_contribs[Pi_idx][SubTerm_idx] -> dict of (c_row, c_col, coeff)
    # Note: The sub-term Z (C) structure is the same for all Strassen decompositions of a 2x2 product.
    # The 2x2 product Z = XY has Z00, Z01, Z10, Z11.
    # Strassen 2x2 Z construction:
    # Z00 = t0 + t3 - t4 + t6
    # Z01 = t2 + t4
    # Z10 = t1 + t3
    # Z11 = t0 - t1 + t2 + t5
    
    # So for each Pi, we map the 2x2 result Z to the appropriate 2x2 block of C.
    
    # Pi=0 (P1) -> C00 (coeff 1), C11 (coeff 1)
    # Pi=1 (P2) -> C10 (coeff 1), C11 (coeff -1)
    # Pi=2 (P3) -> C01 (coeff 1), C11 (coeff 1)
    # Pi=3 (P4) -> C00 (coeff 1), C10 (coeff 1)
    # Pi=4 (P5) -> C00 (coeff -1), C01 (coeff 1)
    # Pi=5 (P6) -> C11 (coeff 1)
    # Pi=6 (P7) -> C00 (coeff 1)
    
    # Let's define the mapping of 2x2 Strassen sub-terms (t0..t6) to local C coords (0..1, 0..1)
    # t0 -> C00 (+1), C11 (+1)
    # t1 -> C10 (+1), C11 (-1)
    # t2 -> C01 (+1), C11 (+1)
    # t3 -> C00 (+1), C10 (+1)
    # t4 -> C00 (-1), C01 (+1)
    # t5 -> C11 (+1)
    # t6 -> C00 (+1)
    
    # Wait, let's re-verify the 2x2 Strassen output mapping.
    # Z00 = t0 + t3 - t4 + t6
    # Z01 = t2 + t4
    # Z10 = t1 + t3
    # Z11 = t0 - t1 + t2 + t5
    
    # So:
    # t0 contributes to Z00 (+1), Z11 (+1)
    # t1 contributes to Z10 (+1), Z11 (-1)
    # t2 contributes to Z01 (+1), Z11 (+1)
    # t3 contributes to Z00 (+1), Z10 (+1)
    # t4 contributes to Z00 (-1), Z01 (+1)
    # t5 contributes to Z11 (+1)
    # t6 contributes to Z00 (+1)
    
    # Local C coords: (0,0)=Z00, (0,1)=Z01, (1,0)=Z10, (1,1)=Z11
    
    local_c_map = {
        0: { (0,0): 1, (1,1): 1 },
        1: { (1,0): 1, (1,1): -1 },
        2: { (0,1): 1, (1,1): 1 },
        3: { (0,0): 1, (1,0): 1 },
        4: { (0,0): -1, (0,1): 1 },
        5: { (1,1): 1 },
        6: { (0,0): 1 }
    }
    
    # Now, for each Pi (0..6), we have a pair of A-block and B-block.
    # Pi=0: A00+A11, B00+B11. 
    # This is a linear combination. We can't just pick one block.
    # We must expand (A00+A11) into individual elements.
    # A00+A11 is a 2x2 matrix where element (r,c) is A00[r,c] + A11[r,c].
    
    # So for Pi=0, the X matrix is A00+A11.
    # X[0,0] = A00[0,0] + A11[0,0] -> indices (0,0) and (2,2)
    # X[0,1] = A00[0,1] + A11[0,1] -> indices (0,1) and (2,3)
    # X[1,0] = A00[1,0] + A11[1,0] -> indices (1,0) and (3,2)
    # X[1,1] = A00[1,1] + A11[1,1] -> indices (1,1) and (3,3)
    
    # We need a generic way to generate the U, V entries for a linear combination of blocks.
    
    # Let's define the block combinations for each Pi.
    # Pi 0: X = A00 + A11, Y = B00 + B11
    # Pi 1: X = A10 + A11, Y = B00
    # Pi 2: X = A00, Y = B01 - B11
    # Pi 3: X = A11, Y = B10 - B00
    # Pi 4: X = A00 + A01, Y = B11
    # Pi 5: X = A10 - A00, Y = B00 + B01
    # Pi 6: X = A01 - A11, Y = B10 + B11
    
    # We'll create a helper to get the global indices for a local 2x2 coordinate in a block.
    # block_id: 0..3
    # local (r, c): 0..1
    
    def get_global_idx(block_id, r, c):
        # A blocks
        if block_id == 0: return (r, c)
        if block_id == 1: return (r, c+2)
        if block_id == 2: return (r+2, c)
        if block_id == 3: return (r+2, c+2)
        
    # Define Pi definitions
    # Each Pi is a list of (coeff, block_id) for X and Y.
    
    Pi_defs = [
        (
            [(1, 0), (1, 3)], # X: A00 + A11
            [(1, 0), (1, 3)]  # Y: B00 + B11
        ),
        (
            [(1, 2), (1, 3)], # X: A10 + A11
            [(1, 0)]          # Y: B00
        ),
        (
            [(1, 0)],         # X: A00
            [(1, 1), (-1, 3)] # Y: B01 - B11
        ),
        (
            [(1, 3)],         # X: A11
            [(1, 2), (-1, 0)] # Y: B10 - B00
        ),
        (
            [(1, 0), (1, 1)], # X: A00 + A01
            [(1, 3)]          # Y: B11
        ),
        (
            [(1, 2), (-1, 0)],# X: A10 - A00
            [(1, 0), (1, 1)]  # Y: B00 + B01
        ),
        (
            [(1, 1), (-1, 3)],# X: A01 - A11
            [(1, 2), (1, 3)]  # Y: B10 + B11
        )
    ]
    
    # C block mapping for each Pi
    # Pi 0 -> C00 (1), C11 (1)
    # Pi 1 -> C10 (1), C11 (-1)
    # Pi 2 -> C01 (1), C11 (1)
    # Pi 3 -> C00 (1), C10 (1)
    # Pi 4 -> C00 (-1), C01 (1)
    # Pi 5 -> C11 (1)
    # Pi 6 -> C00 (1)
    
    C_block_map = [
        [(0, 1), (3, 1)], # P1
        [(2, 1), (3, -1)],# P2
        [(1, 1), (3, 1)], # P3
        [(0, 1), (2, 1)], # P4
        [(0, -1), (1, 1)],# P5
        [(3, 1)],         # P6
        [(0, 1)]          # P7
    ]
    
    # Now build the matrices
    for pi_idx in range(7):
        x_defs, y_defs = Pi_defs[pi_idx]
        c_defs = C_block_map[pi_idx]
        
        for sub_idx in range(7):
            rank_idx = pi_idx * 7 + sub_idx
            
            x_coeffs, y_coeffs, _ = strassen_terms[sub_idx]
            
            # Build U contributions
            # For each local (r,c) in x_coeffs, add to U[global_r, global_c, rank_idx]
            for (lr, lc), coeff in x_coeffs.items():
                # x_defs is a list of (block_coeff, block_id)
                # We need to sum the contributions from each block
                total_coeff = 0
                # Wait, x_coeffs is the Strassen term definition for the 2x2 product.
                # The "X" in the Strassen term is the 2x2 matrix (A00+A11 etc).
                # So we iterate over the blocks in x_defs.
                
                # Actually, my loop structure is slightly off.
                # x_coeffs defines which elements of the 2x2 X matrix are used.
                # X is formed by x_defs.
                # So for a specific local element (lr, lc) of X:
                # X[lr, lc] = sum(block_coeff * A_block[global_idx])
                
                # We already extracted x_coeffs for this sub_idx.
                # Now we need to expand X[lr, lc].
                
                # Let's restructure:
                # For each sub_idx:
                #   For each (lr, lc) in x_coeffs:
                #      coeff_x = x_coeffs[(lr, lc)]
                #      For each (bc, bid) in x_defs:
                #          gr, gc = get_global_idx(bid, lr, lc)
                #          U[u_idx(gr, gc), rank_idx] += coeff_x * bc
                
                pass # This logic is done inside the loop below
                
            # Corrected Logic:
            
            # 1. Determine U contributions
            for (lr, lc), coeff_x in x_coeffs.items():
                for (bc, bid) in x_defs:
                    gr, gc = get_global_idx(bid, lr, lc)
                    U[u_idx(gr, gc), rank_idx] += coeff_x * bc
            
            # 2. Determine V contributions
            for (lr, lc), coeff_y in y_coeffs.items():
                for (bc, bid) in y_defs:
                    gr, gc = get_global_idx(bid, lr, lc)
                    V[v_idx(gr, gc), rank_idx] += coeff_y * bc
            
            # 3. Determine W contributions
            # The sub-term sub_idx contributes to specific local C coordinates (from local_c_map).
            # For each (lc_r, lc_c, coeff_c) in local_c_map[sub_idx]:
            #   This contributes to the C block defined by c_defs.
            #   Wait, c_defs is a list of (c_block_id, coeff_block).
            #   The local C element (lc_r, lc_c) maps to global C element (gr, gc).
            #   The global C element is in block c_block_id at local pos (lc_r, lc_c).
            
            for (lc_r, lc_c), coeff_c in local_c_map[sub_idx].items():
                for (cbid, coeff_block) in c_defs:
                    # Get global index in C block
                    # C blocks are same layout as A blocks
                    gr, gc = get_global_idx(cbid, lc_r, lc_c)
                    # W index is w_idx(gc, gr) ? No, W[k*n+i, r].
                    # C is (i, k). Global row i, global col k.
                    # w_idx(k, i) = k * 4 + i.
                    W[w_idx(gc, gr), rank_idx] += coeff_c * coeff_block

    # Verification
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for r in range(49):
                    val += U[u_idx(i,j), r] * V[v_idx(j,k), r] * W[w_idx(k,i), r]
                if abs(val - 1.0) > 1e-9:
                    return {"U": np.zeros((16, 64)).tolist(), "V": np.zeros((16, 64)).tolist(), "W": np.zeros((16, 64)).tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
