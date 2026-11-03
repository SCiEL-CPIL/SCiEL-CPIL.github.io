import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen coefficients for 2x2x2
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
    
    def add_strassen_block(offset, a_j_start, b_j_start, b_k_start, c_k_start):
        # a_j_start: 0 or 2 (for A_L or A_R)
        # b_j_start: 0 or 2 (for B top or bottom)
        # b_k_start: 0 or 2 (for B left or right)
        # c_k_start: 0 or 2 (for C left or right)
        
        # A indices: i in 0..1, j in a_j_start..a_j_start+1
        # B indices: j in b_j_start..b_j_start+1, k in b_k_start..b_k_start+1
        # C indices: i in 0..1, k in c_k_start..c_k_start+1
        
        # Local indices within the 2x2 blocks
        # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 relative to block
        # But we map to global.
        
        # Term 1: (A00+A11)(B00+B11)
        # A00: i=0, j=a_j_start
        # A11: i=1, j=a_j_start+1
        # B00: j=b_j_start, k=b_k_start
        # B11: j=b_j_start+1, k=b_k_start+1
        
        r = offset + 0
        U[0*m + a_j_start, r] += 1
        U[1*m + a_j_start + 1, r] += 1
        V[b_j_start*p + b_k_start, r] += 1
        V[(b_j_start+1)*p + b_k_start + 1, r] += 1
        W[c_k_start*n + 0, r] += 1 # C00
        W[(c_k_start+1)*n + 1, r] += 1 # C11
        
        # Term 2: (A10+A11)B00
        r = offset + 1
        U[1*m + a_j_start, r] += 1
        U[1*m + a_j_start + 1, r] += 1
        V[b_j_start*p + b_k_start, r] += 1
        W[(c_k_start)*n + 1, r] += 1 # C10
        W[(c_k_start+1)*n + 1, r] -= 1 # C11
        
        # Term 3: A00(B01-B11)
        r = offset + 2
        U[0*m + a_j_start, r] += 1
        V[b_j_start*p + b_k_start + 1, r] += 1
        V[(b_j_start+1)*p + b_k_start + 1, r] -= 1
        W[c_k_start*n + 0, r] += 1 # C00
        W[(c_k_start+1)*n + 0, r] += 1 # C01
        
        # Term 4: A11(B10-B00)
        r = offset + 3
        U[1*m + a_j_start + 1, r] += 1
        V[(b_j_start+1)*p + b_k_start, r] += 1
        V[b_j_start*p + b_k_start, r] -= 1
        W[c_k_start*n + 0, r] += 1 # C00
        W[(c_k_start)*n + 1, r] += 1 # C10
        
        # Term 5: (A00+A01)B11
        r = offset + 4
        U[0*m + a_j_start, r] += 1
        U[0*m + a_j_start + 1, r] += 1
        V[(b_j_start+1)*p + b_k_start + 1, r] += 1
        W[c_k_start*n + 0, r] -= 1 # C00
        W[(c_k_start+1)*n + 0, r] += 1 # C01
        
        # Term 6: (A10-A00)(B00+B01)
        r = offset + 5
        U[1*m + a_j_start, r] += 1
        U[0*m + a_j_start, r] -= 1
        V[b_j_start*p + b_k_start, r] += 1
        V[b_j_start*p + b_k_start + 1, r] += 1
        W[(c_k_start+1)*n + 1, r] += 1 # C11
        
        # Term 7: (A01-A11)(B10+B11)
        r = offset + 6
        U[0*m + a_j_start + 1, r] += 1
        U[1*m + a_j_start + 1, r] -= 1
        V[(b_j_start+1)*p + b_k_start, r] += 1
        V[(b_j_start+1)*p + b_k_start + 1, r] += 1
        W[c_k_start*n + 0, r] += 1 # C00
        W[(c_k_start+1)*n + 1, r] += 1 # C11

    # Block 1: A_L (j=0,1) * B_LL (j=0,1, k=0,1) -> C_L (k=0,1)
    add_strassen_block(0, 0, 0, 0, 0)
    
    # Block 2: A_R (j=2,3) * B_RL (j=2,3, k=0,1) -> C_L (k=0,1)
    add_strassen_block(7, 2, 2, 0, 0)
    
    # Block 3: A_L (j=0,1) * B_LR (j=0,1, k=2,3) -> C_R (k=2,3)
    add_strassen_block(14, 0, 0, 2, 2)
    
    # Block 4: A_R (j=2,3) * B_RR (j=2,3, k=2,3) -> C_R (k=2,3)
    add_strassen_block(21, 2, 2, 2, 2)
    
    # Now for the second half of B (k=4..7) and C (k=4..7)
    # B is 4x8.
    # B[:, 4:8] corresponds to k=4,5,6,7.
    # We split B[:, 4:8] into two 4x4 blocks? No, 4x4 is j=0..3, k=4..7.
    # We split this 4x4 into four 2x2 blocks.
    # B_LL2: j=0,1, k=4,5
    # B_LR2: j=0,1, k=6,7
    # B_RL2: j=2,3, k=4,5
    # B_RR2: j=2,3, k=6,7
    
    # C[:, 4:8] corresponds to k=4,5,6,7.
    # C_L2: k=4,5
    # C_R2: k=6,7
    
    # Block 5: A_L * B_LL2 -> C_L2
    add_strassen_block(28, 0, 0, 4, 4)
    
    # Block 6: A_R * B_RL2 -> C_L2
    add_strassen_block(35, 2, 2, 4, 4)
    
    # Block 7: A_L * B_LR2 -> C_R2
    add_strassen_block(42, 0, 0, 6, 6)
    
    # Block 8: A_R * B_RR2 -> C_R2
    add_strassen_block(49, 2, 2, 6, 6)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
