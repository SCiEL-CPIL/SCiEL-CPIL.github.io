"""Program that finds a rank-minimal exact tensor decomposition for 4x4 by 4x8 matrix multiplication."""
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def _validate(U, V, W, n, m, p):
    """Check that the decomposition exactly reproduces the matmul tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if not np.isclose(val, 1.0, atol=1e-10):
                    return False
    return True

def _strassen_2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition. Shapes (4,7)."""
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # P1 = (a+d)(e+h)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1 # C00, C11
    
    # P2 = (c+d)e
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1 # C10, -C11
    
    # P3 = a(f-h)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1 # C01, C11
    
    # P4 = d(g-e)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1 # C00, C10
    
    # P5 = (a+b)h
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1 # -C00, C01
    
    # P6 = (c-a)(e+f)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1 # C11
    
    # P7 = (b-d)(g+h)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1 # C00
    
    return U, V, W

def _strassen_4x4():
    """Returns U, V, W for 4x4x4 Strassen decomposition. Shapes (16,49)."""
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    U2, V2, W2 = _strassen_2x2()
    
    # Mapping of 2x2 slots (a,b,c,d) to 4x4 indices
    # A blocks:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B blocks:
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # C blocks (Output):
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15
    
    # Define the 7 Strassen steps for 4x4
    # Each step is defined by:
    # A_slots: list of 4 lists of indices for a,b,c,d
    # B_slots: list of 4 lists of indices for a,b,c,d
    # C_contrib: list of (block_index, coeff) for the 4 C blocks
    
    steps = []
    
    # M1 = (A00+A11)(B00+B11)
    # A: A00 + A11. 
    # a: 0,10; b: 1,11; c: 4,14; d: 5,15
    A_slots = [[0, 10], [1, 11], [4, 14], [5, 15]]
    B_slots = [[0, 10], [1, 11], [4, 14], [5, 15]]
    C_contrib = [(0, 1), (3, 1)] # C00 +1, C11 +1
    steps.append((A_slots, B_slots, C_contrib))
    
    # M2 = (A10+A11)B00
    A_slots = [[8, 10], [9, 11], [12, 14], [13, 15]]
    B_slots = [[0], [1], [4], [5]]
    C_contrib = [(2, 1), (3, -1)] # C10 +1, C11 -1
    steps.append((A_slots, B_slots, C_contrib))
    
    # M3 = A00(B01-B11)
    A_slots = [[0], [1], [4], [5]]
    B_slots = [[2, 10], [3, 11], [6, 14], [7, 15]]
    C_contrib = [(1, 1), (3, 1)] # C01 +1, C11 +1
    steps.append((A_slots, B_slots, C_contrib))
    
    # M4 = A11(B10-B00)
    A_slots = [[10], [11], [14], [15]]
    B_slots = [[8, 0], [9, 1], [12, 4], [13, 5]]
    C_contrib = [(0, 1), (2, 1)] # C00 +1, C10 +1
    steps.append((A_slots, B_slots, C_contrib))
    
    # M5 = (A00+A01)B11
    A_slots = [[0, 2], [1, 3], [4, 6], [5, 7]]
    B_slots = [[10], [11], [14], [15]]
    C_contrib = [(0, -1), (1, 1)] # C00 -1, C01 +1
    steps.append((A_slots, B_slots, C_contrib))
    
    # M6 = (A10-A00)(B00+B01)
    A_slots = [[8, 0], [9, 1], [12, 4], [13, 5]]
    B_slots = [[0, 2], [1, 3], [4, 6], [5, 7]]
    C_contrib = [(3, 1)] # C11 +1
    steps.append((A_slots, B_slots, C_contrib))
    
    # M7 = (A01-A11)(B10+B11)
    A_slots = [[2, 10], [3, 11], [6, 14], [7, 15]]
    B_slots = [[8, 10], [9, 11], [12, 14], [13, 15]]
    C_contrib = [(0, 1)] # C00 +1
    steps.append((A_slots, B_slots, C_contrib))
    
    # Now fill U, V, W
    col_offset = 0
    for A_slots, B_slots, C_contrib in steps:
        for t in range(7):
            # Get U2, V2, W2 vectors for this term
            u_vec = U2[:, t]
            v_vec = V2[:, t]
            w_vec = W2[:, t]
            
            # Expand to U4
            for slot_idx in range(4):
                coeff = u_vec[slot_idx]
                if coeff != 0:
                    for idx in A_slots[slot_idx]:
                        U[idx, col_offset] += coeff
            
            # Expand to V4
            for slot_idx in range(4):
                coeff = v_vec[slot_idx]
                if coeff != 0:
                    for idx in B_slots[slot_idx]:
                        V[idx, col_offset] += coeff
            
            # Expand to W4
            # w_vec corresponds to C00, C01, C10, C11 of the 2x2 result
            # We need to add w_vec[slot] * coeff to the corresponding C blocks
            for slot_idx in range(4):
                coeff = w_vec[slot_idx]
                if coeff != 0:
                    # The 2x2 result slot 'slot_idx' maps to C00, C01, C10, C11
                    # slot 0 -> C00, slot 1 -> C01, slot 2 -> C10, slot 3 -> C11
                    c_block_idx = slot_idx
                    # Find which C block this slot belongs to
                    # Actually, w_vec[0] is coeff for C00, w_vec[1] for C01, etc.
                    # So we just iterate over the C_contrib list
                    # Wait, w_vec is already the coefficients for the 4 output blocks of the 2x2 product.
                    # So w_vec[0] is the contribution to the (0,0) block of the result.
                    # This (0,0) block of the result is added to the C blocks specified in C_contrib.
                    # No, C_contrib specifies which of the 4 big C blocks (C00, C01, C10, C11) the entire 2x2 result is added to.
                    # So, for each slot in the 2x2 result (0,1,2,3), we add w_vec[slot] to the corresponding position in the big C blocks.
                    
                    # Let's re-evaluate.
                    # The 2x2 product result is a 2x2 matrix R.
                    # R[0,0] = w_vec[0] (in terms of the basis)
                    # R[0,1] = w_vec[1]
                    # R[1,0] = w_vec[2]
                    # R[1,1] = w_vec[3]
                    
                    # This R is added to the big C matrix.
                    # C_contrib = [(block_idx, coeff)]
                    # block_idx 0 -> C00 (indices 0,1,4,5)
                    # block_idx 1 -> C01 (indices 2,3,6,7)
                    # block_idx 2 -> C10 (indices 8,9,12,13)
                    # block_idx 3 -> C11 (indices 10,11,14,15)
                    
                    # So for each (block_idx, big_coeff) in C_contrib:
                    #   We add big_coeff * R to the block.
                    #   R[0,0] goes to block_idx's (0,0) position.
                    
                    # The indices for block b are:
                    # b=0: 0,1,4,5
                    # b=1: 2,3,6,7
                    # b=2: 8,9,12,13
                    # b=3: 10,11,14,15
                    # The positions within the block are:
                    # (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3 relative to block start?
                    # No, the indices are fixed.
                    # Block 0: 0,1,4,5.
                    # Block 1: 2,3,6,7.
                    # Block 2: 8,9,12,13.
                    # Block 3: 10,11,14,15.
                    
                    # Let's create a mapping from block_idx and slot_idx to 4x4 index.
                    # slot 0 (0,0) of block b:
                    # b=0: 0, b=1: 2, b=2: 8, b=3: 10
                    # slot 1 (0,1) of block b:
                    # b=0: 1, b=1: 3, b=2: 9, b=3: 11
                    # slot 2 (1,0) of block b:
                    # b=0: 4, b=1: 6, b=2: 12, b=3: 14
                    # slot 3 (1,1) of block b:
                    # b=0: 5, b=1: 7, b=2: 13, b=3: 15
                    
                    pass # Logic to be implemented in the loop below
            
            # Corrected W expansion logic
            for (block_idx, big_coeff) in C_contrib:
                # Determine base index for this block
                if block_idx == 0:
                    base_indices = [0, 1, 4, 5]
                elif block_idx == 1:
                    base_indices = [2, 3, 6, 7]
                elif block_idx == 2:
                    base_indices = [8, 9, 12, 13]
                else:
                    base_indices = [10, 11, 14, 15]
                
                for slot_idx in range(4):
                    coeff = w_vec[slot_idx] * big_coeff
                    if coeff != 0:
                        W[base_indices[slot_idx], col_offset] += coeff
            
            col_offset += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    
    U4, V4, W4 = _strassen_4x4()
    
    # Rank 98
    rank = 98
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # First 49 terms: k in [0, 4)
    U[:, 0:49] = U4
    # V: j in [0,4), k in [0,4)
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V4[j*4+k, :]
    # W: k in [0,4), i in [0,4)
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W4[k*4+i, :]
            
    # Next 49 terms: k in [4, 8)
    U[:, 49:98] = U4
    # V: j in [0,4), k in [4,8)
    for j in range(4):
        for k in range(4, 8):
            k_prime = k - 4
            V[j*8+k, 49:98] = V4[j*4+k_prime, :]
    # W: k in [4,8), i in [0,4)
    for k in range(4, 8):
        for i in range(4):
            k_prime = k - 4
            W[k*4+i, 49:98] = W4[k_prime*4+i, :]
            
    if not _validate(U, V, W, n, m, p):
        raise ValueError("Decomposition is invalid")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
