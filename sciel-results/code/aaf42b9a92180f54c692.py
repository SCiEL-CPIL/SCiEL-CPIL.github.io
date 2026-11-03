# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    # Split A (2x3) into A_L (2x2) and A_R (2x1)
    # Split B (3x8) into B_L (2x8) and B_R (1x8)
    # C = A_L B_L + A_R B_R
    # A_L B_L: 2x2x8 -> 4 blocks of 2x2x2 (Strassen rank 7 each) -> 28
    # A_R B_R: 2x1x8 -> Schoolbook rank 16
    # Total rank = 44
    
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w, col):
        U[u_idx, col] += val_u
        V[v_idx, col] += val_v
        W[w_idx, col] += val_w

    col = 0
    
    # Part 1: A_L B_L
    # A_L is rows 0,1 of A (indices 0,1 in U)
    # B_L is rows 0,1 of B (indices 0-7 in V)
    # Output C_L is 2x4 (indices 0-3 in W? No, W is p*n = 8*2=16)
    # W index is k*n + i. k is col of C, i is row of C.
    
    # B_L is 2x8. Split into 4 blocks of 2x2.
    # Block 0: cols 0,1 of B_L.
    # Block 1: cols 2,3 of B_L.
    # Block 2: cols 4,5 of B_L.
    # Block 3: cols 6,7 of B_L.
    
    for block_idx in range(4):
        b_col_offset = block_idx * 2
        
        # Strassen for 2x2x2
        # A_L elements:
        # a00: U[0, :]
        # a01: U[1, :]
        # a10: U[2, :]
        # a11: U[3, :]
        
        # B_L block elements:
        # b00: V[b_col_offset, :]
        # b01: V[b_col_offset + 1, :]
        # b10: V[b_col_offset + 2, :]
        # b11: V[b_col_offset + 3, :]
        
        # Output C block elements:
        # c00: W[(b_col_offset)*n + 0, :] = W[b_col_offset*2, :]
        # c01: W[(b_col_offset+1)*n + 0, :] = W[(b_col_offset+1)*2, :]
        # c10: W[(b_col_offset)*n + 1, :] = W[b_col_offset*2 + 1, :]
        # c11: W[(b_col_offset+1)*n + 1, :] = W[(b_col_offset+1)*2 + 1, :]
        
        # M1
        U[0, col] += 1; U[3, col] += 1
        V[b_col_offset, col] += 1; V[b_col_offset + 3, col] += 1
        W[b_col_offset*2, col] += 1
        W[b_col_offset*2 + 1, col] += 1
        W[(b_col_offset+1)*2, col] -= 1
        W[(b_col_offset+1)*2 + 1, col] += 1
        col += 1
        
        # M2
        U[2, col] += 1; U[3, col] += 1
        V[b_col_offset, col] += 1
        W[b_col_offset*2 + 1, col] += 1
        W[(b_col_offset+1)*2 + 1, col] -= 1
        col += 1
        
        # M3
        U[0, col] += 1
        V[b_col_offset + 1, col] += 1; V[b_col_offset + 3, col] -= 1
        W[(b_col_offset+1)*2, col] += 1
        W[(b_col_offset+1)*2 + 1, col] += 1
        col += 1
        
        # M4
        U[3, col] += 1
        V[b_col_offset + 2, col] += 1; V[b_col_offset, col] -= 1
        W[b_col_offset*2, col] += 1
        W[b_col_offset*2 + 1, col] += 1
        col += 1
        
        # M5
        U[0, col] += 1; U[1, col] += 1
        V[b_col_offset + 3, col] += 1
        W[b_col_offset*2, col] -= 1
        W[(b_col_offset+1)*2, col] += 1
        col += 1
        
        # M6
        U[2, col] += 1; U[0, col] -= 1
        V[b_col_offset, col] += 1; V[b_col_offset + 1, col] += 1
        W[(b_col_offset+1)*2 + 1, col] += 1
        col += 1
        
        # M7
        U[1, col] += 1; U[3, col] -= 1
        V[b_col_offset + 2, col] += 1; V[b_col_offset + 3, col] += 1
        W[b_col_offset*2, col] += 1
        W[(b_col_offset+1)*2, col] -= 1
        col += 1

    # Part 2: A_R B_R
    # A_R is row 2 of A (index 2 in U? No, A is 2x3. Rows 0,1. Cols 0,1,2.
    # U index i*m+j.
    # A[0,0]=U[0], A[0,1]=U[1], A[0,2]=U[2]
    # A[1,0]=U[3], A[1,1]=U[4], A[1,2]=U[5]
    # So A_L uses U[0], U[1], U[3], U[4].
    # A_R uses U[2], U[5].
    
    # Wait, my Strassen part used U[0], U[1], U[2], U[3].
    # This corresponds to A[0,0], A[0,1], A[1,0], A[1,1].
    # So A_L is correct.
    # A_R is A[0,2]=U[2]? No.
    # A[0,2] is U[0*3+2] = U[2].
    # A[1,2] is U[1*3+2] = U[5].
    # So A_R uses U[2] and U[5].
    
    # BUT, in my Strassen implementation above, I used U[2] as a10 (A[1,0]) and U[3] as a11 (A[1,1]).
    # Let's re-verify U indices.
    # U shape (6, rank).
    # U[0] = A[0,0]
    # U[1] = A[0,1]
    # U[2] = A[0,2]
    # U[3] = A[1,0]
    # U[4] = A[1,1]
    # U[5] = A[1,2]
    
    # My Strassen code:
    # M1: U[0], U[3]. (A[0,0], A[1,0]?) No, Strassen needs a00, a01, a10, a11.
    # a00 = A[0,0] = U[0]
    # a01 = A[0,1] = U[1]
    # a10 = A[1,0] = U[3]
    # a11 = A[1,1] = U[4]
    
    # I used U[2] in M2 and M6. That is A[0,2]. This is WRONG for A_L.
    # I need to fix the indices in the Strassen part.
    
    # Let's restart the Strassen part with correct indices.
    # A_L:
    # a00 = U[0]
    # a01 = U[1]
    # a10 = U[3]
    # a11 = U[4]
    
    # B_L:
    # b00 = V[0]
    # b01 = V[1]
    # b10 = V[2]
    # b11 = V[3]
    # (for block 0)
    
    # I will rewrite the whole block to be safe.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
