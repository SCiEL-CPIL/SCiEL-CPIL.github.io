"""FTW-inspired decomposition for 2x4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]

def _frac(x):
    """Represent half-integer as float."""
    return float(x)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 6
    
    # We use a rank-28 decomposition based on Kronecker product structure.
    # The key insight: 2x4x6 can be viewed as (2x2x2) ⊗ (1x2x3) in a structured way.
    # 
    # Actually, let's think about this more carefully. The standard approach gives rank 42.
    # But we can do better by exploiting shared structure.
    #
    # Alternative: Use the fact that we're multiplying 2x4 by 4x6.
    # We can partition the middle dimension m=4 into two blocks of 2.
    # A = [A_left, A_right] where each is 2x2
    # B = [B_top; B_bot] where each is 2x6
    # C = A_left * B_top + A_right * B_bot
    #
    # This is a sum of two 2x2x6 multiplications.
    # Each 2x2x6 multiplication: we can further partition p=6 into three blocks of 2.
    # So each 2x2x6 = sum of three 2x2x2 multiplications.
    # Total: 2 * 3 * 7 = 42 (same as before).
    #
    # To beat 42, we need a smarter approach. Let's try rank 28.
    #
    # Actually, let me reconsider. The known lower bound for 2x4x6 is at least 16 (from n*m*p).
    # The trivial rank is 2*4*6 = 48.
    # Strassen-based gives 42.
    #
    # Let me try a different decomposition. What if we use rank 24?
    # 
    # Actually, let's stick with a clean rank-42 implementation but verify it works.
    # The previous implementation had bugs. Let me write it carefully.
    
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen for 2x2x2:
    # Given X (2x2), Y (2x2), compute Z = X*Y
    # 7 products:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    # For each of the 3 column blocks (r=0,1,2):
    #   Compute Cr = A1*Br_top + A2*Br_bot
    #   where A1, A2 are 2x2 (left/right halves of A)
    #   Br_top, Br_bot are 2x2 (top/bottom halves of B's 4x2 block)
    
    col = 0
    for r in range(3):
        # b_offset = 2*r (column offset in B and C)
        b_off = 2 * r
        
        # Part 1: Strassen for A1 * Br_top
        # A1[i,j] -> U row: i*4 + j
        # Br_top[j,kb] -> V row: j*6 + (b_off + kb)
        # Output (i,kb) -> W row: (b_off + kb)*2 + i
        
        # M1: (A1[0,0]+A1[1,1]) * (Br_top[0,0]+Br_top[1,1])
        U[0*4+0][col] += 1; U[1*4+1][col] += 1
        V[0*6+b_off+0][col] += 1; V[1*6+b_off+1][col] += 1
        col += 1
        # M2: (A1[1,0]+A1[1,1]) * Br_top[0,0]
        U[1*4+0][col] += 1; U[1*4+1][col] += 1
        V[0*6+b_off+0][col] += 1
        col += 1
        # M3: A1[0,0] * (Br_top[0,1]-Br_top[1,1])
        U[0*4+0][col] += 1
        V[0*6+b_off+1][col] += 1; V[1*6+b_off+1][col] += -1
        col += 1
        # M4: A1[1,1] * (Br_top[1,0]-Br_top[0,0])
        U[1*4+1][col] += 1
        V[1*6+b_off+0][col] += 1; V[0*6+b_off+0][col] += -1
        col += 1
        # M5: (A1[0,0]+A1[0,1]) * Br_top[1,1]
        U[0*4+0][col] += 1; U[0*4+1][col] += 1
        V[1*6+b_off+1][col] += 1
        col += 1
        # M6: (A1[1,0]-A1[0,0]) * (Br_top[0,0]+Br_top[0,1])
        U[1*4+0][col] += 1; U[0*4+0][col] += -1
        V[0*6+b_off+0][col] += 1; V[0*6+b_off+1][col] += 1
        col += 1
        # M7: (A1[0,1]-A1[1,1]) * (Br_top[1,0]+Br_top[1,1])
        U[0*4+1][col] += 1; U[1*4+1][col] += -1
        V[1*6+b_off+0][col] += 1; V[1*6+b_off+1][col] += 1
        col += 1
        
        # Part 2: Strassen for A2 * Br_bot
        # A2[i,j] -> U row: i*4 + (j+2)
        # Br_bot[j,kb] -> V row: (j+2)*6 + (b_off + kb)
        # Output (i,kb) -> W row: (b_off + kb)*2 + i
        
        # M8: (A2[0,0]+A2[1,1]) * (Br_bot[0,0]+Br_bot[1,1])
        U[0*4+2][col] += 1; U[1*4+3][col] += 1
        V[2*6+b_off+0][col] += 1; V[3*6+b_off+1][col] += 1
        col += 1
        # M9: (A2[1,0]+A2[1,1]) * Br_bot[0,0]
        U[1*4+2][col] += 1; U[1*4+3][col] += 1
        V[2*6+b_off+0][col] += 1
        col += 1
        # M10: A2[0,0] * (Br_bot[0,1]-Br_bot[1,1])
        U[0*4+2][col] += 1
        V[2*6+b_off+1][col] += 1; V[3*6+b_off+1][col] += -1
        col += 1
        # M11: A2[1,1] * (Br_bot[1,0]-Br_bot[0,0])
        U[1*4+3][col] += 1
        V[3*6+b_off+0][col] += 1; V[2*6+b_off+0][col] += -1
        col += 1
        # M12: (A2[0,0]+A2[0,1]) * Br_bot[1,1]
        U[0*4+2][col] += 1; U[0*4+3][col] += 1
        V[3*6+b_off+1][col] += 1
        col += 1
        # M13: (A2[1,0]-A2[0,0]) * (Br_bot[0,0]+Br_bot[0,1])
        U[1*4+2][col] += 1; U[0*4+2][col] += -1
        V[2*6+b_off+0][col] += 1; V[2*6+b_off+1][col] += 1
        col += 1
        # M14: (A2[0,1]-A2[1,1]) * (Br_bot[1,0]+Br_bot[1,1])
        U[0*4+3][col] += 1; U[1*4+3][col] += -1
        V[3*6+b_off+0][col] += 1; V[3*6+b_off+1][col] += 1
        col += 1
    
    # Set W: output combination
    # For block r, base_col = r*14
    # M1..M7: indices 0..6, M8..M14: indices 7..13
    
    for r in range(3):
        base = r * 14
        b_off = 2 * r
        
        # Z00 (i=0, kb=0): M1 + M4 - M5 + M7 + M8 + M11 - M12 + M14
        # W row: (b_off+0)*2 + 0 = 2*b_off
        W[2*b_off+0][base+0] += 1   # M1
        W[2*b_off+0][base+3] += 1   # M4
        W[2*b_off+0][base+4] += -1  # M5
        W[2*b_off+0][base+6] += 1   # M7
        W[2*b_off+0][base+7] += 1   # M8
        W[2*b_off+0][base+10] += 1  # M11
        W[2*b_off+0][base+11] += -1 # M12
        W[2*b_off+0][base+13] += 1  # M14
        
        # Z01 (i=0, kb=1): M3 + M5 + M10 + M12
        # W row: (b_off+1)*2 + 0 = 2*b_off+2
        W[2*b_off+2][base+2] += 1   # M3
        W[2*b_off+2][base+4] += 1   # M5
        W[2*b_off+2][base+9] += 1   # M10
        W[2*b_off+2][base+11] += 1  # M12
        
        # Z10 (i=1, kb=0): M2 + M4 + M9 + M11
        # W row: (b_off+0)*2 + 1 = 2*b_off+1
        W[2*b_off+1][base+1] += 1   # M2
        W[2*b_off+1][base+3] += 1   # M4
        W[2*b_off+1][base+8] += 1   # M9
        W[2*b_off+1][base+10] += 1  # M11
        
        # Z11 (i=1, kb=1): M1 - M2 + M3 + M6 + M8 - M9 + M10 + M13
        # W row: (b_off+1)*2 + 1 = 2*b_off+3
        W[2*b_off+3][base+0] += 1   # M1
        W[2*b_off+3][base+1] += -1  # M2
        W[2*b_off+3][base+2] += 1   # M3
        W[2*b_off+3][base+5] += 1   # M6
        W[2*b_off+3][base+7] += 1   # M8
        W[2*b_off+3][base+8] += -1  # M9
        W[2*b_off+3][base+9] += 1   # M10
        W[2*b_off+3][base+12] += 1  # M13
    
    assert len(U) == 8 and len(U[0]) == 42
    assert len(V) == 24 and len(V[0]) == 42
    assert len(W) == 12 and len(W[0]) == 42
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
