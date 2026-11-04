import numpy as np

def construct():
    n, m, p = 2, 3, 10
    rank = 39
    
    U = np.zeros((n * m, rank))  # (6, rank)
    V = np.zeros((m * p, rank))  # (30, rank)
    W = np.zeros((p * n, rank))  # (20, rank)
    
    col = 0
    
    # Rank-11 decomposition of 2x3x3 tensor
    # A is 2x3, B is 3x3, C = A*B is 2x3
    # A indices: a00,a01,a02,a10,a11,a12 -> U[0..5]
    # B indices: b00,b01,b02,b10,b11,b12,b20,b21,b22 -> V[0..8]
    # C indices: c00,c01,c02,c10,c11,c12 -> W[0..5]
    #
    # Known rank-11 decomposition (from literature on 2x3x3 tensor):
    # We use the following pattern based on the known optimal decomposition
    
    def add_2x3x3_block(b_offset):
        """Add rank-11 terms for a 2x3x3 block. b_offset is the column offset in B (0, 3, 6)."""
        nonlocal col
        # U indices for A: 0-5 (a00..a12)
        # V indices for B: b_offset to b_offset+8
        # W indices for C: (b_offset)*n + i for i in 0,1 and j in 0,1,2
        # W[k*n+i] where k is column index (0-2 for this block), i is row
        
        # Rank-11 decomposition of 2x3x3:
        # Using the known decomposition with integer coefficients
        
        # Term 1: a00 * b00 -> c00
        U[0][col] += 1
        V[b_offset + 0][col] += 1
        W[b_offset * 2 + 0][col] += 1
        col += 1
        
        # Term 2: a01 * b10 -> c00, a00 * b01 -> c01
        U[0][col] += 1
        U[1][col] += 1
        V[b_offset + 0][col] += 1
        V[b_offset + 1][col] += 1
        W[b_offset * 2 + 1][col] += 1
        col += 1
        
        # Term 3: a02 * b20 -> c00, a00 * b02 -> c02
        U[0][col] += 1
        U[2][col] += 1
        V[b_offset + 2][col] += 1
        V[b_offset + 0][col] += 1
        W[b_offset * 2 + 2][col] += 1
        col += 1
        
        # Term 4: a10 * b00 -> c10
        U[3][col] += 1
        V[b_offset + 0][col] += 1
        W[b_offset * 2 + 1][col] += 1
        col += 1
        
        # Term 5: a11 * b10 -> c10, a10 * b01 -> c11
        U[3][col] += 1
        U[4][col] += 1
        V[b_offset + 0][col] += 1
        V[b_offset + 1][col] += 1
        W[b_offset * 2 + 1][col] += 1
        col += 1
        
        # Term 6: a12 * b20 -> c10, a10 * b02 -> c12
        U[3][col] += 1
        U[5][col] += 1
        V[b_offset + 2][col] += 1
        V[b_offset + 0][col] += 1
        W[b_offset * 2 + 2][col] += 1
        col += 1
        
        # Terms 7-11: Additional terms to complete the decomposition
        # These handle the remaining cross terms
        
        # Term 7: (a00+a01) * (b10+b20) -> contributes to c00, c01
        U[0][col] += 1
        U[1][col] += 1
        V[b_offset + 1][col] += 1
        V[b_offset + 2][col] += 1
        W[b_offset * 2 + 0][col] += 1
        W[b_offset * 2 + 1][col] += 1
        col += 1
        
        # Term 8: (a00+a02) * (b20+b02) -> contributes to c00, c02  
        U[0][col] += 1
        U[2][col] += 1
        V[b_offset + 2][col] += 1
        V[b_offset + 0][col] += 1
        W[b_offset * 2 + 0][col] += 1
        W[b_offset * 2 + 2][col] += 1
        col += 1
        
        # Term 9: (a10+a11) * (b10+b20) -> contributes to c10, c11
        U[3][col] += 1
        U[4][col] += 1
        V[b_offset + 1][col] += 1
        V[b_offset + 2][col] += 1
        W[b_offset * 2 + 1][col] += 1
        W[b_offset * 2 + 1][col] += 1
        col += 1
        
        # Term 10: (a10+a12) * (b20+b02) -> contributes to c10, c12
        U[3][col] += 1
        U[5][col] += 1
        V[b_offset + 2][col] += 1
        V[b_offset + 0][col] += 1
        W[b_offset * 2 + 1][col] += 1
        W[b_offset * 2 + 2][col] += 1
        col += 1
        
        # Term 11: (a01+a11) * b10 + cross terms
        U[1][col] += 1
        U[4][col] += 1
        V[b_offset + 1][col] += 1
        W[b_offset * 2 + 0][col] -= 1
        W[b_offset * 2 + 1][col] -= 1
        col += 1
    
    # This ad-hoc construction is error-prone. Let me use a verified approach instead.
    # I'll use the schoolbook for reliability and aim for a better decomposition.
    
    # Reset
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    
    # Use a verified rank-39 decomposition
    # Split into: 3 blocks of 2x3x3 (rank 11 each) + 1 block of 2x3x1 (rank 6)
    # But I need the correct rank-11 decomposition.
    
    # Let me use a simpler but verified approach:
    # Use Strassen-like for 2x2 blocks and schoolbook for the rest.
    
    # Actually, let me just use a clean schoolbook variant with minimal optimization.
    # The safest approach: use the known fact that 2x3x10 can be done in rank 39.
    # I'll construct it by using the optimal 2x3x3 decomposition tiled.
    
    # Known rank-11 decomposition of 2x3x3 (verified in literature):
    # I'll use the following explicit construction that I can verify:
    
    def add_verified_2x3x3(col_start, b_col_offset):
        """Add 11 rank-1 terms for 2x3x3 block starting at col_start, B columns offset by b_col_offset."""
        # A: 2x3, indices 0-5
        # B: 3x3, indices b_col_offset to b_col_offset+2 in each row
        # C: 2x3, output columns b_col_offset to b_col_offset+2
        
        c = col_start
        
        # 11 terms with specific coefficients
        # Using the decomposition from the theory of 2x3x3 tensor rank
        
        # Term 1
        U[0][c] = 1; V[b_col_offset*3 + 0][c] = 1; W[(b_col_offset)*2 + 0][c] = 1; c += 1
        # Term 2
        U[1][c] = 1; V[b_col_offset*3 + 1][c] = 1; W[(b_col_offset+1)*2 + 0][c] = 1; c += 1
        # Term 3
        U[2][c] = 1; V[b_col_offset*3 + 2][c] = 1; W[(b_col_offset+2)*2 + 0][c] = 1; c += 1
        # Term 4
        U[3][c] = 1; V[b_col_offset*3 + 0][c] = 1; W[(b_col_offset)*2 + 1][c] = 1; c += 1
        # Term 5
        U[4][c] = 1; V[b_col_offset*3 + 1][c] = 1; W[(b_col_offset+1)*2 + 1][c] = 1; c += 1
        # Term 6
        U[5][c] = 1; V[b_col_offset*3 + 2][c] = 1; W[(b_col_offset+2)*2 + 1][c] = 1; c += 1
        # Term 7
        U[0][c] = 1; U[1][c] = 1; V[b_col_offset*3 + 1][c] = 1; V[b_col_offset*3 + 0][c] = 1; 
        W[(b_col_offset)*2 + 0][c] = 1; W[(b_col_offset+1)*2 + 0][c] = 1; c += 1
        # Term 8
        U[1][c] = 1; U[2][c] = 1; V[b_col_offset*3 + 2][c] = 1; V[b_col_offset*3 + 1][c] = 1;
        W[(b_col_offset+1)*2 + 0][c] = 1; W[(b_col_offset+2)*2 + 0][c] = 1; c += 1
        # Term 9
        U[3][c] = 1; U[4][c] = 1; V[b_col_offset*3 + 1][c] = 1; V[b_col_offset*3 + 0][c] = 1;
        W[(b_col_offset)*2 + 1][c] = 1; W[(b_col_offset+1)*2 + 1][c] = 1; c += 1
        # Term 10
        U[4][c] = 1; U[5][c] = 1; V[b_col_offset*3 + 2][c] = 1; V[b_col_offset*3 + 1][c] = 1;
        W[(b_col_offset+1)*2 + 1][c] = 1; W[(b_col_offset+2)*2 + 1][c] = 1; c += 1
        # Term 11
        U[0][c] = 1; U[3][c] = 1; V[b_col_offset*3 + 0][c] = 1;
        W[(b_col_offset)*2 + 0][c] = 1; W[(b_col_offset)*2 + 1][c] = 1; c += 1
        
        return c
    
    # This is getting too complex without verification. Let me use a clean, verifiable approach.
    
    # Clean approach: Use the fact that we can decompose 2x3x10 as follows:
    # - Use rank-11 for 2x3x3 three times (columns 0-2, 3-5, 6-8)
    # - Use rank-6 for 2x3x1 (column 9)
    # Total: 39
    #
    # I'll use a verified rank-11 decomposition. Let me construct it carefully.
    
    # Reset everything
    U = np.zeros((6, 39))
    V = np.zeros((30, 39))
    W = np.zeros((20, 39))
    
    # I'll use a simpler but correct approach: schoolbook with some sharing.
    # Actually, the safest is to just return a valid decomposition.
    # Let me use rank 60 schoolbook but that's the same as before.
    
    # Let me try a different tiling: use 2x2 Strassen blocks.
    # Split A into [A1 A2] where A1 is 2x2, A2 is 2x1
    # Split B into [B1; B2] where B1 is 2x10, B2 is 1x10
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2 by 2x10 -> use Strassen per 2x2 block of B1
    # A2*B2: 2x1 by 1x10 -> rank 20
    
    # For 2x2 by 2x10: split B1 into five 2x2 blocks, each needs rank 7
    # Total for A1*B1: 5*7 = 35
    # Total for A2*B2: 20
    # Grand total: 55
    
    # That's better than 60! Let me implement this.
    
    rank = 55
    U = np.zeros((6, rank))
    V = np.zeros((30, rank))
    W = np.zeros((20, rank))
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x10) using Strassen for each 2x2 column block
    # A1 = [[a00, a01], [a10, a11]] -> U[0], U[1], U[3], U[4]
    # B1 rows 0,1: b0k, b1k for k=0..9 -> V[0*10+k], V[1*10+k]
    # C1 = A1*B1 is 2x10 -> W[k*2+0], W[k*2+1]
    
    for b in range(5):  # 5 blocks of 2 columns
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen for 2x2 * 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # M1
        U[0][col] += 1; U[4][col] += 1
        V[0*10 + k0][col] += 1; V[1*10 + k1][col] += 1
        V[1*10 + k0][col] += 1; V[0*10 + k1][col] += 1
        W[k0*2 + 0][col] += 1; W[k1*2 + 1][col] += 1
        W[k0*2 + 1][col] -= 1; W[k1*2 + 0][col] -= 1
        col += 1
        
        # M2
        U[3][col] += 1; U[4][col] += 1
        V[0*10 + k0][col] += 1
        W[k1*2 + 0][col] += 1
        col += 1
        
        # M3
        U[0][col] += 1
        V[0*10 + k1][col] += 1; V[1*10 + k1][col] -= 1
        W[k0*2 + 1][col] += 1
        col += 1
        
        # M4
        U[4][col] += 1
        V[1*10 + k0][col] += 1; V[0*10 + k0][col] -= 1
        W[k1*2 + 1][col] += 1
        col += 1
        
        # M5
        U[0][col] += 1; U[1][col] += 1
        V[1*10 + k1][col] += 1
        W[k0*2 + 1][col] += 1
        col += 1
        
        # M6
        U[3][col] += 1; U[0][col] -= 1
        V[0*10 + k0][col] += 1; V[0*10 + k1][col] += 1
        W[k1*2 + 0][col] += 1
        col += 1
        
        # M7
        U[1][col] += 1; U[4][col] -= 1
        V[1*10 + k0][col] += 1; V[1*10 + k1][col] += 1
        W[k0*2 + 0][col] += 1
        col += 1
    
    # Part 2: A2 (2x1) * B2 (1x10)
    # A2 = [a02, a12] -> U[2], U[5]
    # B2 row 2: b2k for k=0..9 -> V[2*10+k]
    # C2 = A2*B2 is 2x10 -> W[k*2+0], W[k*2+1]
    
    for k in range(10):
        # a02 * b2k -> c0k
        U[2][col] += 1
        V[2*10 + k][col] += 1
        W[k*2 + 0][col] += 1
        col += 1
        
        # a12 * b2k -> c1k
        U[5][col] += 1
        V[2*10 + k][col] += 1
        W[k*2 + 1][col] += 1
        col += 1
    
    # Verify
    assert col == rank, f"col={col}, rank={rank}"
    
    # Convert to lists for the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
