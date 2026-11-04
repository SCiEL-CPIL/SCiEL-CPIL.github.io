import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 36

    # We'll use a Strassen-based approach for the 2x2x2 sub-blocks
    # and schoolbook for the rest.
    # 
    # Strategy: Split the 2x3 matrix into [A1 | A2] where A1 is 2x2 and A2 is 2x1
    # Split the 3x6 matrix into [B1; B2] where B1 is 2x6 and B2 is 1x6
    # Then C = A1*B1 + A2*B2
    # 
    # A1*B1: 2x2 times 2x6. We can split B1 into [B1a | B1b] where each is 2x3
    # A1*B1a: 2x2 * 2x3 -> use Strassen (rank 7) for the 2x2 part, but B1a is 2x3
    # Actually, let's think differently.
    #
    # Better approach: 
    # A is 2x3, B is 3x6.
    # Write A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # Write B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 1x6
    # C = A_left * B_top + A_right * B_bottom
    #
    # A_left * B_top: 2x2 * 2x6. Split B_top into three 2x2 blocks: B_top = [B1, B2, B3]
    # A_left * B1: 2x2 * 2x2 -> Strassen rank 7
    # A_left * B2: 2x2 * 2x2 -> Strassen rank 7  
    # A_left * B3: 2x2 * 2x2 -> Strassen rank 7
    # A_right * B_bottom: 2x1 * 1x6 -> schoolbook rank 6
    # Total: 7 + 7 + 7 + 6 = 27
    #
    # But wait, we can do better. Let me reconsider.
    #
    # Actually, for 2x2 * 2x3, we can think of it as 2x2 * [2x2 | 2x1]
    # The 2x2*2x2 part uses Strassen (7), the 2x2*2x1 part... 
    # 2x2 * 2x1: result is 2x1. Each output C[i] = sum_j A[i,j]*B[j,0]
    # This is just 4 scalar multiplications, rank 4.
    # So 2x2*2x3 = Strassen(2x2*2x2) + schoolbook(2x2*2x1) = 7 + 4 = 11
    #
    # So A_left * B_top (2x2 * 2x6) = split B_top into three 2x2 blocks? No, 2x6 = three 2x2 blocks
    # Each 2x2*2x2 uses Strassen rank 7. Total 21.
    # A_right * B_bottom (2x1 * 1x6) = 6 scalar mults, rank 6.
    # Total: 21 + 6 = 27.
    #
    # Hmm, that's 27. Let me see if I can do better.
    #
    # Alternative: Don't split B_top into 2x2 blocks. Instead, treat 2x2 * 2x6 directly.
    # C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k] for i=0,1 and k=0..5
    # That's 12 terms, rank 12 with schoolbook.
    # With Strassen-style, for 2x2*2x6, we can use 7 multiplications per 2x2 output block
    # but the output is 2x6, so we have 3 such blocks: 7*3 = 21. Same as before.
    #
    # Let me try yet another decomposition.
    #
    # What if I use the fact that rank(2x3x6) might be achievable with a different split?
    # 
    # Known results: rank(2x2x2) = 7, rank(2x2x3) = 10 or 11?
    # Actually rank(2x2x3) is known to be 10.
    # 
    # If rank(2x2x3) = 10, then:
    # A_left * B_top_left (2x2 * 2x3) = 10
    # A_left * B_top_right (2x2 * 2x3) = 10
    # A_right * B_bottom (2x1 * 1x6) = 6
    # Total: 26
    #
    # Or: A * B where we view it as 2x3 * 3x6
    # Split columns of B: B = [B1 | B2] where B1 is 3x3, B2 is 3x3
    # A * B1: 2x3 * 3x3 -> rank(2x3x3)
    # A * B2: 2x3 * 3x3 -> rank(2x3x3)
    # If rank(2x3x3) = 10, total = 20. That would be great!
    #
    # Is rank(2x3x3) = 10? Let me think...
    # 2x3 * 3x3: C[i,k] = sum_j A[i,j] B[j,k], i=0,1; j=0,1,2; k=0,1,2
    # Schoolbook: 18 terms.
    # We can split A into [A1 | A2] where A1 is 2x2, A2 is 2x1
    # B into [B1; B2] where B1 is 2x3, B2 is 1x3
    # C = A1*B1 + A2*B2
    # A1*B1: 2x2*2x3 -> rank 10 (if known)
    # A2*B2: 2x1*1x3 -> rank 3
    # Total: 13
    #
    # Or split differently. Actually, let me just construct a valid decomposition
    # with the Strassen approach for 2x2x2 blocks and see what rank I get.
    #
    # Let me go with the 27-rank construction and verify it.
    # Actually, let me try to be more careful and construct rank 27.
    
    # Construction:
    # A (2x3) = [A0 | A1] where A0 is 2x2 (columns 0,1), A1 is 2x1 (column 2)
    # B (3x6) = [B0; B1] where B0 is 2x6 (rows 0,1), B1 is 1x6 (row 2)
    # C = A0 * B0 + A1 * B1
    #
    # A0 * B0: 2x2 * 2x6. Split B0 into [B0a | B0b | B0c] each 2x2
    # A0*B0a: 2x2*2x2 -> Strassen, 7 terms
    # A0*B0b: 2x2*2x2 -> Strassen, 7 terms
    # A0*B0c: 2x2*2x2 -> Strassen, 7 terms
    # A1*B1: 2x1*1x6 -> schoolbook, 6 terms
    # Total: 7+7+7+6 = 27
    
    rank = 27
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 27
    V = [[0.0] * rank for _ in range(m * p)]  # 18 x 27
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 27
    
    def strassen_2x2x2(U, V, W, col_offset, 
                       u_indices, v_indices, w_indices,
                       a00, a01, a10, a11,
                       b00, b01, b10, b11,
                       c00, c01, c10, c11):
        """
        Add Strassen decomposition for 2x2*2x2 starting at col_offset.
        u_indices: (u00, u01, u10, u11) - indices in U for A entries
        v_indices: (v00, v01, v10, v11) - indices in V for B entries  
        w_indices: (w00, w01, w10, w11) - indices in W for C entries
        a00,a01,a10,a11: row indices in U for A[0,0],A[0,1],A[1,0],A[1,1]
        b00,b01,b10,b11: row indices in V for B[0,0],B[0,1],B[1,0],B[1,1]
        c00,c01,c10,c11: row indices in W for C[0,0],C[0,1],C[1,0],C[1,1]
        """
        # M1 = (a00+a11)(b00+b11)
        U[a00][col_offset] += 1.0
        U[a11][col_offset] += 1.0
        V[b00][col_offset] += 1.0
        V[b11][col_offset] += 1.0
        W[c00][col_offset] += 1.0
        W[c11][col_offset] += 1.0
        W[c01][col_offset] -= 1.0
        W[c10][col_offset] -= 1.0
        
        # M2 = (a10+a11)b00
        U[a10][col_offset+1] += 1.0
        U[a11][col_offset+1] += 1.0
        V[b00][col_offset+1] += 1.0
        W[c10][col_offset+1] += 1.0
        W[c11][col_offset+1] += 1.0
        
        # M3 = a00(b01-b11)
        U[a00][col_offset+2] += 1.0
        V[b01][col_offset+2] += 1.0
        V[b11][col_offset+2] -= 1.0
        W[c00][col_offset+2] += 1.0
        W[c01][col_offset+2] += 1.0
        
        # M4 = a11(b10-b00)
        U[a11][col_offset+3] += 1.0
        V[b10][col_offset+3] += 1.0
        V[b00][col_offset+3] -= 1.0
        W[c10][col_offset+3] += 1.0
        W[c11][col_offset+3] += 1.0
        
        # M5 = (a00+a01)b11
        U[a00][col_offset+4] += 1.0
        U[a01][col_offset+4] += 1.0
        V[b11][col_offset+4] += 1.0
        W[c00][col_offset+4] += 1.0
        W[c01][col_offset+4] += 1.0
        
        # M6 = (a10-a00)(b00+b01)
        U[a10][col_offset+5] += 1.0
        U[a00][col_offset+5] -= 1.0
        V[b00][col_offset+5] += 1.0
        V[b01][col_offset+5] += 1.0
        W[c10][col_offset+5] += 1.0
        W[c11][col_offset+5] += 1.0
        
        # M7 = (a01-a11)(b10+b11)
        U[a01][col_offset+6] += 1.0
        U[a11][col_offset+6] -= 1.0
        V[b10][col_offset+6] += 1.0
        V[b11][col_offset+6] += 1.0
        W[c00][col_offset+6] += 1.0
        W[c01][col_offset+6] += 1.0
        
        return col_offset + 7
    
    # A0 is A[:,0:2], so A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # A1 is A[:,2], so A[0,2]=U[2], A[1,2]=U[5]
    # B0 is B[0:2,:], B1 is B[2,:]
    # B[j,k] index in V: j*p + k = j*6 + k
    # C[i,k] index in W: k*n + i = k*2 + i
    
    col = 0
    
    # A0 * B0a: A0 (2x2) * B[0:2, 0:2] (2x2)
    # B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[6], B[1,1]=V[7]
    # C[0,0]=W[0], C[0,1]=W[2], C[1,0]=W[1], C[1,1]=W[3]
    col = strassen_2x2x2(U, V, W, col,
                         (0,1,3,4), (0,1,6,7), (0,2,1,3),
                         0, 1, 3, 4,
                         0, 1, 6, 7,
                         0, 2, 1, 3)
    
    # A0 * B0b: A0 (2x2) * B[0:2, 2:4] (2x2)
    # B[0,2]=V[2], B[0,3]=V[3], B[1,2]=V[8], B[1,3]=V[9]
    # C[0,2]=W[4], C[0,3]=W[6], C[1,2]=W[5], C[1,3]=W[7]
    col = strassen_2x2x2(U, V, W, col,
                         (0,1,3,4), (2,3,8,9), (4,6,5,7),
                         0, 1, 3, 4,
                         2, 3, 8, 9,
                         4, 6, 5, 7)
    
    # A0 * B0c: A0 (2x2) * B[0:2, 4:6] (2x2)
    # B[0,4]=V[4], B[0,5]=V[5], B[1,4]=V[10], B[1,5]=V[11]
    # C[0,4]=W[8], C[0,5]=W[10], C[1,4]=W[9], C[1,5]=W[11]
    col = strassen_2x2x2(U, V, W, col,
                         (0,1,3,4), (4,5,10,11), (8,10,9,11),
                         0, 1, 3, 4,
                         4, 5, 10, 11,
                         8, 10, 9, 11)
    
    # A1 * B1: A[:,2] (2x1) * B[2,:] (1x6) -> schoolbook, 6 terms
    # A[0,2]=U[2], A[1,2]=U[5]
    # B[2,k]=V[12+k] for k=0..5
    # C[i,k]=W[k*2+i]
    for k in range(6):
        # C[0,k] += A[0,2]*B[2,k]
        U[2][col] += 1.0
        V[12+k][col] += 1.0
        W[k*2+0][col] += 1.0
        col += 1
        # C[1,k] += A[1,2]*B[2,k]
        U[5][col] += 1.0
        V[12+k][col] += 1.0
        W[k*2+1][col] += 1.0
        col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    # Verify
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
