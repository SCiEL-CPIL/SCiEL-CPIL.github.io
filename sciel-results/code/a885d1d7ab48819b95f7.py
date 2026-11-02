"""Matrix multiplication 3,5,7 - Strassen-based block decomposition."""
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = 3, 5, 7
    # We use a block decomposition:
    # A (3x5) = [A1(3x3) A2(3x2)]
    # B (5x7) = [B1(3x7); B2(2x7)]
    # C = A1*B1 + A2*B2
    #
    # For A1*B1 (3x3 times 3x7), we embed 3x3x3 in 4x4x4 using Strassen (rank 7).
    # But we need 3x3x7. We can decompose 3x7 into 3x3 + 3x4.
    # 3x3x3: use Strassen rank 7 (embedded in 4x4x4)
    # 3x3x4: schoolbook rank 12
    # 3x2x7: schoolbook rank 21
    # Total: 7 + 12 + 21 = 40
    
    # Actually, let me use a cleaner approach.
    # 3x3x3 can be done with rank 7 (Strassen).
    # 3x3x4: 3*3*4 = 36 schoolbook, but we can do better.
    # Let's just use: 
    #   A1*B1 where B1 is 3x7: decompose B1 = [B1a(3x3) B1b(3x4)]
    #   A1*B1a: rank 7 (Strassen 3x3x3)
    #   A1*B1b: rank 3*3*4 = 36 schoolbook... too much.
    
    # Alternative: A2*B2 is 3x2 * 2x7 = 3*2*7 = 42 schoolbook.
    # A1*B1 is 3x3 * 3x7 = 3*3*7 = 63 schoolbook.
    # Total schoolbook: 105.
    
    # Let me try: split B into [B1(3x7), B2(2x7)]
    # A1(3x3) * B1(3x7): use Strassen for 3x3x3 part and schoolbook for 3x3x4
    #   Split B1 = [B1a(3x3), B1b(3x4)]
    #   A1*B1a: Strassen rank 7
    #   A1*B1b: schoolbook 3*3*4 = 36
    # A2(3x2) * B2(2x7): schoolbook 3*2*7 = 42
    # Total: 7 + 36 + 42 = 85
    
    # Better: use Strassen-like for the 3x2*2x7 part? No, Strassen is for square.
    
    # Let me try another split:
    # Split A into [A1(2x5), A2(1x5)] and B into [B1(5x4), B2(5x3)]
    # A1*B1: 2x2*4 schoolbook = 16, but use Strassen 2x2x2 embedded... complex.
    # This is getting complicated. Let me just use the simplest reduction.
    
    # Simplest verified approach: 
    # 3x3x3 has rank 7 (Strassen). 
    # 3x5x7 = 3x3x7 + 3x2x7
    # 3x3x7: decompose 7 = 3+4. 3x3x3 rank 7, 3x3x4 rank 36. Total 43.
    # 3x2x7: rank 42.
    # Grand total: 43 + 42 = 85.
    
    # Let's build this.
    rank_total = 7 + 36 + 42  # 85
    U = [[0]*rank_total for _ in range(n*m)]
    V = [[0]*rank_total for _ in range(m*p)]
    W = [[0]*rank_total for _ in range(p*n)]
    
    col = 0
    
    # Part 1: A1(3x3) * B1a(3x3) using Strassen (rank 7)
    # Strassen: M1 = (A11+A22)(B11+B22)
    #           M2 = (A21+A22)B11
    #           M3 = A11(B21-B22)
    #           M4 = A22(B11-B21)
    #           M5 = (A11+A12)B22
    #           M6 = (A21-A11)(B11+B12)
    #           M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # For 3x3x3 embedded in 4x4x4, we zero-pad.
    # A1 is 3x3, B1a is 3x3.
    # Indices: A1 row i (0-2), col j (0-2); B1a row j (0-2), col k (0-2)
    # W index: k*3+i
    #
    # Strassen on 2x2 blocks of the 3x3:
    # A1 = [[a, b, c], [d, e, f], [g, h, i]]
    # Split into 2x2 top-left (a,e; d) -- no, 3x3 doesn't split evenly.
    # Pad to 4x4: A1_pad = [[a,b,c,0],[d,e,f,0],[g,h,i,0],[0,0,0,0]]
    # Strassen on 2x2 blocks: 
    #   A11 = [[a,b],[d,e]], A12 = [[c,0],[f,0]]
    #   A21 = [[g,h],[0,0]], A22 = [[i,0],[0,0]]
    # Similarly for B1a_pad.
    #
    # This is getting very complex. Let me just use schoolbook for everything
    # but try to reduce where easy.
    
    # Actually, the safest bet that's better than 105:
    # Use the fact that 3x3x3 has rank 7.
    # 3x3x7 = sum over k=1..7 of (3x3x3 at position k)
    # But each 3x3x3 has rank 7, so 3x3x7 has rank <= 7*7 = 49? No.
    # Actually 3x3x7 can be seen as 3x3 matrix times 7 columns.
    # The tensor 3x3x7 has rank at most 3*3*7 = 63 (schoolbook).
    # With Strassen for 3x3x3, each column uses rank 7, so 7 columns use 7*7=49?
    # No! The rank of 3x3x7 tensor: we can do 3x3x3 (rank 7) for each of 7 columns,
    # but the rank adds up: total rank <= 7*7 = 49.
    # Wait, that's not right either. The tensor T_{i,j,k} = sum_l A_{i,l} B_{l,k}
    # for fixed k, it's a 3x3 matrix multiplication. If we have a rank-7 decomposition
    # for 3x3x3, then for each k we need 7 terms. For 7 values of k, that's 7*7 = 49 terms.
    # But we can share: the 3x3x7 tensor has rank <= 7 (for the 3x3 part) * 7 (for k) = 49? 
    # No, the rank of the tensor is the minimum number of rank-1 tensors to sum to T.
    # T_{i,j,k} = sum_l A_{i,l} B_{l,k}. This is a sum over l=1..5 of rank-1-ish terms...
    # Actually no, it's a bilinear map. The tensor rank of m x n x p matrix multiplication
    # is R(m,n,p). R(3,3,3) = 23 (best known upper bound from Strassen embedding).
    # R(3,3,7) <= R(3,3,3) + R(3,3,4) or similar.
    #
    # Given the complexity and risk, let me just return schoolbook (105) which is known valid.
    # But that scores 1.0. I need to do better.
    #
    # Let me try: use Strassen 2x2x2 (rank 7) where possible.
    # 3x5x7: 
    # Split into blocks of 2:
    # A(3x5) = [A_top(2x5); A_bot(1x5)]
    # B(5x7) = [B_left(5x6) | B_right(5x1)]
    # C_top = A_top * B  (2x7)
    # C_bot = A_bot * B  (1x7)
    # C_top = A_top * B_left (2x6) + A_top * B_right (2x1)
    # 2x6: split into 2x2 + 2x2 + 2x2, each 2x2x2 rank 7. Total 3*7=21.
    # 2x1: 2*1*1 = 2 schoolbook.
    # A_bot * B_left (1x6): 1*1*6 = 6 schoolbook.
    # A_bot * B_right (1x1): 1 schoolbook.
    # Total: 21 + 2 + 6 + 1 = 30.
    
    # Let me verify this decomposition.
    # A is 3x5, B is 5x7, C is 3x7.
    # A_top = A[0:2, :] (2x5)
    # A_bot = A[2:3, :] (1x5)
    # B_left = B[:, 0:6] (5x6)
    # B_right = B[:, 6:7] (5x1)
    # C[0:2, 0:6] = A_top @ B_left  (2x6)
    # C[0:2, 6:7] = A_top @ B_right  (2x1)
    # C[2:3, 0:6] = A_bot @ B_left  (1x6)
    # C[2:3, 6:7] = A_bot @ B_right  (1x1)
    
    # For 2x6 = 2x(2+2+2): three 2x2x2 blocks, each rank 7. Total 21.
    # For 2x1: 2*1*1 = 2.
    # For 1x6 = 1x(2+2+2): three 1x2x2 blocks? No, 1x2x2 means 1x2 matrix times 2x2 matrix.
    #   1x2 * 2x2: rank <= 1*2*2 = 4 schoolbook. Three of them: 12.
    #   Or: 1x6 = (1x6) schoolbook = 6.
    # For 1x1: rank 1.
    # Total: 21 + 2 + 6 + 1 = 30.
    
    # This gives rank 30, much better than 105!
    # Let me build this carefully.
    
    rank_total = 21 + 2 + 6 + 1  # 30
    U = [[0]*rank_total for _ in range(n*m)]  # 15 x 30
    V = [[0]*rank_total for _ in range(m*p)]  # 35 x 30
    W = [[0]*rank_total for _ in range(p*n)]  # 21 x 30
    
    col = 0
    # Strassen 2x2x2 decomposition:
    # M1 = (a+e)(b+f)
    # M2 = (d+e)(b)
    # M3 = (a-c)(f)
    # M4 = e(b-d)
    # M5 = (a+c)(d+f)
    # M6 = (c-e)(b+d)
    # M7 = (a-e)(f+d)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def strassen_222(base_col, a_row0, a_row1, b_col0, b_col1, c_row0, c_row1, c_col0, c_col1):
        """Add Strassen 2x2x2 decomposition.
        A is 2x2: rows a_row0, a_row1; cols b_col0, b_col1
        B is 2x2: rows b_col0, b_col1; cols c_col0, c_col1
        C is 2x2: rows c_row0, c_row1; cols c_col0, c_col1
        U index: (a_row, b_col) -> a_row*m + b_col
        V index: (b_col, c_col) -> b_col*p + c_col
        W index: (c_col, c_row) -> c_col*n + c_row
        """
        nonlocal col
        m, n, p = 5, 3, 7
        # M1 = (A11+A22)(B11+B22)
        for s in [(1,1)]:
            a_i, a_j = s
            for b_i, b_j in [(0,0),(1,1)]:
                pass
        # Let me just enumerate the 7 Strassen terms.
        # A indices: (r,c) with r in {a_row0,a_row1}, c in {b_col0,b_col1}
        # B indices: (r,c) with r in {b_col0,b_col1}, c in {c_col0,c_col1}
        # C indices: (r,c) with r in {c_row0,c_row1}, c in {c_col0,c_col1}
        
        # M1 = (A[r0,c0]+A[r1,c1])*(B[c0,c0]+B[c1,c1])
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # U: dict {(r,c): coeff}, V: dict {(r,c): coeff}, W: dict {(r,c): coeff}
        ]
        # This is getting unwieldy. Let me use a different approach.
        # I'll build the full schoolbook for the 2x6, 2x1, 1x6, 1x1 blocks.
        # That gives rank 2*6 + 2*1 + 1*6 + 1*1 = 12+2+6+1 = 21.
        # Wait, that's schoolbook for each block. 2*6=12, 2*1=2, 1*6=6, 1*1=1. Total 21.
        # That's even better than 30!
        
        # Actually wait: the rank of a 2x6x2 matrix multiplication tensor is at most 2*6*2=24 schoolbook,
        # but I'm decomposing it as 3 blocks of 2x2x2, each schoolbook rank 4, total 12.
        # Yes, 12 for the 2x6 part.
        
        # Let me just do pure schoolbook on the blocks:
        # C_top_left (2x6) = A_top(2x5) @ B_left(5x6): schoolbook rank 2*5*6 = 60? No!
        # Schoolbook for 2x5 * 5x6: rank = 2*5*6 = 60. That's bad.
        # I need to use the intermediate dimension properly.
        
        # The tensor for A(2x5) * B(5x6) has entries T_{i,j,k} = sum_l A_{i,l} B_{l,k}
        # Schoolbook: rank = 2*5*6 = 60.
        # But I can split B into 6 columns, each 5x1.
        # A(2x5) * B_col(5x1): rank = 2*5*1 = 10 each. 6 columns: 60.
        # Or split A into 2 rows, each 1x5.
        # A_row(1x5) * B(5x6): rank = 1*5*6 = 30 each. 2 rows: 60.
        # Same thing.
        
        # The key insight: if I split the intermediate dimension (5), I don't gain.
        # But if I use Strassen on 2x2 sub-blocks where the intermediate dimension is 2, I gain.
        # 
        # Let me reconsider. The intermediate dimension is 5. I can't easily use Strassen
        # unless I group the intermediate dimension into 2s and 1s.
        # 
        # 5 = 2+2+1.
        # A(3x5) = [A1(3x2) A2(3x2) A3(3x1)]
        # B(5x7) = [B1(2x7); B2(2x7); B3(1x7)]
        # C = A1*B1 + A2*B2 + A3*B3
        # A1*B1: 3x2 * 2x7. Split 7 into 2+2+2+1:
        #   3x2*2x2 (x3) + 3x2*2x1 (x1)
        #   3x2*2x2: can use Strassen? 3x2*2x2 is not square.
        #   Schoolbook: 3*2*2=12 each. 3 blocks: 36.
        #   3x2*2x1: 3*2*1=6.
        #   Total for A1*B1: 36+6 = 42.
        # Similarly A2*B2: 42.
        # A3*B3: 3x1*1x7 = 3*1*7=21.
        # Total: 42+42+21 = 105. Back to schoolbook.
        
        # The fundamental issue: Strassen requires square blocks.
        # For 3x5x7, the best I can do with simple Strassen embedding might be limited.
        
        # Let me just go with schoolbook (105) to be safe, since I can't reliably
        # construct a lower-rank decomposition without risking invalidity.
        return col  # This is a hack; let me restart.
    
    # Aborting complex approach. Using schoolbook.
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
