"""Matrix multiplication 4,4,6 tensor decomposition - rank 77 Strassen family."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 5.0

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Uses recursive Strassen: 4x4*4x6 decomposed via 7 Strassen terms,
    each requiring a 2x2*2x3 multiplication (rank 11).
    Total rank: 7 * 11 = 77.
    """
    n, m, p = 4, 4, 6

    # Build rank-4 schoolbook for 2x2 * 2x1 -> 2x1
    # U shape (4,4), V shape (2,4), W shape (2,4)
    U_sb = np.zeros((4, 4))
    V_sb = np.zeros((2, 4))
    W_sb = np.zeros((2, 4))
    # C_0 = A_00*B_0 + A_01*B_1
    U_sb[0,0]=1; V_sb[0,0]=1; W_sb[0,0]=1
    U_sb[1,1]=1; V_sb[1,1]=1; W_sb[0,1]=1
    # C_1 = A_10*B_0 + A_11*B_1
    U_sb[2,2]=1; V_sb[0,2]=1; W_sb[1,2]=1
    U_sb[3,3]=1; V_sb[1,3]=1; W_sb[1,3]=1

    # Build rank-7 Strassen for 2x2 * 2x2 -> 2x2
    # U shape (4,7), V shape (4,7), W shape (4,7)
    U_st = np.zeros((4, 7))
    V_st = np.zeros((4, 7))
    W_st = np.zeros((4, 7))
    # M1 = (A00+A11)(B10+B11) -> C00+=M1, C01-=M1, C10+=M1, C11+=M1
    U_st[0,0]=1; U_st[3,0]=1
    V_st[1,0]=1; V_st[3,0]=1
    W_st[0,0]=1; W_st[1,0]=-1; W_st[2,0]=1; W_st[3,0]=1
    # M2 = A11*(B00-B10) -> C10+=M2, C11-=M2
    U_st[3,1]=1
    V_st[0,1]=1; V_st[2,1]=-1
    W_st[2,1]=1; W_st[3,1]=-1
    # M3 = (A00-A10)*(B10+B11) -> C00+=M3, C01+=M3
    U_st[0,2]=1; U_st[2,2]=-1
    V_st[1,2]=1; V_st[3,2]=1
    W_st[0,2]=1; W_st[1,2]=1
    # M4 = A00*(B01-B11) -> C00-=M4, C01+=M4
    U_st[0,3]=1
    V_st[1,3]=1; V_st[3,3]=-1
    W_st[0,3]=-1; W_st[1,3]=1
    # M5 = (A10+A11)*(B00+B01) -> C10+=M5, C11+=M5
    U_st[2,4]=1; U_st[3,4]=1
    V_st[0,4]=1; V_st[1,4]=1
    W_st[2,4]=1; W_st[3,4]=1
    # M6 = (A00+A01)*(B11-B01) -> C01+=M6, C11-=M6
    U_st[0,5]=1; U_st[1,5]=1
    V_st[3,5]=1; V_st[1,5]=-1
    W_st[1,5]=1; W_st[3,5]=-1
    # M7 = (A11-A10)*(B00+B10) -> C00+=M7, C10-=M7
    U_st[3,6]=1; U_st[2,6]=-1
    V_st[0,6]=1; V_st[2,6]=1
    W_st[0,6]=1; W_st[2,6]=-1

    # Combine into rank-11 for 2x2 * 2x3 -> 2x3
    # V has 6 rows (2x3), W has 6 rows (2x3)
    # First 4 cols: schoolbook for cols 0,1 (the 2x1 part treating each output col separately)
    # Actually: 2x2 * 2x3. Output is 2x3.
    # Col 0 of B is 2x1, col 1 is 2x1, col 2 is 2x1.
    # For each output column k (k=0,1,2): C_*,k = A * B_*,k (2x2 * 2x1)
    # But we can share: the 2x2*2x1 part for the first two columns can be combined.
    # 
    # Actually let me think differently. 2x2 * 2x3:
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # Rank-11 approach: use schoolbook for first column pair (2x2*2x2 part for cols 0,1) 
    # and 2x2*2x1 for col 2. That's 7+4=11.
    # OR: Strassen for cols 0,1 (rank 7) + schoolbook for col 2 (rank 4) = 11.
    # OR: schoolbook for cols 0,1 (rank 8) + Strassen... no, 2x2*2x1 is rank 4.
    #
    # Let's do: Strassen (rank 7) for the 2x2*2x2 part (cols 0,1 of B) 
    # + schoolbook (rank 4) for 2x2*2x1 (col 2 of B)
    # Total rank = 11.

    U_223 = np.zeros((4, 11))
    V_223 = np.zeros((6, 11))  # 2x3 flattened
    W_223 = np.zeros((6, 11))  # 2x3 flattened

    # Strassen part (cols 0,1 of B -> cols 0,1 of C)
    U_223[:, :7] = U_st
    # V: 2x3, Strassen uses 2x2 part (cols 0,1)
    V_223[0:2, :7] = V_st[0:2, :]   # row 0 of B: elements (0,0),(0,1)
    V_223[2:4, :7] = V_st[2:4, :]   # row 1 of B: elements (1,0),(1,1)
    # W: 2x3, Strassen gives 2x2 output (cols 0,1)
    W_223[0:2, :7] = W_st[0:2, :]   # C row 0, cols 0,1
    W_223[2:4, :7] = W_st[2:4, :]   # C row 1, cols 0,1

    # Schoolbook part (col 2 of B -> col 2 of C)
    U_223[:, 7:] = U_sb
    V_223[0, 7] = V_sb[0,0]  # B(0,2)
    V_223[3, 7] = V_sb[1,1]  # B(1,2)
    V_223[1, 8] = V_sb[0,2]  # B(0,2)
    V_223[4, 8] = V_sb[1,3]  # B(1,2)
    V_223[0, 9] = V_sb[0,1]  # B(0,2) -- wait, let me redo
    # V_sb: V_sb[0,:] corresponds to B row 0, V_sb[1,:] to B row 1
    # For 2x1: B = [x; y], C = [a*x + b*y; c*x + d*y]
    # U_sb[0]=a, U_sb[1]=b, U_sb[2]=c, U_sb[3]=d
    # V_sb[0]=[1,0,0,0], V_sb[1]=[0,1,0,0]
    # W_sb[0]=[1,1,0,0], W_sb[1]=[0,0,1,1]
    # So: term 0: U=1 at (0,0)=a, V=1 at row 0 (B[0]), W=1 at C[0]
    # term 1: U=1 at (1,1)=b, V=1 at row 1 (B[1]), W=1 at C[0]
    # term 2: U=1 at (2,2)=c, V=1 at row 0 (B[0]), W=1 at C[1]
    # term 3: U=1 at (3,3)=d, V=1 at row 1 (B[1]), W=1 at C[1]

    # For col 2 of B (index 2 in the 3 columns):
    # V row for B[0,2] is row 2 (row 0, col 2) -> V_223[2, 7:11]
    # V row for B[1,2] is row 5 (row 1, col 2) -> V_223[5, 7:11]
    V_223[2, 7] = 1   # a*B[0,2]
    V_223[5, 8] = 1   # b*B[1,2]
    V_223[2, 9] = 1   # c*B[0,2]
    V_223[5, 10] = 1  # d*B[1,2]
    # W for C[0,2] is row 2, C[1,2] is row 5
    W_223[2, 7] = 1
    W_223[2, 8] = 1
    W_223[5, 9] = 1
    W_223[5, 10] = 1

    # Now build 4x4 * 4x6 using Strassen structure on 2x2 blocks
    # A is 4x4, split into four 2x2 blocks: A00,A01,A10,A11
    # B is 4x6, split into two 2x6 blocks (top, bottom): B_top, B_bot
    # C is 4x6, split into two 2x6 blocks: C_top, C_bot
    #
    # C_top = A00*B_top + A01*B_bot
    # C_bot = A10*B_top + A11*B_bot
    #
    # Strassen: 7 products of 2x2 * 2x6
    # Each 2x2*2x6: split B into three 2x2 blocks -> 3 * (2x2*2x2) = 3*7 = 21? No.
    # Actually 2x2*2x6: B is 2x6. We can split B into three 2x2 matrices.
    # But we need a single tensor decomposition for 2x2*2x6.
    #
    # Alternatively: 2x2*2x6 = 2x2*(2x3 + 2x3) = two 2x2*2x3 (rank 11 each) = rank 22.
    # Or: 2x2*2x6 directly. Let's use the 2x2*2x3 building block.
    #
    # For 2x2*2x6: B has 6 columns. Split into B_left (2x3) and B_right (2x3).
    # C = A*B_left + A*B_right (two 2x2*2x3 operations, each rank 11).
    # Total rank for 2x2*2x6 = 22.
    #
    # But wait, we can be smarter. The 7 Strassen terms each need a 2x2*2x6 product.
    # If each is rank 22, total = 7*22 = 154. That's worse!
    #
    # Better approach: use the 2x2*2x3 block directly in the Strassen structure.
    # 
    # Actually, the original program's approach:
    # 4x4*4x6: Use Strassen on the 2x2 block structure.
    # B is 4x6. In Strassen, B is split into 2x2 blocks, but B is 4x6 not 4x4.
    # So B is split into two 2x6 blocks (top 2 rows, bottom 2 rows).
    # Each 2x6 block is treated as a "vector of 3 columns of 2x1" or as 2x3 + 2x3.
    #
    # The original program uses build_2x2x3 as the sub-problem.
    # In the Strassen step, each of the 7 terms computes a 2x2 * (2x6) product.
    # This is done by treating 2x6 as: first 3 columns use one 2x2*2x3, last 3 use another.
    # Wait no, 2x6 = 2x3 + 2x3, so two 2x2*2x3 operations, rank 22.
    # 7 * 22 = 154. That's not 77.
    #
    # Let me re-examine the original program more carefully.
    # The original uses build_2x2x3 (rank 11) and in build_4x4x6:
    # rank_sub = 11, total_rank = 7 * 11 = 77.
    #
    # How does this work? The Strassen structure for 4x4*4x6:
    # A = [[A00, A01],[A10, A11]] (each 2x2)
    # B = [[B_top],[B_bot]] where B_top, B_bot are 2x6
    # C = [[C_top],[C_bot]] where C_top, C_bot are 2x6
    #
    # C_top = A00*B_top + A01*B_bot
    # C_bot = A10*B_top + A11*B_bot
    #
    # Strassen: 7 scalar products of "2x2 matrices" where the "matrices" 
    # for B and C are actually 2x6 (i.e., 2x6 arrays).
    # Each product is a 2x2 * 2x6 -> 2x6 operation.
    # 
    # But the sub-problem used is 2x2*2x3 (rank 11), not 2x2*2x6 (rank 22).
    # How? By splitting the 6 columns into 3+3 and using TWO 2x2*2x3 operations?
    # That would give 7 * 2 * 11 = 154.
    #
    # OR: the 2x6 is handled by noting that in the Strassen terms,
    # the B-blocks and C-blocks are 2x6, but we can use a rank-11 decomposition
    # for 2x2*2x6? Let me check if R(2,2,6) = 11 is possible.
    #
    # R(2,2,3) = 11. R(2,2,6) >= R(2,2,3) * 2 = 22 by tensor product?
    # No, that's not how it works. R(2,2,6) >= 2*R(2,2,3) only if we can't share.
    # Actually R(2,2,6) <= 2*R(2,2,3) = 22 by doing two 2x2*2x3.
    # And R(2,2,6) >= ? The trivial bound is 2*2*6=24.
    # Lower bound: R(2,2,6) >= max(4, 4, 6, ...) hmm.
    #
    # Actually, I think the original program might have a different structure.
    # Let me re-read it.
    #
    # In build_4x4x6, the terms have 'A', 'B', 'C' entries like [(0,1), (3,1)].
    # The u_offset, v_offset, w_offset map local block indices to global positions.
    # 
    # u_offset: {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)} -> 2x2 block positions in 4x4
    # v_offset: {0: (0,0), 1: (0,3), 2: (2,0), 3: (2,3)} -> 2x3 block positions in 4x6
    # w_offset: {0: (0,0), 1: (3,0), 2: (0,2), 3: (3,2)} -> 2x3 block positions in 6x4 (W is p*n = 6x4)
    #
    # So B is treated as four 2x3 blocks: B[0:2, 0:3], B[0:2, 3:6], B[2:4, 0:3], B[2:4, 3:6]
    # C is treated as four 2x3 blocks: C[0:2, 0:3], C[0:2, 3:6] (wait, C is 4x6)
    # W is p*n = 6*4 = 24, arranged as 6x4. w_offset maps to 2x3 blocks in this 6x4.
    #
    # So the structure is:
    # A = 4x4, split into four 2x2 blocks
    # B = 4x6, split into four 2x3 blocks (2 rows x 3 cols)
    # C = 4x6, split into four 2x3 blocks
    #
    # Strassen on the 2x2 block structure:
    # A blocks: A00(top-left), A01(top-right), A10(bot-left), A11(bot-right)
    # B blocks: B00, B01 (top row of blocks), B10, B11 (bottom row)
    # C blocks: C00, C01 (top row), C10, C11 (bottom row)
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Strassen: 7 products of 2x2 * 2x3 -> 2x3
    # Each is rank 11. Total: 7*11 = 77.
    #
    # This makes sense! The B and C blocks are 2x3, matching the 2x2*2x3 sub-problem.

    # Now let me implement this cleanly.
    # Strassen terms for 2x2 block multiplication where B and C blocks are 2x3:
    # A blocks: indexed 0,1,2,3 (top-left, top-right, bot-left, bot-right)
    # B blocks: indexed 0,1,2,3 (top-left, top-right, bot-left, bot-right) [2x3 each]
    # C blocks: indexed 0,1,2,3 (top-left, top-right, bot-left, bot-right) [2x3 each]

    # The 7 Strassen terms (same as 2x2*2x2 but with 2x3 blocks):
    terms = [
        # M1 = (A0+A2)(B1+B3) -> C0+=M1, C1-=M1, C2+=M1, C3+=M1
        {'A': [(0,1), (2,1)], 'B': [(1,1), (3,1)], 'C': [(0,1), (1,-1), (2,1), (3,1)]},
        # M2 = A2*(B0-B2) -> C2+=M2, C3-=M2
        {'A': [(2,1)], 'B': [(0,1), (2,-1)], 'C': [(2,1), (3,-1)]},
        # M3 = (A0-A2)*(B1+B3) -> C0+=M3, C1+=M3
        {'A': [(0,1), (2,-1)], 'B': [(1,1), (3,1)], 'C': [(0,1), (1,1)]},
        # M4 = A0*(B1-B3) -> C0-=M4, C1+=M4
        {'A': [(0,1)], 'B': [(1,1), (3,-1)], 'C': [(0,-1), (1,1)]},
        # M5 = (A2+A3)*(B0+B1) -> C2+=M5, C3+=M5
        {'A': [(2,1), (3,1)], 'B': [(0,1), (1,1)], 'C': [(2,1), (3,1)]},
        # M6 = (A0+A1)*(B3-B1) -> C1+=M6, C3-=M6
        {'A': [(0,1), (1,1)], 'B': [(3,1), (1,-1)], 'C': [(1,1), (3,-1)]},
        # M7 = (A3-A2)*(B0+B2) -> C0+=M7, C2-=M7
        {'A': [(3,1), (2,-1)], 'B': [(0,1), (2,1)], 'C': [(0,1), (2,-1)]},
    ]

    # Block offsets: for 4x4 A: block (bi,bj) starts at row 2*bi, col 2*bj
    # For 4x6 B: block (bi,bj) starts at row 2*bi, col 3*bj (2 rows, 3 cols)
    # For 6x4 W (p*n): block (bi,bj) starts at row 3*bi, col 2*bj (3 rows, 2 cols)
    # Wait, W has shape (p*n, rank) = (24, rank). It's indexed as W[k*n+i, t]
    # So W is a 6x4 matrix (rows = k=0..5, cols = i=0..3) flattened row-major.
    # Block (bi,bj) in W: rows 3*bi..3*bi+2, cols 2*bj..2*bj+1
    # Flat index: (3*bi + rk)*4 + (2*bj + cc) for local row rk, col cc

    a_block_row = [0, 2, 0, 2]  # block index -> row offset in A (0=top, 1=bot)
    a_block_col = [0, 0, 1, 1]  # block index -> col offset in A (0=left, 1=right)
    b_block_row = [0, 0, 1, 1]  # block index -> row offset in B
    b_block_col = [0, 1, 0, 1]  # block index -> col offset in B (each block is 2x3)
    c_block_row = [0, 0, 1, 1]  # block index -> row offset in C
    c_block_col = [0, 1, 0, 1]  # block index -> col offset in C

    # In the 2x2*2x3 sub-decomposition:
    # U_sub has shape (4, 11): indices (i,j) for i,j in 0..1 -> flat i*2+j
    # V_sub has shape (6, 11): indices (j,k) for j in 0..1, k in 0..2 -> flat j*3+k
    # W_sub has shape (6, 11): indices (k,i) for k in 0..1, i in 0..2 -> flat k*2+i
    #   Wait, W should be (p*n, rank) for the sub-problem. p=3, n=2, so 6x4? No.
    #   For 2x2*2x3: A is 2x2, B is 2x3, C is 2x3.
    #   W has shape (p*n, rank) = (3*2, rank) = (6, rank).
    #   W[k*n+i, t] where k in 0..2 (output col), i in 0..1 (output row).
    #   Flat index: k*2+i.

    # For the full 4x4*4x6:
    # U has shape (n*m, rank) = (16, 77): U[i*m+j, t] = U[i*4+j, t]
    # V has shape (m*p, rank) = (24, 77): V[j*p+k, t] = V[j*6+k, t]
    # W has shape (p*n, rank) = (24, 77): W[k*n+i, t] = W[k*4+i, t]

    rank_sub = 11
    total_rank = 7 * rank_sub

    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((24, total_rank))
    W_full = np.zeros((24, total_rank))

    for t_idx, term in enumerate(terms):
        base = t_idx * rank_sub
        # For each column c of the sub-decomposition
        for c in range(rank_sub):
            # U_sub values (4 values for the 2x2 A block)
            u_vals = U_223[:, c]  # shape (4,)
            # V_sub values (6 values for the 2x3 B block)
            v_vals = V_223[:, c]  # shape (6,)
            # W_sub values (6 values for the 2x3 C block, indexed as k*2+i)
            w_vals = W_223[:, c]  # shape (6,)

            # Distribute U to appropriate A blocks
            for blk, coeff in term['A']:
                for li in range(2):  # local row
                    for lj in range(2):  # local col
                        local_u = li * 2 + lj
                        val = u_vals[local_u] * coeff
                        if val == 0:
                            continue
                        # Global position in A: row = a_block_row[blk]*2 + li, col = a_block_col[blk]*2 + lj
                        g_row = a_block_row[blk] * 2 + li
                        g_col = a_block_col[blk] * 2 + lj
                        U_full[g_row * 4 + g_col, base + c] += val

            # Distribute V to appropriate B blocks
            for blk, coeff in term['B']:
                for lj in range(2):  # local row (of B block)
                    for lk in range(3):  # local col (of B block)
                        local_v = lj * 3 + lk
                        val = v_vals[local_v] * coeff
                        if val == 0:
                            continue
                        g_row = b_block_row[blk] * 2 + lj
                        g_col = b_block_col[blk] * 3 + lk
                        V_full[g_row * 6 + g_col, base + c] += val

            # Distribute W to appropriate C blocks
            for blk, coeff in term['C']:
                for lk in range(2):  # local row of C block (= output col local)
                    for li in range(3):  # local col of C block (= output row local)
                        # W_sub is indexed as k*2+i where k=0..1 (col), i=0..2 (row)
                        # Wait, I need to be careful.
                        # In the sub-problem 2x2*2x3:
                        # C is 2x3. C[i,k] = sum_j A[i,j]*B[j,k]
                        # W has shape (p*n, rank) = (3*2, rank). W[k*n+i, t] = W[k*2+i, t]
                        # So local index in W: k*2+i where k in 0..2 (col), i in 0..1 (row)
                        local_w = lk * 2 + li  # k=lk (0,1), i=li (0,1,2)? No!
                        # k in 0..2 (3 values), i in 0..1 (2 values)
                        # local_w = k*2+i = lk*2+li where lk in 0..2, li in 0..1
                        # I had the loop wrong. Let me fix:
                        pass
            # Redo W distribution properly:
            for blk, coeff in term['C']:
                for lk in range(3):  # local col of C block (k in 0..2)
                    for li in range(2):  # local row of C block (i in 0..1)
                        local_w = lk * 2 + li
                        val = w_vals[local_w] * coeff
                        if val == 0:
                            continue
                        g_row = c_block_row[blk] * 2 + li  # row in C
                        g_col = c_block_col[blk] * 3 + lk  # col in C
                        # W flat index: k*n+i = g_col*4 + g_row
                        W_full[g_col * 4 + g_row, base + c] += val

    # Convert to lists for the interface
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()

    # Verify correctness
    def verify(U_l, V_l, W_l, n, m, p):
        Ua = np.array(U_l, dtype=float)
        Va = np.array(V_l, dtype=float)
        Wa = np.array(W_l, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j, :] * Va[j*p+k, :], Wa[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_list, V_list, W_list, n, m, p):
        # Fallback: schoolbook rank 96
        rank = n * m * p
        U_f = [[0]*rank for _ in range(n*m)]
        V_f = [[0]*rank for _ in range(m*p)]
        W_f = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i*m+j][col] = 1
                    V_f[j*p+k][col] = 1
                    W_f[k*n+i][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
