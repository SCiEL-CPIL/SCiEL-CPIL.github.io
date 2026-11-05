"""Matrix multiplication 4x5x5 - rank reduction via Strassen block decomposition."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n * m, m * p, p * n  # 20, 25, 20
    
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Strategy: Use Strassen decomposition on 2x2 blocks
    # Split 4x5 by 5x5 into blocks of size (2,2) x (2,2) x (2,2) = 8 rank terms each
    # But we can use Strassen for 2x2x2 = 7 terms
    # 
    # For 4x4x4: split into 2x2x2 blocks, each can use Strassen (7 terms)
    # Total for 4x4x4: 8 * 7 = 56 terms (vs 64 schoolbook)
    # For the remaining 1x5 and 5x1 parts, use schoolbook
    #
    # Actually, let's think more carefully.
    # 4x5 matrix * 5x5 matrix
    # Split rows of first matrix: 4 = 2 + 2
    # Split cols of first / rows of second: 5 = 2 + 2 + 1
    # Split cols of second: 5 = 2 + 2 + 1
    #
    # Block structure:
    # A = [A11 A12 A13]  (2x2, 2x2, 2x1)
    #     [A21 A22 A23]  (2x2, 2x2, 2x1)
    # B = [B11 B12 B13]  (2x2, 2x2, 2x1)
    #     [B21 B22 B23]  (2x2, 2x2, 2x1)
    #     [B31 B32 B33]  (1x2, 1x2, 1x1)
    #
    # C = A*B has blocks:
    # C11 = A11*B11 + A12*B21 + A13*B31
    # C12 = A11*B12 + A12*B22 + A13*B32
    # C13 = A11*B13 + A12*B23 + A13*B33
    # C21 = A21*B11 + A22*B21 + A23*B31
    # C22 = A21*B12 + A22*B22 + A23*B32
    # C23 = A21*B13 + A22*B23 + A23*B33
    #
    # For each block product, we need to handle the multiplication.
    # The 2x2 * 2x2 products can use Strassen (7 terms each).
    # The 2x2 * 2x1, 2x1 * 1x2, etc. are smaller.
    
    # Let's use a different approach: direct Strassen-like construction
    # We'll build the tensor decomposition directly.
    
    # For a 2x2x2 matrix multiplication, Strassen uses 7 terms:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # C11 = M1 - M4 + M6 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M5
    
    # Let's build a general framework for block-structured Strassen.
    
    # For 4x4x4 (which is our core), we can tile it as 2x2 blocks of 2x2x2.
    # 4 = 2*2, so we have a 2x2 grid of 2x2x2 blocks.
    # Each 2x2x2 block uses 7 Strassen terms.
    # Total: 4 blocks * 7 = 28 terms for the 4x4x4 core.
    # Plus the edge effects for the 5th row/col.
    
    # Actually, let me reconsider. The problem is 4x5 * 5x5.
    # Let's split as:
    # Rows of A: 4 = 2 + 2
    # Cols of A / Rows of B: 5 = 2 + 2 + 1
    # Cols of B: 5 = 2 + 2 + 1
    
    # This gives a 2x3 grid of block products for the "inner" dimension,
    # and a 2x2 output grid (for the first 4 cols of B) plus 2x1 (for last col).
    
    # This is getting complex. Let me try a simpler approach:
    # Use the known rank for 4x4x4 = 49 (Strassen based)
    # And handle the extra dimensions separately.
    
    # For 4x4x4, rank = 49 (using Strassen recursively: 4x4x4 = 8 * 7 = 56? No...)
    # Actually 4x4x4: split into 2x2 blocks of 2x2x2.
    # 2x2x2 has rank 7 (Strassen).
    # 4x4x4 = 2^3 blocks of 2x2x2, each rank 7, so 8*7 = 56.
    # But there are better decompositions. Known: M(4,4,4) = 49.
    
    # For our problem 4x5x5:
    # We can write it as 4x4x4 + 4x1x5 + 4x5x1 + 1x1x1 terms
    # = 49 + 20 + 20 + 1 = 90? That's not much better.
    
    # Better approach: Use the full 4x5x5 structure with mixed Strassen.
    
    # Let me just implement a direct construction:
    # We'll create the rank-1 terms explicitly.
    
    # Start with schoolbook (rank 100) and try to reduce.
    
    # Actually, let's implement the block-Strassen approach properly.
    
    # Define Strassen for 2x2x2:
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 matrix mult using Strassen, rank 7."""
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # A is 2x2, B is 2x2, C = A*B is 2x2
        # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)*(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1  # A11, A22
        V[0, 0] = 1; V[3, 0] = 1  # B11, B22
        W[0, 0] = 1; W[3, 0] = 1  # C11, C22
        
        # M2 = (A21+A22)*B11
        U[2, 1] = 1; U[3, 1] = 1  # A21, A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1; W[3, 1] = 1  # C21, C22
        
        # M3 = A11*(B12-B22)
        U[0, 2] = 1  # A11
        V[1, 2] = 1; V[3, 2] = -1  # B12, -B22
        W[1, 2] = 1; W[3, 2] = 1  # C12, C22
        
        # M4 = A22*(B21-B11)
        U[3, 3] = 1  # A22
        V[2, 3] = 1; V[0, 3] = -1  # B21, -B11
        W[2, 3] = 1; W[0, 3] = -1  # C21, -C11
        
        # M5 = (A11+A12)*B22
        U[0, 4] = 1; U[1, 4] = 1  # A11, A12
        V[3, 4] = 1  # B22
        W[1, 4] = 1; W[3, 4] = -1  # C12, -C22
        
        # M6 = (A21-A11)*(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1  # A21, -A11
        V[0, 5] = 1; V[1, 5] = 1  # B11, B12
        W[0, 5] = -1; W[2, 5] = 1  # -C11, C21
        
        # M7 = (A12-A22)*(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1  # A12, -A22
        V[2, 6] = 1; V[3, 6] = 1  # B21, B22
        W[1, 6] = -1; W[3, 6] = 1  # -C12, C22
        
        return U, V, W
    
    # Now for 4x4x4: tile as 2x2 grid of 2x2x2 blocks
    # 4x4x4 = sum over block positions (bi, bj, bk) where bi,bj,bk in {0,1}
    # Each block is 2x2x2 with local indices.
    
    # For 4x4x4, the block at position (bi, bj, bk) handles:
    # A[2*bi:2*bi+2, 2*bj:2*bj+2] * B[2*bj:2*bj+2, 2*bk:2*bk+2] -> C[2*bi:2*bi+2, 2*bk:2*bk+2]
    
    def build_4x4x4():
        """Build 4x4x4 decomposition using block Strassen, rank 56."""
        U = np.zeros((16, 56))
        V = np.zeros((16, 56))
        W = np.zeros((16, 56))
        
        col = 0
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    # Get Strassen components for this block
                    u_local, v_local, w_local = strassen_2x2x2()
                    
                    # Map local indices to global
                    # A global index: 2*bi + local_row, 2*bj + local_col -> flat: (2*bi+lr)*4 + (2*bj+lc)
                    # B global index: 2*bj + local_row, 2*bk + local_col -> flat: (2*bj+lr)*4 + (2*bk+lc)  
                    # Wait, B is 4x4, so B[row, col] where row is inner dim, col is outer
                    # For A(4x4) * B(4x4) -> C(4x4):
                    # A[row_a, col_a] * B[col_a, col_b] -> C[row_a, col_b]
                    # So A index: row_a * 4 + col_a
                    # B index: col_a * 4 + col_b
                    # C index: row_a * 4 + col_b
                    
                    for lr in range(2):
                        for lc in range(2):
                            # A local (lr, lc) -> global row = 2*bi+lr, col = 2*bj+lc
                            ga = (2*bi+lr)*4 + (2*bj+lc)
                            for c in range(7):
                                U[ga, col+c] = u_local[lr*2+lc, c]
                    
                    for lr in range(2):
                        for lc in range(2):
                            # B local (lr, lc) -> global row = 2*bj+lr, col = 2*bk+lc
                            gb = (2*bj+lr)*4 + (2*bk+lc)
                            for c in range(7):
                                V[gb, col+c] = v_local[lr*2+lc, c]
                    
                    for lr in range(2):
                        for lc in range(2):
                            # C local (lr, lc) -> global row = 2*bi+lr, col = 2*bk+lc
                            gc = (2*bi+lr)*4 + (2*bk+lc)
                            for c in range(7):
                                W[gc, col+c] = w_local[lr*2+lc, c]
                    
                    col += 7
        
        return U, V, W
    
    # For 4x5x5:
    # A is 4x5, B is 5x5, C is 4x5
    # Split: A rows = 4 (2+2), inner dim = 5 (2+2+1), B cols = 5 (2+2+1)
    
    # Let's build this step by step.
    # The core 4x4x4 part (first 4 inner dims, first 4 output cols) uses rank 56.
    # Then we have additional terms for the 5th inner dim and 5th output col.
    
    # Actually, let me just implement the full 4x5x5 with a systematic approach.
    
    # Split dimensions:
    # A: 4 rows x 5 cols -> split cols as 2+2+1
    # B: 5 rows x 5 cols -> split rows as 2+2+1, cols as 2+2+1
    # C: 4 rows x 5 cols -> split cols as 2+2+1
    
    # Block products needed:
    # For each (bi, bj, bk) where bi in {0,1} (A row block), bj in {0,1,2} (inner block), bk in {0,1,2} (B col block)
    # A block: size (2 if bi<2 else 0) x (2 if bj<2 else 1)
    # Wait, bi only goes 0,1 for 4 rows (2+2)
    # bj goes 0,1,2 for 5 cols (2+2+1)
    # bk goes 0,1,2 for 5 cols (2+2+1)
    
    # A block at (bi, bj): size (2) x (2 if bj<2 else 1)
    # B block at (bj, bk): size (2 if bj<2 else 1) x (2 if bk<2 else 1)
    # C block at (bi, bk): size (2) x (2 if bk<2 else 1)
    
    # For each such block product, we need a decomposition.
    # The sizes vary: 2x2x2, 2x2x1, 2x1x2, 2x1x1
    
    # Let's handle each case:
    
    def decomp_2x2x2():
        """2x2 A * 2x2 B -> 2x2 C, rank 7 (Strassen)"""
        return strassen_2x2x2()
    
    def decomp_2x2x1():
        """2x2 A * 2x1 B -> 2x1 C, rank 4 (schoolbook)"""
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        for c in range(4):
            U[c, c] = 1
            V[c % 2, c] = 1  # B is 2x1, index = row
            W[c // 2, c] = 1  # C is 2x1, index = row
        # Actually, A is 2x2 (4 elements), B is 2x1 (2 elements), C is 2x1 (2 elements)
        # C[i] = sum_j A[i,j] * B[j]
        # For i=0: C[0] = A[0,0]*B[0] + A[0,1]*B[1]
        # For i=1: C[1] = A[1,0]*B[0] + A[1,1]*B[1]
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # Term 0: A[0,0]*B[0] -> C[0]
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # Term 1: A[0,1]*B[1] -> C[0]
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # Term 2: A[1,0]*B[0] -> C[1]
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # Term 3: A[1,1]*B[1] -> C[1]
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def decomp_2x1x2():
        """2x1 A * 1x2 B -> 2x2 C, rank 4 (schoolbook)"""
        # A is 2x1 (2 elements), B is 1x2 (2 elements), C is 2x2 (4 elements)
        # C[i,j] = A[i] * B[j]
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                c = i*2 + j
                U[i, c] = 1
                V[j, c] = 1
                W[c, c] = 1
        return U, V, W
    
    def decomp_2x1x1():
        """2x1 A * 1x1 B -> 2x1 C, rank 2"""
        U = np.zeros((2, 2))
        V = np.zeros((1, 2))
        W = np.zeros((2, 2))
        for i in range(2):
            U[i, i] = 1
            V[0, i] = 1
            W[i, i] = 1
        return U, V, W
    
    # Now assemble 4x5x5:
    # A: 4x5, split as rows 2+2, cols 2+2+1
    # B: 5x5, split as rows 2+2+1, cols 2+2+1
    # C: 4x5, split as rows 2+2, cols 2+2+1
    
    U_list = []
    V_list = []
    W_list = []
    
    a_row_offsets = [0, 2]  # 2 rows each
    inner_offsets = [0, 2, 4]  # sizes 2, 2, 1
    b_col_offsets = [0, 2, 4]  # sizes 2, 2, 1
    a_col_sizes = [2, 2, 1]
    b_row_sizes = [2, 2, 1]
    c_col_sizes = [2, 2, 1]
    
    for bi in range(2):
        a_row_off = a_row_offsets[bi]
        a_row_sz = 2
        for bj in range(3):
            inner_off = inner_offsets[bj]
            inner_sz = a_col_sizes[bj]
            for bk in range(3):
                b_col_off = b_col_offsets[bk]
                b_col_sz = c_col_sizes[bk]
                
                # A block: a_row_sz x inner_sz
                # B block: inner_sz x b_col_sz
                # C block: a_row_sz x b_col_sz
                
                if inner_sz == 2 and b_col_sz == 2:
                    u_l, v_l, w_l = decomp_2x2x2()
                elif inner_sz == 2 and b_col_sz == 1:
                    u_l, v_l, w_l = decomp_2x2x1()
                elif inner_sz == 1 and b_col_sz == 2:
                    u_l, v_l, w_l = decomp_2x1x2()
                else:  # inner_sz == 1 and b_col_sz == 1
                    u_l, v_l, w_l = decomp_2x1x1()
                
                # Map local to global indices
                # A global: row = a_row_off + lr, col = inner_off + lc
                # A flat index: (a_row_off+lr)*5 + (inner_off+lc)
                # B global: row = inner_off + lr, col = b_col_off + lc
                # B flat index: (inner_off+lr)*5 + (b_col_off+lc)
                # C global: row = a_row_off + lr, col = b_col_off + lc
                # C flat index: (a_row_off+lr)*5 + (b_col_off+lc)
                
                U_new = np.zeros((nm, u_l.shape[1]))
                V_new = np.zeros((mp, v_l.shape[1]))
                W_new = np.zeros((pn, w_l.shape[1]))
                
                for lr in range(a_row_sz):
                    for lc in range(inner_sz):
                        ga = (a_row_off+lr)*5 + (inner_off+lc)
                        for c in range(u_l.shape[1]):
                            U_new[ga, c] = u_l[lr*inner_sz+lc, c]
                
                for lr in range(inner_sz):
                    for lc in range(b_col_sz):
                        gb = (inner_off+lr)*5 + (b_col_off+lc)
                        for c in range(v_l.shape[1]):
                            V_new[gb, c] = v_l[lr*b_col_sz+lc, c]
                
                for lr in range(a_row_sz):
                    for lc in range(b_col_sz):
                        gc = (a_row_off+lr)*5 + (b_col_off+lc)
                        for c in range(w_l.shape[1]):
                            W_new[gc, c] = w_l[lr*b_col_sz+lc, c]
                
                U_list.append(U_new)
                V_list.append(V_new)
                W_list.append(W_new)
    
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    
    rank = U.shape[1]
    
    # Verify the decomposition
    if time.time() < deadline - 10:
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            rank = n*m*p
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
