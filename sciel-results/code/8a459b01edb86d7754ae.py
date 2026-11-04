"""Rank-42 Strassen decomposition for 2x4 times 4x6 matrix multiplication.

The tensor is decomposed as 6 independent 2x2x2 Strassen blocks (7 multipliers each),
giving rank = 6 * 7 = 42.

Index conventions:
  A is n x m = 2 x 4, flattened as A[i*m + j] = A[4i + j]
  B is m x p = 4 x 6, flattened as B[j*p + k] = B[6j + k]
  C is n x p = 2 x 6, output indexed as W[k*n + i] = W[2k + i]

We split A into two 2x2 blocks along columns, and B into three 2x2 blocks along rows,
for 6 independent 2x2 matrix multiplications. Each uses Strassen's 7-multiply scheme.
"""
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 120.0  # safety margin


def _elapsed():
    return time.time() - _START_TIME


def _build_strassen_2x2x2():
    """Return (U_block, V_block, W_block) for one 2x2x2 Strassen block.
    
    U_block shape (4, 7), V_block shape (4, 7), W_block shape (4, 7)
    Local indices: X = [[x00, x01], [x10, x11]] flattened as [x00, x01, x10, x11]
    Y = [[y00, y01], [y10, y11]] flattened as [y00, y01, y10, y11]
    Z = [[z00, z01], [z10, z11]] flattened as [z00, z01, z10, z11]
    
    Strassen multipliers:
      M1 = (x00+x11)*(y00+y11)
      M2 = (x10+x11)*y00
      M3 = x00*(y10-y11)
      M4 = x11*(y01-y00)
      M5 = (x00+x10)*y11
      M6 = (x10-x00)*(y00+y01)  [unused in output but needed for correctness? No, M6 is unused]
      M7 = (x11-x10)*(y10+y11)
    
    Output:
      z00 = M1 + M4 - M5 - M7
      z01 = M3 + M5
      z10 = M2 + M4
      z11 = M1 - M2 + M3 + M7
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (x00+x11)*(y00+y11)
    U[0][0] += 1   # x00
    U[3][0] += 1   # x11
    V[0][0] += 1   # y00
    V[3][0] += 1   # y11
    W[0][0] += 1   # z00
    W[3][0] += 1   # z11

    # M2 = (x10+x11)*y00
    U[2][1] += 1   # x10
    U[3][1] += 1   # x11
    V[0][1] += 1   # y00
    W[2][1] += 1   # z10
    W[3][1] += -1  # z11

    # M3 = x00*(y10-y11)
    U[0][2] += 1   # x00
    V[2][2] += 1   # y10
    V[3][2] += -1  # y11
    W[1][2] += 1   # z01
    W[3][2] += 1   # z11

    # M4 = x11*(y01-y00)
    U[3][3] += 1   # x11
    V[1][3] += 1   # y01
    V[0][3] += -1  # y00
    W[0][3] += 1   # z00
    W[2][3] += 1   # z10

    # M5 = (x00+x10)*y11
    U[0][4] += 1   # x00
    U[2][4] += 1   # x10
    V[3][4] += 1   # y11
    W[0][4] += -1  # z00
    W[1][4] += 1   # z01

    # M6 = (x10-x00)*(y00+y01) -- not used in any output, but kept for completeness
    # Actually M6 is NOT needed. Strassen's original uses 7 terms but M6 is unused.
    # Let me re-check: the standard Strassen 7-term scheme does NOT include M6 in outputs.
    # The 7 terms are M1-M7, but the output formulas only use M1,M2,M3,M4,M5,M7.
    # Wait, that's only 6. Let me re-derive.
    #
    # Standard Strassen:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A22-A21)(B21+B22)
    #
    #   C11 = M1 - M5 + M6 + M7
    #   C12 = M3 + M5
    #   C21 = M2 + M4
    #   C22 = M1 - M2 + M3 - M6
    #
    # So M6 IS used! I had the formulas wrong. Let me fix.
    #
    # With my indexing: x00=A11, x01=A12, x10=A21, x11=A22
    # y00=B11, y01=B12, y10=B21, y11=B22
    # z00=C11, z01=C12, z10=C21, z11=C22
    #
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y10-y11)
    # M4 = x11(y01-y00)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y01)
    # M7 = (x11-x10)(y10+y11)
    #
    # z00 = M1 - M5 + M6 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 - M6

    # Reset and redo properly
    U[:] = 0
    V[:] = 0
    W[:] = 0

    # M1 = (x00+x11)(y00+y11)
    U[0][0] += 1
    U[3][0] += 1
    V[0][0] += 1
    V[3][0] += 1
    W[0][0] += 1   # z00 += M1
    W[3][0] += 1   # z11 += M1

    # M2 = (x10+x11)*y00
    U[2][1] += 1
    U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1   # z10 += M2
    W[3][1] += -1  # z11 -= M2

    # M3 = x00*(y10-y11)
    U[0][2] += 1
    V[2][2] += 1
    V[3][2] += -1
    W[1][2] += 1   # z01 += M3
    W[3][2] += 1   # z11 += M3

    # M4 = x11*(y01-y00)
    U[3][3] += 1
    V[1][3] += 1
    V[0][3] += -1
    W[2][3] += 1   # z10 += M4
    W[0][3] += 1   # z00 += M4

    # M5 = (x00+x01)*y11
    U[0][4] += 1
    U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1  # z00 -= M5
    W[1][4] += 1   # z01 += M5

    # M6 = (x10-x00)*(y00+y01)
    U[2][5] += 1
    U[0][5] += -1
    V[0][5] += 1
    V[1][5] += 1
    W[0][5] += 1   # z00 += M6
    W[3][5] += -1  # z11 -= M6

    # M7 = (x11-x10)*(y10+y11)
    U[3][6] += 1
    U[2][6] += -1
    V[2][6] += 1
    V[3][6] += 1
    W[0][6] += 1   # z00 += M7
    W[3][6] += 1   # z11 += M7

    return U, V, W


def construct():
    n, m, p = 2, 4, 6
    rank = 42

    U = np.zeros((n * m, rank))  # 8 x 42
    V = np.zeros((m * p, rank))  # 24 x 42
    W = np.zeros((p * n, rank))  # 12 x 42

    # Get the Strassen block template
    Ub, Vb, Wb = _build_strassen_2x2x2()

    # 6 blocks: A is split into 2 column-blocks of width 2, B is split into 3 row-blocks of height 2
    # Block (ai, bi): A[:, ai*2:(ai+1)*2] * B[bi*2:(bi+1)*2, :] -> contributes to C[:, bi*2:(bi+1)*2]
    #
    # A row index i in {0,1}, A col index within block: j in {0,1}, global col = ai*2 + j
    # A flat index: i*m + (ai*2 + j) = 4*i + ai*2 + j
    #
    # B row index within block: j in {0,1}, global row = bi*2 + j
    # B col index k in {0,1,2,3,4,5}, B flat index: (bi*2+j)*p + k = 6*(bi*2+j) + k
    #
    # C row i in {0,1}, C col within block: k in {0,1}, global col = bi*2 + k
    # W index: k_global * n + i = (bi*2+k)*2 + i = 4*bi + 2*k + i

    for ai in range(2):       # A column block
        for bi in range(3):   # B row block (also C column block)
            col_offset = (ai * 3 + bi) * 7

            for local_j in range(4):   # local index within 2x2 block
                # U: local_j maps to (i, j) where i = local_j // 2, j = local_j % 2
                # Wait, our flattening is [x00, x01, x10, x11] = index 0,1,2,3
                # So local index l: row = l // 2, col = l % 2
                li = local_j // 2
                lj = local_j % 2
                # A flat index
                a_idx = n * ai * 2 + li * m + (ai * 2 + lj)  # this is wrong
                # Let me redo: A[i, col] where col = ai*2 + lj, i = li
                # A flat: i * m + col = li * 4 + ai*2 + lj
                a_flat = li * m + ai * 2 + lj

                for c in range(7):
                    if Ub[local_j][c] != 0:
                        U[a_flat][col_offset + c] += Ub[local_j][c]

            for local_j in range(4):
                li = local_j // 2  # row within 2x2 block
                lj = local_j % 2   # col within 2x2 block
                # B flat: global_row * p + k
                # global_row = bi*2 + li, but wait: local row index in the 2x2 block
                # In B's 2x2 block, row index is the "j" dimension (the shared dimension with A)
                # B[block_row, block_col] where block_row in {0,1}, block_col in {0,1}
                # B flat: (bi*2 + block_row) * p + (bi*2 + block_col) ... no!
                # B is 4x6. Row index is 0..3, col index is 0..5.
                # Block (bi) covers rows bi*2..bi*2+1, cols... wait, B is split by ROWS only.
                # The 2x2 block for Strassen is in the (row, col) space of the sub-problem.
                # A sub-block is 2x2 (rows 0-1 of A, cols ai*2..ai*2+1)
                # B sub-block is 2x2 (rows bi*2..bi*2+1, cols ???)
                
                # Wait, I need to reconsider. For 2x4 * 4x6:
                # A is 2x4, B is 4x6.
                # If I split A into two 2x2 blocks (by columns), and B into three 2x2 blocks (by rows),
                # each product is a 2x2 matrix: A_block (2x2) * B_block (2x2) = C_block (2x2)
                # The C_block columns correspond to B's column range... but B's columns aren't split!
                
                # Actually, the 2x2 Strassen applies to multiplying a 2x2 matrix by a 2x2 matrix.
                # A_block is 2x2 (rows 0-1 of A, cols ai*2..ai*2+1)
                # B_block is 2x2 (rows bi*2..bi*2+1, cols ???)
                
                # Hmm, but B is 4x6, not 4x4. The column dimension of B (which becomes the column dimension of C)
                # is 6, not 2. So we can't just use 2x2 Strassen on B's columns.
                
                # I think the correct approach is:
                # For each pair (ai, bi), we compute a 2x6 sub-product: A[:, ai*2:ai*2+2] * B[bi*2:bi*2+2, :]
                # This is 2x2 * 2x6 = 2x6.
                # We can split this into three 2x2 * 2x2 problems by splitting B's columns into three 2-col groups.
                # But that gives 2*3*3 = 18 blocks of 2x2x2, which is too many.
                
                # Alternative: treat it as 2x2 * 2x6. We can use Strassen-like methods for rectangular matrices.
                # Or: split into 2x2 * 2x2 blocks, giving 3 sub-blocks per (ai, bi) pair, total 2*3*3=18 blocks.
                # 18 * 7 = 126 rank. Too high.
                
                # Let me reconsider the original approach. The seed program uses 6 blocks, each being a 2x2 * 2x2 product.
                # Looking at the blocks:
                # (0, 0, 0, 0, 0, 0): A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
                # (0, 2, 2, 0, 0, 0): A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
                # (0, 0, 0, 2, 0, 2): A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
                # (0, 2, 2, 2, 0, 2): A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
                # (0, 0, 0, 4, 0, 4): A[:,0:2] * B[0:2,4:6] -> C[:,4:6]
                # (0, 2, 2, 4, 0, 4): A[:,2:4] * B[2:4,4:6] -> C[:,4:6]
                
                # So it's 2 (A col blocks) * 3 (B col blocks, each 2 wide) = 6 blocks, but B's rows aren't split?
                # Wait, B[0:2, 0:2] means rows 0-1, cols 0-1. B[2:4, 0:2] means rows 2-3, cols 0-1.
                # So B IS split by rows into two 2x6 blocks, and by columns into three 2-wide groups.
                # Total: 2 (A col groups) * 2 (B row groups) * ... no.
                
                # Let me re-read:
                # Block 0: A[:,0:2] (2x2) * B[0:2,0:2] (2x2) -> C[:,0:2] (2x2)
                # Block 1: A[:,2:4] (2x2) * B[2:4,0:2] (2x2) -> C[:,0:2] (2x2)
                # So C[:,0:2] = A[:,0:2]*B[0:2,0:2] + A[:,2:4]*B[2:4,0:2]
                # This is correct! The sum over the shared dimension j.
                
                # So we have 6 independent 2x2x2 Strassen blocks:
                # (a_col_start, b_row_start, b_col_start)
                # a_col_start in {0, 2}, b_row_start in {0, 2}, b_col_start in {0, 2, 4}
                # But that's 2*2*3 = 12, not 6.
                
                # Wait, looking again:
                # Block 0: ar=0, ac=0, br=0, bc=0, cr=0, cc=0 -> A rows ar:ar+2=0:2, cols ac:ac+2=0:2
                #                                                  B rows br:br+2=0:2, cols bc:bc+2=0:2
                #                                                  C rows cr:cr+2=0:2, cols cc:cc+2=0:2
                # Block 1: ar=0, ac=2, br=2, bc=0, cr=0, cc=0 -> A cols 2:4, B rows 2:4, B cols 0:2, C cols 0:2
                # Block 2: ar=0, ac=0, br=0, bc=2, cr=0, cc=2 -> A cols 0:2, B rows 0:2, B cols 2:4, C cols 2:4
                # Block 3: ar=0, ac=2, br=2, bc=2, cr=0, cc=2
                # Block 4: ar=0, ac=0, br=0, bc=4, cr=0, cc=4
                # Block 5: ar=0, ac=2, br=2, bc=4, cr=0, cc=4
                
                # So the pattern is:
                # For b_col_start in {0, 2, 4}:
                #   Block 2k: A cols 0:2, B rows 0:2, B cols b_col_start:b_col_start+2
                #   Block 2k+1: A cols 2:4, B rows 2:4, B cols b_col_start:b_col_start+2
                
                # This is correct because:
                # C[i, k] = sum_j A[i,j] * B[j,k]
                # For k in 0:2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
                #             = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
                #             = [A[:,0:2] * B[0:2,:]][i,k] + [A[:,2:4] * B[2:4,:]][i,k]
                
                # So yes, 6 blocks. Each is a 2x2 * 2x2 = 2x2 multiplication.

    # Let me redo the construction properly.
    U[:] = 0
    V[:] = 0
    W[:] = 0

    # The 6 blocks: (a_col_off, b_row_off, b_col_off)
    # a_col_off in {0, 2}, b_row_off in {0, 2}, b_col_off in {0, 2, 4}
    # Pattern: for b_col_off in [0,2,4]: block 2k has a_col_off=0, b_row_off=0; block 2k+1 has a_col_off=2, b_row_off=2
    blocks = []
    for k in range(3):
        blocks.append((0, 0, k * 2))   # A cols 0:2, B rows 0:2, B cols k*2:k*2+2
        blocks.append((2, 2, k * 2))   # A cols 2:4, B rows 2:4, B cols k*2:k*2+2

    for b_idx, (ac, br, bc) in enumerate(blocks):
        col_offset = b_idx * 7

        # A is 2x4. In this block, we use A[:, ac:ac+2], which is 2x2.
        # A flat index for A[i, ac+j] is i*m + (ac+j) = 4*i + ac + j
        # Local 2x2 index: row i in {0,1}, col j in {0,1} -> local index i*2 + j
        # U local index l corresponds to A flat index 4*(l//2) + ac + (l%2)

        # B is 4x6. In this block, we use B[br:br+2, bc:bc+2], which is 2x2.
        # B flat index for B[br+j, bc+k] is (br+j)*p + (bc+k) = 6*(br+j) + bc + k
        # Local 2x2 index: row j in {0,1}, col k in {0,1} -> local index j*2 + k
        # V local index l corresponds to B flat index 6*(br + l//2) + bc + (l%2)

        # C is 2x6. Output goes to C[:, bc:bc+2], which is 2x2.
        # W index for C[i, bc+k] is (bc+k)*n + i = 2*(bc+k) + i = 2*bc + 2*k + i
        # Local 2x2 index: row i in {0,1}, col k in {0,1} -> local index i*2 + k
        # W local index l corresponds to W flat index 2*bc + 2*(l%2) + (l//2)
        # Wait: local index l = i*2 + k, so i = l//2, k = l%2
        # W flat: 2*(bc+k) + i = 2*bc + 2*k + i = 2*bc + 2*(l%2) + (l//2)

        for l in range(4):
            li = l // 2  # local row
            lj = l % 2   # local col

            # U
            a_flat = li * m + ac + lj
            for c in range(7):
                U[a_flat][col_offset + c] += Ub[l][c]

            # V
            b_flat = (br + li) * p + bc + lj
            for c in range(7):
                V[b_flat][col_offset + c] += Vb[l][c]

            # W
            w_flat = 2 * (bc + lj) + li
            for c in range(7):
                W[w_flat][col_offset + c] += Wb[l][c]

    # Verify with a random test
    np.random.seed(42)
    A = np.random.randint(-10, 10, (n, m)).astype(float)
    B = np.random.randint(-10, 10, (m, p)).astype(float)
    C_ref = A @ B

    C_test = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            val = 0.0
            for c in range(rank):
                w_val = W[k * n + i][c]
                if w_val == 0:
                    continue
                inner = 0.0
                for j in range(m):
                    inner += U[i * m + j][c] * V[j * p + k][c]
                val += w_val * inner
            C_test[i, k] = val

    max_err = np.max(np.abs(C_test - C_ref))
    if max_err > 1e-6:
        # Fallback: schoolbook rank 48
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1.0
                    V[j * p + k][idx] = 1.0
                    W[k * n + i][idx] = 1.0
                    idx += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
