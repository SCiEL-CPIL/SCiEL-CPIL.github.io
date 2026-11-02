"""Matrix multiplication 4,4,7 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# Strassen 2x2x2 decomposition (7 terms)
# For multiplying two 2x2 matrices:
# M = A * B where A, B are 2x2
# C[0,0] = M1 - M2 + M4 + M5
# C[0,1] = M3 + M5
# C[1,0] = M2 + M4
# C[1,1] = M1 - M3 + M4 + M6
# where:
# M1 = (A00+A11)*(B00+B11)
# M2 = (A10+A11)*B00
# M3 = A00*(B01-B11)
# M4 = A11*(B10-B00)
# M5 = (A00+A01)*B11
# M6 = (A10-A00)*(B00+B01)
# M7 = (A01-A11)*(B10+B11)

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2*2x2 with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = -1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[0][1] = -1; W[2][1] = 1
    
    # M3 = A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    # M4 = A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    return U, V, W


def _build_block(n1, m1, p1, U1, V1, W1, offset, U, V, W):
    """Place a block decomposition into the full matrices.
    
    n1, m1, p1: dimensions of the block
    U1, V1, W1: factor matrices for the block
    offset: column offset
    U, V, W: full factor matrices to fill
    """
    rank1 = U1.shape[1]
    U[offset:offset+rank1, :] = 0  # will be overwritten
    # Actually we need to place at specific positions
    # U1 has shape (n1*m1, rank1), placed at rows 0..n1*m1-1 of U's block
    # But in the full problem, the block occupies specific row ranges
    
    # This function is too complex for arbitrary placements. Let's do it differently.
    pass


def _build_4x4x7():
    """Build a decomposition for 4x4 * 4x7 using block structure.
    
    Split the 4x7 output into blocks. We can write:
    C (4x7) = A (4x4) * B (4x7)
    
    Split B into B1 (4x4) and B2 (4x3):
    C = A*B1 + A*B2
    
    For A*B1: use Strassen-based 4x4*4x4 decomposition
    For A*B2: use a decomposition for 4x4*4x3
    
    Total rank = rank(4x4*4x4) + rank(4x4*4x3)
    
    For 4x4*4x4: We can use two 2x2 blocks.
    A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    C = [[A11*B11+A12*B21, A11*B12+A12*B22],
         [A21*B11+A22*B21, A21*B12+A22*B22]]
    
    Each 2x2*2x2 product uses rank 7. There are 4 such products.
    Total for 4x4*4x4: 4*7 = 28.
    
    For 4x4*4x3:
    A = [[A11, A12], [A21, A22]], B2 = [[B21_top, B22_top], [B21_bot, B22_bot]]
    where B21_top, B22_top are 2x3, etc.
    
    C2 = A*B2 = [[A11*B21_top + A12*B21_bot, A11*B22_top + A12*B22_bot],
                  [A21*B21_top + A22*B21_bot, A21*B22_top + A22*B22_bot]]
    
    Each term like A11*B21_top is 2x2 * 2x3.
    
    For 2x2 * 2x3: we can think of it as two 2x2*2x1 products.
    2x2*2x1: A (2x2) * b (2x1) = [a00*b0+a01*b1, a10*b0+a11*b1]
    This has rank 4 (schoolbook) or can we do better?
    
    Actually, let's just use schoolbook for the 2x2*2x3 part.
    2x2 * 2x3: 2*2*3 = 12 terms schoolbook.
    
    There are 4 such 2x2*2x3 products in the 4x4*4x3.
    So 4*12 = 48.
    
    Total: 28 + 48 = 76.
    
    But schoolbook is 4*4*7 = 112. So 76 < 112.
    """
    n, m, p = 4, 4, 7
    rank = 28 + 48  # 76
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: 4x4 * 4x4 (first 4 columns of B)
    # Use 4 Strassen 2x2x2 blocks
    # A indices: A11 rows 0-1, A12 rows 0-1 cols 2-3, A21 rows 2-3 cols 0-1, A22 rows 2-3 cols 2-3
    # In row-major: A[i,j] at index i*4+j
    # A11: i in {0,1}, j in {0,1} -> indices 0,1,4,5
    # A12: i in {0,1}, j in {2,3} -> indices 2,3,6,7
    # A21: i in {2,3}, j in {0,1} -> indices 8,9,12,13
    # A22: i in {2,3}, j in {2,3} -> indices 10,11,14,15
    
    # B1 (4x4): B11 cols 0-1, B12 cols 2-3
    # B11: j in {0,1}, k in {0,1} -> indices 0,1,4,5 (in B's 4x4 part, row j, col k -> j*4+k)
    # Wait, B is 4x7. B1 is the first 4 columns. B1[j,k] for j in 0..3, k in 0..3.
    # B11: j in {0,1}, k in {0,1} -> indices 0,1,4,5
    # B12: j in {0,1}, k in {2,3} -> indices 2,3,6,7
    # B21: j in {2,3}, k in {0,1} -> indices 8,9,12,13
    # B22: j in {2,3}, k in {2,3} -> indices 10,11,14,15
    
    # C1 (4x4 output, first 4 cols):
    # C11: i in {0,1}, k in {0,1} -> indices 0,1,4,5 (in C's 4x7, row i, col k -> i*7+k)
    # C12: i in {0,1}, k in {2,3} -> indices 2,3,6,7
    # C21: i in {2,3}, k in {0,1} -> indices 14,15,18,19
    # C22: i in {2,3}, k in {2,3} -> indices 16,17,20,21
    
    # Strassen 2x2x2: U1 (4,7), V1 (4,7), W1 (4,7)
    # U1 row i2*2+j2, V1 row j2*2+k2, W1 row k2*2+i2
    # where i2, j2, k2 in {0,1}
    
    U1, V1, W1 = _strassen_2x2x2()
    
    # Block (1,1): A11 * B11 -> C11
    # A11 rows: 0,1,4,5 -> local 0,1,2,3
    # B11 rows: 0,1,4,5 -> local 0,1,2,3
    # C11 rows: 0,1,4,5 -> local 0,1,2,3
    col_off = 0
    for c in range(7):
        U[0, col_off+c] = U1[0, c]
        U[1, col_off+c] = U1[1, c]
        U[4, col_off+c] = U1[2, c]
        U[5, col_off+c] = U1[3, c]
        V[0, col_off+c] = V1[0, c]
        V[1, col_off+c] = V1[1, c]
        V[4, col_off+c] = V1[2, c]
        V[5, col_off+c] = V1[3, c]
        W[0, col_off+c] = W1[0, c]
        W[1, col_off+c] = W1[1, c]
        W[4, col_off+c] = W1[2, c]
        W[5, col_off+c] = W1[3, c]
    col_off += 7
    
    # Block (1,2): A11 * B12 -> C12
    # A11 rows: 0,1,4,5
    # B12 rows: 2,3,6,7
    # C12 rows: 2,3,6,7
    for c in range(7):
        U[0, col_off+c] = U1[0, c]
        U[1, col_off+c] = U1[1, c]
        U[4, col_off+c] = U1[2, c]
        U[5, col_off+c] = U1[3, c]
        V[2, col_off+c] = V1[0, c]
        V[3, col_off+c] = V1[1, c]
        V[6, col_off+c] = V1[2, c]
        V[7, col_off+c] = V1[3, c]
        W[2, col_off+c] = W1[0, c]
        W[3, col_off+c] = W1[1, c]
        W[6, col_off+c] = W1[2, c]
        W[7, col_off+c] = W1[3, c]
    col_off += 7
    
    # Block (2,1): A21 * B11 -> C21
    # A21 rows: 8,9,12,13
    # B11 rows: 0,1,4,5
    # C21 rows: 14,15,18,19
    for c in range(7):
        U[8, col_off+c] = U1[0, c]
        U[9, col_off+c] = U1[1, c]
        U[12, col_off+c] = U1[2, c]
        U[13, col_off+c] = U1[3, c]
        V[0, col_off+c] = V1[0, c]
        V[1, col_off+c] = V1[1, c]
        V[4, col_off+c] = V1[2, c]
        V[5, col_off+c] = V1[3, c]
        W[14, col_off+c] = W1[0, c]
        W[15, col_off+c] = W1[1, c]
        W[18, col_off+c] = W1[2, c]
        W[19, col_off+c] = W1[3, c]
    col_off += 7
    
    # Block (2,2): A22 * B22 -> C22
    # A22 rows: 10,11,14,15
    # B22 rows: 10,11,14,15
    # C22 rows: 16,17,20,21
    for c in range(7):
        U[10, col_off+c] = U1[0, c]
        U[11, col_off+c] = U1[1, c]
        U[14, col_off+c] = U1[2, c]
        U[15, col_off+c] = U1[3, c]
        V[10, col_off+c] = V1[0, c]
        V[11, col_off+c] = V1[1, c]
        V[14, col_off+c] = V1[2, c]
        V[15, col_off+c] = V1[3, c]
        W[16, col_off+c] = W1[0, c]
        W[17, col_off+c] = W1[1, c]
        W[20, col_off+c] = W1[2, c]
        W[21, col_off+c] = W1[3, c]
    col_off += 7
    
    # Part 2: 4x4 * 4x3 (last 3 columns of B)
    # B2 = B[:, 4:7], shape 4x3
    # Split into 2x2 blocks of A and 2x3 blocks of B2
    # A11 (2x2) * B2_11 (2x3) -> C2_11 (2x3)
    # A12 (2x2) * B2_12 (2x3) -> C2_12 (2x3)
    # A21 (2x2) * B2_21 (2x3) -> C2_21 (2x3)
    # A22 (2x2) * B2_22 (2x3) -> C2_22 (2x3)
    
    # B2 indices in B (4x7): row j, col k (k in 4,5,6) -> j*7+k
    # B2_11: j in {0,1}, k in {4,5,6} -> indices 4,5,6, 11,12,13
    # B2_12: j in {0,1}, k in {4,5,6} -> same as B2_11? No.
    # Wait, B2 is 4x3. The rows of B2 are j=0..3, cols are k=4,5,6.
    # B2_11: j in {0,1}, k in {4,5,6} -> indices 0*7+4=4, 0*7+5=5, 0*7+6=6, 1*7+4=11, 1*7+5=12, 1*7+6=13
    # B2_12: j in {0,1}, k in {4,5,6} -> same indices! That's wrong.
    
    # I think I'm confusing myself. Let me restart the indexing.
    # B is 4x7. B[j,k] is at index j*7+k in the V matrix (m*p = 4*7 = 28 rows).
    # B2 = B[:, 4:7] has entries B[j, 4+k'] for j in 0..3, k' in 0..2.
    # So B2[j, k'] is at V index j*7 + 4 + k'.
    
    # B2_11: j in {0,1}, k' in {0,1,2} -> V indices: 0*7+4=4, 0*7+5=5, 0*7+6=6, 1*7+4=11, 1*7+5=12, 1*7+6=13
    # B2_21: j in {2,3}, k' in {0,1,2} -> V indices: 2*7+4=18, 2*7+5=19, 2*7+6=20, 3*7+4=25, 3*7+5=26, 3*7+6=27
    
    # For 2x2 * 2x3 schoolbook:
    # A (2x2) rows: a00, a01, a10, a11 (local 0,1,2,3)
    # B (2x3) rows: b00, b01, b02, b10, b11, b12 (local 0,1,2,3,4,5)
    # C (2x3) rows: c00, c01, c02, c10, c11, c12 (local 0,1,2,3,4,5)
    # 
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c02 = a00*b02 + a01*b12
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    # c12 = a10*b02 + a11*b12
    
    # Rank 12 schoolbook.
    
    def _add_2x2_2x3_block(U_rows, V_rows, W_rows, col_offset):
        """Add a 2x2 * 2x3 schoolbook block."""
        # U_rows: list of 4 row indices in U
        # V_rows: list of 6 row indices in V
        # W_rows: list of 6 row indices in W
        terms = [
            # (U_idx, V_idx, W_idx)
            (0, 0, 0), (2, 3, 0),  # c00 = a00*b00 + a10*b10
            (0, 1, 1), (2, 4, 1),  # c01 = a00*b01 + a10*b11
            (0, 2, 2), (2, 5, 2),  # c02 = a00*b02 + a10*b12
            (1, 0, 3), (3, 3, 3),  # c10 = a01*b00 + a11*b10
            (1, 1, 4), (3, 4, 4),  # c11 = a01*b01 + a11*b11
            (1, 2, 5), (3, 5, 5),  # c12 = a01*b02 + a11*b12
        ]
        for t, (ui, vi, wi) in enumerate(terms):
            U[U_rows[ui], col_offset + t] = 1
            V[V_rows[vi], col_offset + t] = 1
            W[W_rows[wi], col_offset + t] = 1
        return 12
    
    # C2 indices in W (p*n = 7*4 = 28 rows): row k, col i -> k*4+i
    # C2_11: i in {0,1}, k in {4,5,6} -> W indices: 4*4+0=16, 4*4+1=17, 5*4+0=20, 5*4+1=21, 6*4+0=24, 6*4+1=25
    # C2_12: i in {0,1}, k in {4,5,6} -> same as C2_11? No, C2_12 is for A12*B2_12.
    # Wait, I need to be more careful.
    
    # Let me redefine. The output C is 4x7. C[i,k] is at W index k*4+i.
    # C[:, 4:7] is the part from A*B2.
    # C2[i, k'] = C[i, 4+k'] for i in 0..3, k' in 0..2.
    # W index for C2[i, k'] = (4+k')*4 + i = 16 + 4*k' + i.
    
    # C2_11: i in {0,1}, k' in {0,1,2} -> W indices: 16+0+0=16, 16+0+1=17, 16+4+0=20, 16+4+1=21, 16+8+0=24, 16+8+1=25
    # C2_12: i in {0,1}, k' in {0,1,2} -> same W indices! That's a problem.
    
    # Oh wait, C2_11 and C2_12 are different parts of the output.
    # C2[i, k'] = (A*B2)[i, k'] = sum_j A[i,j] * B2[j, k']
    # = sum_j A[i,j] * B[4+k', j]  ... no, B2[j, k'] = B[j, 4+k']
    
    # So C2[i, k'] = sum_{j=0}^{3} A[i,j] * B[j, 4+k']
    
    # This is just the last 3 columns of C. There's no further splitting needed.
    # The block structure I was trying to use doesn't apply cleanly here.
    
    # Let me just do schoolbook for the 4x4 * 4x3 part directly.
    # 4*4*3 = 48 terms.
    
    # For each (i, j, k') where i in 0..3, j in 0..3, k' in 0..2:
    # U[i*4+j, col] = 1
    # V[j*7 + 4 + k', col] = 1
    # W[(4+k')*4 + i, col] = 1
    
    for i in range(4):
        for j in range(4):
            for kp in range(3):
                U[i*4 + j, col_off] = 1
                V[j*7 + 4 + kp, col_off] = 1
                W[(4+kp)*4 + i, col_off] = 1
                col_off += 1
    
    return U, V, W, rank


def construct():
    """Construct the tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    deadline = time.time() + 120
    
    U, V, W, rank = _build_4x4x7()
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
