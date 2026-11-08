"""Matrix multiplication 2x4x4 with rank-minimal exact tensor decomposition."""
import time
import random

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact decomposition for 2x4x4 matrix multiplication.
    
    Strategy: Split the 4 columns of A into two 2-column blocks, and the 4 rows of B
    into two 2-row blocks. This gives 4 sub-problems of 2x2x2, each solvable with
    Strassen (rank 7). Total rank = 4 * 7 = 28.
    
    The multiplication is: C = A * B where A is 2x4, B is 4x4, C is 2x4.
    
    Let A = [A1 | A2] where A1, A2 are 2x2.
    Let B = [B1; B2] where B1, B2 are 2x2 (top and bottom halves).
    
    Then C = A1*B1 + A2*B2.
    
    For each 2x2x2 multiplication, use Strassen's algorithm with 7 products.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to decompose C = A1*B1 + A2*B2
    # Each 2x2x2 block multiplication uses 7 rank-1 terms.
    # Total: 4 blocks * 7 = 28 rank-1 terms.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Indexing:
    # A[i, j] where i in 0..1 (rows), j in 0..3 (cols)
    #   A1: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3 in A
    #   A2: A[0,2], A[0,3], A[1,2], A[1,3] -> indices 4,5,6,7 in A
    #
    # B[j, k] where j in 0..3 (rows), k in 0..3 (cols)
    #   B1: rows 0,1 -> B[0,0],B[0,1],B[1,0],B[1,1] -> indices 0,1,4,5 in B
    #   B2: rows 2,3 -> B[2,0],B[2,1],B[3,0],B[3,1] -> indices 8,9,12,13 in B
    #
    # C[i, k] where i in 0..1 (rows), k in 0..3 (cols)
    #   Output C has 2 rows, 4 cols
    #   W[k*n + i, r] = W[k*2 + i, r]
    
    # Strassen coefficients for a 2x2x2 multiplication:
    # For matrices X (2x2), Y (2x2), Z = X*Y (2x2):
    # X indices: x00, x01, x10, x11
    # Y indices: y00, y01, y10, y11
    # Z indices: z00, z01, z10, z11
    
    # The 7 Strassen products and their contributions:
    # Each product is a rank-1 term: U_part * V_part * W_part
    
    # For a single 2x2x2 with X (2x2), Y (2x2), Z (2x2):
    # U rows correspond to X entries, V rows to Y entries, W rows to Z entries.
    
    # Strassen terms (u_coeff, v_coeff, w_coeff) for each of the 7 products:
    # Term 1: (x00+x11)*(y00+y11) contributes to z00(+1), z01(0), z10(0), z11(-1)
    #   Wait, let me re-derive properly.
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B21-B22)
    # M4 = (A22)(B12-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Strassen product Mi, we have:
    # U coefficients for the A-matrix entries
    # V coefficients for the B-matrix entries
    # W coefficients for the C-matrix entries
    
    # Let's define the 7 terms for a 2x2x2:
    # Each term t has: u_vec (length 4), v_vec (length 4), w_vec (length 4)
    # where indices 0,1,2,3 correspond to (00,01,10,11)
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00:+1, C11:-1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, -1]),
        # M2: (A10+A11)(B00) -> C10:+1, C11:-1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3: (A00)(B10-B11) -> C01:+1, C11:+1
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # M4: (A11)(B01-B00) -> C00:+1, C10:+1
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5: (A00+A01)(B11) -> C00:-1, C01:+1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (A10-A00)(B00+B01) -> C11:+1
        ([0, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11) -> C00:+1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Now we have 4 sub-problems:
    # Block (A1, B1): A1 is A[:, 0:2], B1 is B[0:2, :]
    # Block (A2, B2): A2 is A[:, 2:4], B2 is B[2:4, :]
    
    # For block (Ai, Bj), the contribution to C is Ai * Bj.
    
    # A1 entries: A[0,0]=idx0, A[0,1]=idx1, A[1,0]=idx2, A[1,1]=idx3
    # A2 entries: A[0,2]=idx4, A[0,3]=idx5, A[1,2]=idx6, A[1,3]=idx7
    
    # B1 entries (rows 0,1): B[0,0]=idx0, B[0,1]=idx1, B[1,0]=idx4, B[1,1]=idx5
    # B2 entries (rows 2,3): B[2,0]=idx8, B[2,1]=idx9, B[3,0]=idx12, B[3,1]=idx13
    
    # C entries: C[0,0]=idx0, C[0,1]=idx1, C[0,2]=idx2, C[0,3]=idx3
    #             C[1,0]=idx4, C[1,1]=idx5, C[1,2]=idx6, C[1,3]=idx7
    # W index: W[k*n + i] = W[k*2 + i]
    # C[0,0] -> W[0*2+0]=W[0], C[0,1] -> W[1*2+0]=W[2], C[0,2] -> W[2*2+0]=W[4], C[0,3] -> W[3*2+0]=W[6]
    # C[1,0] -> W[0*2+1]=W[1], C[1,1] -> W[1*2+1]=W[3], C[1,2] -> W[2*2+1]=W[5], C[1,3] -> W[3*2+1]=W[7]
    
    # For a 2x2 result Z with entries z00, z01, z10, z11:
    # z00 -> C[i0, k0], z01 -> C[i0, k1], z10 -> C[i1, k0], z11 -> C[i1, k1]
    # where i0, i1 are the row indices in C, k0, k1 are the col indices in C.
    
    # For block (A1, B1): i0=0, i1=1, k0=0, k1=1
    #   z00 -> C[0,0] -> W[0], z01 -> C[0,1] -> W[2], z10 -> C[1,0] -> W[1], z11 -> C[1,1] -> W[3]
    
    # For block (A2, B2): i0=0, i1=1, k0=2, k1=3
    #   z00 -> C[0,2] -> W[4], z01 -> C[0,3] -> W[6], z10 -> C[1,2] -> W[5], z11 -> C[1,3] -> W[7]
    
    # U indices for A1: [0, 1, 2, 3] corresponding to (00, 01, 10, 11) of A1
    # U indices for A2: [4, 5, 6, 7] corresponding to (00, 01, 10, 11) of A2
    
    # V indices for B1: [0, 1, 4, 5] corresponding to (00, 01, 10, 11) of B1
    #   B1[0,0]=B[0,0]=V[0], B1[0,1]=B[0,1]=V[1], B1[1,0]=B[1,0]=V[4], B1[1,1]=B[1,1]=V[5]
    
    # V indices for B2: [8, 9, 12, 13] corresponding to (00, 01, 10, 11) of B2
    #   B2[0,0]=B[2,0]=V[8], B2[0,1]=B[2,1]=V[9], B2[1,0]=B[3,0]=V[12], B2[1,1]=B[3,1]=V[13]
    
    # W indices for block (A1,B1) result:
    #   z00 -> W[0], z01 -> W[2], z10 -> W[1], z11 -> W[3]
    #   So w_indices = [0, 2, 1, 3]
    
    # W indices for block (A2,B2) result:
    #   z00 -> W[4], z01 -> W[6], z10 -> W[5], z11 -> W[7]
    #   So w_indices = [4, 6, 5, 7]
    
    blocks = [
        # (u_indices, v_indices, w_indices)
        # Block (A1, B1)
        ([0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 1, 3]),
        # Block (A2, B2)
        ([4, 5, 6, 7], [8, 9, 12, 13], [4, 6, 5, 7]),
    ]
    
    col = 0
    for u_idx, v_idx, w_idx in blocks:
        for u_coeff, v_coeff, w_coeff in strassen_terms:
            for t in range(4):
                if u_coeff[t] != 0:
                    U[u_idx[t]][col] += u_coeff[t]
                if v_coeff[t] != 0:
                    V[v_idx[t]][col] += v_coeff[t]
                if w_coeff[t] != 0:
                    W[w_idx[t]][col] += w_coeff[t]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
