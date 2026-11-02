import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    
    # Target rank is 28 (4 blocks of rank 7)
    # We embed into 4x8 x 8x8 -> 4x8 multiplication
    
    # 1. Define the 4x4x4 rank-7 decomposition (Coppersmith-Winograd M4)
    # U4, V4, W4 are 16x7 matrices
    
    # The specific coefficients for M4 (rank 7)
    # Source: Coppersmith & Winograd 1987, or subsequent explicit constructions.
    # One explicit set of 7 bilinear terms for 4x4x4:
    # Let indices be (i,j) for A, (j,k) for B, (i,k) for C.
    # The 7 terms P1..P7 are:
    # P1 = A11 B11
    # P2 = A12 B22
    # P3 = A21 B11
    # P4 = A22 B22
    # P5 = (A11+A22)(B11+B22)
    # P6 = (A12+A21)(B12+B21)
    # P7 = (A11-A22)(B11-B22) ?? 
    # Wait, the standard rank 7 algorithm is more complex.
    
    # Let's use a verified rank-7 decomposition.
    # From "On the rank of the matrix multiplication tensor"
    # The 7 vectors for U, V, W can be constructed.
    
    # Here is a known set of 7 tuples (u, v, w) where u, v, w are 16-dim vectors.
    # Actually, it's easier to just define the non-zero entries.
    
    # U4, V4, W4
    U4 = np.zeros((16, 7), dtype=np.int64)
    V4 = np.zeros((16, 7), dtype=np.int64)
    W4 = np.zeros((16, 7), dtype=np.int64)
    
    # Indices for 4x4 matrix: i*4 + j
    # A indices: 0..15
    # B indices: 0..15
    # C indices: 0..15
    
    # Term 1: A00 * B00 -> C00
    # A00 is idx 0, B00 is idx 0, C00 is idx 0
    U4[0, 0] = 1
    V4[0, 0] = 1
    W4[0, 0] = 1
    
    # Term 2: A01 * B11 -> C01
    # A01 is idx 1, B11 is idx 5, C01 is idx 1
    U4[1, 1] = 1
    V4[5, 1] = 1
    W4[1, 1] = 1
    
    # Term 3: A10 * B00 -> C10
    # A10 is idx 4, B00 is idx 0, C10 is idx 4
    U4[4, 2] = 1
    V4[0, 2] = 1
    W4[4, 2] = 1
    
    # Term 4: A11 * B11 -> C11
    # A11 is idx 5, B11 is idx 5, C11 is idx 5
    U4[5, 3] = 1
    V4[5, 3] = 1
    W4[5, 3] = 1
    
    # Term 5: (A00 + A11) * (B00 + B11) -> C00 + C11
    # U: A00(1), A11(1)
    # V: B00(1), B11(1)
    # W: C00(1), C11(1)
    U4[0, 4] = 1
    U4[5, 4] = 1
    V4[0, 4] = 1
    V4[5, 4] = 1
    W4[0, 4] = 1
    W4[5, 4] = 1
    
    # Term 6: (A01 + A10) * (B01 + B10) -> C01 + C10
    # U: A01(1), A10(1)
    # V: B01(1), B10(1)
    # W: C01(1), C10(1)
    U4[1, 5] = 1
    U4[4, 5] = 1
    V4[1, 5] = 1
    V4[4, 5] = 1
    W4[1, 5] = 1
    W4[4, 5] = 1
    
    # Term 7: (A00 - A11) * (B00 - B11) -> C00 - C11 ?? 
    # This gives C00 and C11.
    # We already have C00, C11 from Term 1, 4, 5.
    # We need to solve for all C entries.
    # The standard M4 algorithm uses specific combinations to cover all 16 C entries with 7 products.
    
    # The above 7 terms only cover C00, C01, C10, C11.
    # We need C02, C03, C12, C13, C20... C33.
    # The 4x4x4 rank 7 algorithm is not just on the 2x2 sub-blocks.
    # It uses the structure of the whole 4x4.
    
    # I will use a known explicit rank-7 decomposition for 4x4x4.
    # Reference: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm#Coppersmith%E2%80%93Winograd
    # The algorithm is often described recursively or with specific patterns.
    
    # Alternative: Use the fact that rank(4x4x4) <= 7 is a known result.
    # I can hardcode the 16x7 matrices from a trusted source or generate them if I know the bilinear forms.
    
    # Let's try to construct it using the "strassen-like" approach for 4x4.
    # Actually, there is a simpler way:
    # The rank of 4x4x4 is 7.
    # I will hardcode the U4, V4, W4 matrices from a known implementation.
    
    # If I cannot find the exact hardcoded values in time, I might have to fall back to schoolbook for the blocks?
    # Rank of 4x4x4 schoolbook is 16.
    # 4 blocks * 16 = 64.
    # 64 < 105.
    # So even using schoolbook for the 4x4 blocks in the 4x8x8 embedding yields rank 64.
    # This is still an improvement over 105.
    
    # Let's implement the 4x8x8 embedding with schoolbook 4x4x4 blocks first to ensure correctness, 
    # then try to optimize to rank 7 if possible.
    
    # Schoolbook 4x4x4 rank is 16.
    # Total rank 64.
    
    # Let's implement the 4x8x8 embedding.
    
    # Dimensions:
    # A: 4x8 (32 elements)
    # B: 8x8 (64 elements)
    # C: 4x8 (32 elements)
    
    # We split A into A1 (4x4), A2 (4x4)
    # We split B into B11, B12, B21, B22 (4x4 each)
    # We split C into C1 (4x4), C2 (4x4)
    
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    
    # We need 4 matrix multiplications of 4x4 matrices.
    # M1 = A1 B11
    # M2 = A2 B21
    # M3 = A1 B12
    # M4 = A2 B22
    
    # Using schoolbook for each 4x4x4 (rank 16):
    # Total rank = 4 * 16 = 64.
    
    # U_total: 32 x 64
    # V_total: 64 x 64
    # W_total: 32 x 64
    
    # Then slice for 3x5x7.
    
    rank = 64
    U = np.zeros((32, rank), dtype=np.int64)
    V = np.zeros((64, rank), dtype=np.int64)
    W = np.zeros((32, rank), dtype=np.int64)
    
    # Helper to fill schoolbook 4x4x4 block
    # A_idx_offset, B_idx_offset, C_idx_offset, col_offset
    
    def add_block(a_off, b_off, c_off, col_off):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    u_idx = a_off + i * 4 + j
                    v_idx = b_off + j * 4 + k
                    w_idx = c_off + k * 4 + i # Trace convention: W[k*n+i, r] for 4x4? 
                    # Wait, the trace convention in the task is W[k*n+i, r].
                    # For a 4x4 block, n=4, p=4.
                    # So W index is k*4 + i.
                    
                    U[u_idx, col_off + i*16 + j*4 + k] = 1
                    V[v_idx, col_off + i*16 + j*4 + k] = 1
                    W[w_idx, col_off + i*16 + j*4 + k] = 1
                    
    # Block 1: A1 * B11 -> C1_part1
    # A1 is rows 0-3, cols 0-3 of A. Offset 0.
    # B11 is rows 0-3, cols 0-3 of B. Offset 0.
    # C1 is rows 0-3, cols 0-3 of C. Offset 0.
    add_block(0, 0, 0, 0)
    
    # Block 2: A2 * B21 -> C1_part2
    # A2 is rows 0-3, cols 4-7 of A. Offset 16 (4*4).
    # B21 is rows 4-7, cols 0-3 of B. Offset 16 (4*4).
    # C1 is rows 0-3, cols 0-3 of C. Offset 0.
    add_block(16, 16, 0, 16)
    
    # Block 3: A1 * B12 -> C2_part1
    # A1 is rows 0-3, cols 0-3 of A. Offset 0.
    # B12 is rows 0-3, cols 4-7 of B. Offset 4 (0*4 + 4).
    # C2 is rows 0-3, cols 4-7 of C. Offset 4 (0*4 + 4).
    add_block(0, 4, 4, 32)
    
    # Block 4: A2 * B22 -> C2_part2
    # A2 is rows 0-3, cols 4-7 of A. Offset 16.
    # B22 is rows 4-7, cols 4-7 of B. Offset 20 (4*4 + 4).
    # C2 is rows 0-3, cols 4-7 of C. Offset 4.
    add_block(16, 20, 4, 48)
    
    # Now slice for 3x5x7
    # A is 3x5. In 4x8 padding:
    # Rows 0-2, Cols 0-4 are valid.
    # A indices in 4x8: i*8 + j.
    # Valid A indices: i in 0..2, j in 0..4.
    
    # B is 5x7. In 8x8 padding:
    # Rows 0-4, Cols 0-6 are valid.
    # B indices in 8x8: i*8 + j.
    # Valid B indices: i in 0..4, j in 0..6.
    
    # C is 3x7. In 4x8 padding:
    # Rows 0-2, Cols 0-6 are valid.
    # C indices in 4x8: i*8 + j.
    # Valid C indices: i in 0..2, j in 0..6.
    # Note: The task says W[k*n+i, r]. For 3x7, n=3, p=7.
    # So W index is k*3 + i.
    # My W matrix is constructed for 4x8 output (n=4, p=8).
    # W index in 4x8 is k*4 + i.
    # I need to map the 4x8 W indices to 3x7 W indices?
    # No, the W matrix in the output must have shape (p*n, rank) = (7*3, rank) = (21, rank).
    # The entries must correspond to the trace convention of the 3x7 multiplication.
    # The decomposition is exact for the padded problem.
    # The valid parts of C are C[i,j] for i<3, j<7.
    # In the 4x8 W matrix, C[i,j] is at index j*4 + i.
    # In the 3x7 W matrix, C[i,j] is at index j*3 + i.
    # So I need to extract the rows of W_total corresponding to valid C entries and permute them to match the 3x7 indexing.
    
    # Let's build the slices.
    
    # U slice: 3x5 -> 15 rows.
    # A indices: i*8 + j for i in 0..2, j in 0..4.
    u_indices = [i*8 + j for i in range(3) for j in range(5)]
    U_final = U[u_indices, :]
    
    # V slice: 5x7 -> 35 rows.
    # B indices: i*8 + j for i in 0..4, j in 0..6.
    v_indices = [i*8 + j for i in range(5) for j in range(7)]
    V_final = V[v_indices, :]
    
    # W slice: 3x7 -> 21 rows.
    # We need to map 4x8 C indices (j*4 + i) to 3x7 W indices (j*3 + i).
    # For each valid i<3, j<7:
    # src_idx = j*4 + i
    # dst_idx = j*3 + i
    w_indices = [j*4 + i for j in range(7) for i in range(3)]
    # Wait, the order in V and U is row-major.
    # U_final rows are ordered by i, then j?
    # u_indices = [i*8 + j for i in range(3) for j in range(5)] -> i=0,j=0..4; i=1,j=0..4...
    # This matches A[i,j].
    
    # V_final rows are ordered by i, then j?
    # v_indices = [i*8 + j for i in range(5) for j in range(7)] -> i=0,j=0..6...
    # This matches B[i,j].
    
    # W_final rows must be ordered such that W[k*n+i, r] corresponds to C[i,k].
    # The task says W[k*n+i, r].
    # So row index is k*3 + i.
    # My w_indices list should generate the source row indices in W_total for the destination rows.
    # Dest row r = k*3 + i.
    # Source row s = k*4 + i.
    # So for r from 0 to 20:
    # k = r // 3
    # i = r % 3
    # s = k*4 + i
    # W_final[r, :] = W_total[s, :]
    
    W_final = np.zeros((21, rank), dtype=np.int64)
    for r in range(21):
        k = r // 3
        i = r % 3
        s = k * 4 + i
        W_final[r, :] = W_total[s, :] if False else W[s, :] # W is the total matrix
    
    return {'U': U_final.tolist(), 'V': V_final.tolist(), 'W': W_final.tolist()}
