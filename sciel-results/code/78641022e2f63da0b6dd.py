import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    n, m, p = 4, 4, 5
    
    # Strassen 2x2x2 basis (rank 7)
    # For multiplying 2x2 matrices: C = A * B
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U_strassen for 2x2x2: shape (4, 7) - maps A entries to M values
    # A is 2x2, indexed as A11, A12, A21, A22 (row-major: A[0,0], A[0,1], A[1,0], A[1,1])
    U_2x2 = [
        [1, 0, 0, 1, 0, 0, 0],  # A11
        [0, 1, 0, 0, 1, 0, 0],  # A12
        [0, 0, 1, 0, 0, 1, 0],  # A21
        [1, 0, 1, 0, 0, 0, 0]   # A22
    ]
    # Wait, let me redo this properly.
    # M1 = (A11 + A22)(B11 + B22): A contributes A11+A22
    # M2 = (A21 + A22)B11: A contributes A21+A22
    # M3 = A11(B21 - B22): A contributes A11
    # M4 = A22(B12 - B11): A contributes A22
    # M5 = (A11 + A12)B22: A contributes A11+A12
    # M6 = (A21 - A11)(B11 + B12): A contributes A21-A11
    # M7 = (A12 - A22)(B21 + B22): A contributes A12-A22
    
    U_2x2 = [
        [1, 1, 1, 0, 1, -1, 0],  # A11: appears in M1, M3, M5, M6
        [0, 0, 0, 0, 1, 0, 1],   # A12: appears in M5, M7
        [0, 1, 0, 0, 0, 1, 0],   # A21: appears in M2, M6
        [1, 1, 0, 1, 0, 0, -1]   # A22: appears in M1, M2, M4, M7
    ]
    
    V_2x2 = [
        [1, 0, 0, 0, 0, 1, 0],   # B11: M1, M2, M6
        [0, 0, -1, -1, 0, 0, 1], # B12: M4, M6, M7
        [0, 0, 1, 0, 0, 0, 1],   # B21: M3, M7
        [1, 0, 0, 0, 1, 0, 0]    # B22: M1, M5, M7
    ]
    
    W_2x2 = [
        [1, 0, 0, 1, -1, 0, 1],  # C11: M1, M4, M5, M7
        [0, 0, 1, 0, 1, 0, 0],   # C12: M3, M5
        [0, 1, 0, 1, 0, 0, 0],   # C21: M2, M4
        [1, -1, 1, 0, 0, 1, 0]   # C22: M1, M2, M3, M6
    ]
    
    # Verify 2x2x2 Strassen
    U_2x2 = np.array(U_2x2, dtype=float)
    V_2x2 = np.array(V_2x2, dtype=float)
    W_2x2 = np.array(W_2x2, dtype=float)
    
    # Build the 4x4x4 tensor using Strassen on 2x2 blocks
    # A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks: A = [[A11,A12],[A21,A22]] where each is 2x2
    # Same for B and C
    
    # For 4x4x4 using Strassen on 2x2 sub-blocks:
    # We have 4 blocks of A (each 2x2), 4 blocks of B (each 2x2), 4 blocks of C (each 2x2)
    # Each block multiplication uses rank 7, so total rank = 4 * 7 = 28
    
    # U for 4x4x4: shape (16, 28)
    # A has 16 entries (4x4), organized as 4 blocks of 2x2
    # Block (bi, bj) of A has entries at positions (bi*2+di, bj*2+dj) for d in {0,1}
    # For each block pair, we apply the 2x2 Strassen decomposition
    
    U_4x4 = [[0] * 28 for _ in range(16)]
    V_4x4 = [[0] * 28 for _ in range(16)]
    W_4x4 = [[0] * 28 for _ in range(16)]
    
    # Block indices: bi, bj in {0,1} for A blocks, ci, cj in {0,1} for B/C blocks
    # A block (bi,bj) multiplies B block (cj,ci) to give C block (bi,cj)
    # Wait, let me think about this more carefully.
    
    # A is 4x4, B is 4x4. C[i,k] = sum_j A[i,j] * B[j,k]
    # Split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where Bij is 2x2
    # C = [[C00, C01], [C10, C11]] where Cij is 2x2
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # For each of these 4 block multiplications, we use Strassen (rank 7)
    # Total rank = 4 * 7 = 28
    
    # Let's build U, V, W for 4x4x4
    # A entries: A[i,j] for i,j in 0..3
    #   A00 block: A[0,0], A[0,1], A[1,0], A[1,1]
    #   A01 block: A[0,2], A[0,3], A[1,2], A[1,3]
    #   A10 block: A[2,0], A[2,1], A[3,0], A[3,1]
    #   A11 block: A[2,2], A[2,3], A[3,2], A[3,3]
    
    # B entries: B[j,k] for j,k in 0..3
    #   B00 block: B[0,0], B[0,1], B[1,0], B[1,1]
    #   B01 block: B[0,2], B[0,3], B[1,2], B[1,3]
    #   B10 block: B[2,0], B[2,1], B[3,0], B[3,1]
    #   B11 block: B[2,2], B[2,3], B[3,2], B[3,3]
    
    # C entries: C[i,k] for i,k in 0..3
    #   C00 block: C[0,0], C[0,1], C[1,0], C[1,1]
    #   C01 block: C[0,2], C[0,3], C[1,2], C[1,3]
    #   C10 block: C[2,0], C[2,1], C[3,0], C[3,1]
    #   C11 block: C[2,2], C[2,3], C[3,2], C[3,3]
    
    # For each block multiplication, assign 7 rank terms
    # Block mult 1: A00 * B00 -> C00, rank offset 0
    # Block mult 2: A01 * B10 -> C00, rank offset 7
    # Block mult 3: A00 * B01 -> C01, rank offset 14
    # Block mult 4: A01 * B11 -> C01, rank offset 21
    # Block mult 5: A10 * B00 -> C10, rank offset 28
    # Block mult 6: A11 * B10 -> C10, rank offset 35
    # Block mult 7: A10 * B01 -> C11, rank offset 42
    # Block mult 8: A11 * B11 -> C11, rank offset 49
    
    # Wait, that's 8 * 7 = 56, not 28. I need to be more careful.
    
    # Actually, C00 = A00*B00 + A01*B10. This is a sum of two 2x2x2 tensors.
    # We can't just add ranks directly unless we share the M values.
    # But Strassen works on a single 2x2x2 product. For sums, we need separate decompositions.
    
    # So for 4x4x4, if we use Strassen naively on each 2x2 block product:
    # There are 4*4 = 16 block products? No.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In block form: C_block(i,c) = sum_j A_block(i,j) * B_block(j,c)
    # For i,c in {0,1}, j in {0,1}: that's 2*2*2 = 8 block multiplications
    # Each is 2x2x2 with rank 7, so total rank = 8*7 = 56
    
    # That's what inspiration 3 mentioned: rank 56 for 4x4x4.
    
    # Let me build the 4x4x4 Strassen decomposition with rank 56.
    
    U_4x4 = [[0] * 56 for _ in range(16)]
    V_4x4 = [[0] * 56 for _ in range(16)]
    W_4x4 = [[0] * 56 for _ in range(16)]
    
    # Block A(i,j): rows 2i..2i+1, cols 2j..2j+1
    # Block B(j,k): rows 2j..2j+1, cols 2k..2k+1
    # Block C(i,k): rows 2i..2i+1, cols 2k..2k+1
    
    # For each (i, j, k) in {0,1}^3, we have A_block(i,j) * B_block(j,k) contributing to C_block(i,k)
    # Wait, C_block(i,k) = sum_j A_block(i,j) * B_block(j,k)
    # So for each (i,k) pair, there are 2 j values.
    
    # Let's enumerate: for i in {0,1}, j in {0,1}, k in {0,1}:
    # A_block(i,j) * B_block(j,k) contributes to C_block(i,k)
    # That's 8 such products, each rank 7, total 56.
    
    U_2x2_np = np.array(U_2x2, dtype=float)
    V_2x2_np = np.array(V_2x2, dtype=float)
    W_2x2_np = np.array(W_2x2, dtype=float)
    
    rank_offset = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # A_block(i,j): entries at A[2i+d1, 2j+d2] for d1,d2 in {0,1}
                # In flat index (row-major): A[2i+d1, 2j+d2] -> (2i+d1)*4 + (2j+d2)
                # B_block(j,k): entries at B[2j+d1, 2k+d2] for d1,d2 in {0,1}
                # In flat index: (2j+d1)*4 + (2k+d2)
                # C_block(i,k): entries at C[2i+d1, 2k+d2] for d1,d2 in {0,1}
                # In flat index: (2i+d1)*4 + (2k+d2)
                
                for d in range(4):
                    # d maps to (d1, d2) where d1 = d//2, d2 = d%2
                    d1 = d // 2
                    d2 = d % 2
                    
                    # A entry: row 2i+d1, col 2j+d2
                    a_idx = (2*i + d1) * 4 + (2*j + d2)
                    # B entry: row 2j+d1, col 2k+d2
                    b_idx = (2*j + d1) * 4 + (2*k + d2)
                    # C entry: row 2i+d1, col 2k+d2
                    c_idx = (2*i + d1) * 4 + (2*k + d2)
                    
                    for s in range(7):
                        U_4x4[a_idx][rank_offset + s] += U_2x2_np[d, s]
                        V_4x4[b_idx][rank_offset + s] += V_2x2_np[d, s]
                        W_4x4[c_idx][rank_offset + s] += W_2x2_np[d, s]
                
                rank_offset += 7
    
    # Now for the 4x4x5 problem:
    # A is 4x4, B is 4x5, C is 4x5
    # C[i,k] = sum_j A[i,j] * B[j,k] for k in 0..4
    
    # Split into: 4x4x4 part (k=0..3) + 4x4x1 part (k=4)
    # For 4x4x4: use Strassen rank 56
    # For 4x4x1: schoolbook rank 16
    
    # Total rank = 56 + 16 = 72
    
    total_rank = 72
    U = [[0] * total_rank for _ in range(16)]  # n*m = 4*4 = 16
    V = [[0] * total_rank for _ in range(20)]  # m*p = 4*5 = 20
    W = [[0] * total_rank for _ in range(20)]  # p*n = 5*4 = 20
    
    # Copy 4x4x4 Strassen part
    for i in range(16):
        for s in range(56):
            U[i][s] = U_4x4[i][s]
            # V: for 4x4x4, B is 4x4, so V has 16 rows
            # But our V is for 4x5, so B has 20 entries
            # The 4x4 part of B uses entries 0..15
            V[i][s] = V_4x4[i][s]
            # W: for 4x4x4, C is 4x4, so W has 16 rows
            # But our W is for 4x5, so C has 20 entries
            # The 4x4 part of C uses entries 0..15
            W[i][s] = W_4x4[i][s]
    
    # Schoolbook for 4x4x1 (k=4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # For each (i,j) pair, one rank term
    for i in range(4):
        for j in range(4):
            s = 56 + i * 4 + j
            # A[i,j]: flat index i*4+j
            U[i * 4 + j][s] = 1
            # B[j,4]: flat index j*5+4
            V[j * 5 + 4][s] = 1
            # C[i,4]: flat index 4*4+i = 16+i (since p*n = 5*4, W[k*n+i] with k=4, i=i)
            W[4 * 4 + i][s] = 1
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 5)
    C_expected = A_test @ B_test
    
    # Compute using decomposition
    # C[i,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    # Wait, the convention is W[k*n+i, r]
    # U has shape (n*m, rank) = (16, 72), indexed by i*m+j
    # V has shape (m*p, rank) = (20, 72), indexed by j*p+k
    # W has shape (p*n, rank) = (20, 72), indexed by k*n+i
    
    C_computed = np.zeros((4, 5))
    for i in range(4):
        for k in range(5):
            val = 0.0
            for j in range(4):
                for s in range(total_rank):
                    val += U_np[i * 4 + j, s] * V_np[j * 5 + k, s] * W_np[k * 4 + i, s]
            C_computed[i, k] = val
    
    max_err = np.max(np.abs(C_computed - C_expected))
    
    if max_err > 1e-10:
        # Fallback to schoolbook rank 80
        rank = 80
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(20)]
        W = [[0] * rank for _ in range(20)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U[i * 4 + j][col] = 1
                    V[j * 5 + k][col] = 1
                    W[k * 4 + i][col] = 1
                    col += 1
    else:
        # Convert to integers if possible
        U = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in U]
        V = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in V]
        W = [[int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
