"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # We need to decompose the matrix multiplication tensor for 4x4 * 4x5 = 4x5
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k), 0 otherwise
    # U: (n*m, rank) = (16, rank)
    # V: (m*p, rank) = (20, rank)  
    # W: (p*n, rank) = (20, rank)
    
    # Strategy: Use the known rank-19 decomposition for 4x4x4 matrix multiplication
    # (Coppersmith et al.) and extend it to handle the 5th column with 4 additional terms.
    # Total rank: 19 + 4 = 23
    
    # First, let's build the rank-19 scheme for 4x4x4.
    # The rank-19 scheme for 4x4x4 uses a specific set of 19 triplets.
    # For 4x4x4: T[i*4+j, j*4+k, k*4+i] = 1
    # We need U4 (16x19), V4 (16x19), W4 (16x19)
    
    # Known rank-19 decomposition for 4x4x4 matrix multiplication
    # This is based on the Coppersmith et al. construction
    # The 19 terms are specified by their (i,j,k) indices and coefficients
    
    # For the 4x4x4 case, the rank-19 scheme can be constructed as follows:
    # We use a combination of Strassen-like blocks
    # Let me use a verified construction
    
    # Actually, let me use a simpler but still valid approach:
    # Use the standard rank-20 decomposition for 4x4x4 (which is well-known)
    # and add 4 terms for the 5th column, giving rank 24.
    
    # Or better: use the rank-19 if I can verify it.
    # Let me use a constructive approach with known integer/half-integer coefficients.
    
    # For 4x4x4, the following rank-19 scheme is known to work:
    # I'll construct it using the method from Coppersmith et al.
    
    # Let me use a verified rank-20 scheme for 4x4x4 and add 4 for the 5th column = rank 24
    # This is simpler to verify and still better than the schoolbook rank of 65.
    
    # Rank-20 scheme for 4x4x4:
    # This uses the following 20 triplets with coefficients
    # I'll use a known valid construction
    
    # Actually, the simplest correct approach: 
    # Use the schoolbook for the first 4 columns (rank 16) and add 4 terms for column 5 = rank 20
    # Wait, that's not right either. Let me think more carefully.
    
    # For 4x4 * 4x5:
    # The output has 5 columns. For each column k (0..4), we compute A * B[:,k]
    # For column k, the standard rank is 16 (4*4), using terms (i, j, k) for all i,j
    # So schoolbook rank = 16 * 5 = 80? No wait.
    
    # Let me re-examine. The tensor T[a,b,c] where:
    # a = i*m+j = i*4+j (i=0..3, j=0..3) -> 16 values
    # b = j*p+k = j*5+k (j=0..3, k=0..4) -> 20 values  
    # c = k*n+i = k*4+i (k=0..4, i=0..3) -> 20 values
    # T[a,b,c] = 1 iff the indices are consistent: a=i*4+j, b=j*5+k, c=k*4+i
    
    # So for each (i,j,k) with i in 0..3, j in 0..3, k in 0..4, we have one non-zero entry.
    # Total non-zero entries: 4*4*5 = 80
    
    # Schoolbook decomposition: one term per (i,j,k), so rank = 80
    # U[i*4+j, col] = 1, V[j*5+k, col] = 1, W[k*4+i, col] = 1
    
    # To reduce rank, we need to find shared structure.
    
    # Key observation: For a fixed k, the sub-tensor for column k is a 4x4x4 matrix multiplication tensor.
    # So we can decompose each column's 4x4x4 tensor with rank r, and the total rank is 5*r
    # if we don't share anything between columns.
    
    # But we CAN share between columns! The U factors don't depend on k, the V factors depend on j and k, 
    # and the W factors depend on k and i.
    
    # Better approach: Use the fact that for each pair (i,j), we have 5 terms (one for each k).
    # These 5 terms share the same U factor (for fixed i,j).
    # So for each (i,j), we need to decompose the 5-dimensional vector in the W space.
    
    # For fixed (i,j), the non-zero W indices are: k*4+i for k=0..4
    # That's 5 specific positions in the 20-dimensional W space.
    # The V factor for this (i,j) has non-zero entries at j*5+k for k=0..4, which is 5 consecutive positions.
    
    # This suggests a block diagonal structure where each (i,j) block handles 5 k-values.
    # For each (i,j), we need a rank-r decomposition of a 5x5 "identity-like" structure.
    # The minimum rank for this is 5 (since the 5 values are linearly independent).
    # So total rank = 16 * 5 = 80? That's the same as schoolbook.
    
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the structure is:
    # For fixed i, j: U has value at position i*4+j
    # V has values at positions j*5, j*5+1, ..., j*5+4 (5 positions)
    # W has values at positions i, i+4, i+8, i+12, i+16 (5 positions)
    
    # The sub-tensor for fixed (i,j) is a rank-1 tensor in the U dimension (just one entry),
    # but a 5x5 structure in the V-W plane.
    # Specifically, for fixed (i,j), the V-W sub-tensor has V at positions j*5+k and W at positions k*4+i.
    # This is a 5x5 identity matrix (if we index properly), which has rank 5.
    # So each (i,j) contributes rank 5, total rank = 16*5 = 80.
    
    # That's the schoolbook rank. To do better, we need to share factors across different (i,j) pairs.
    
    # The standard way to reduce rank in matrix multiplication is to use linear combinations.
    # For example, Strassen's trick for 2x2x2 reduces rank from 8 to 7.
    
    # For 4x4x4, the best known rank is 19 (or 20 depending on the scheme).
    # For 4x4x5, I believe the rank is around 24-30.
    
    # Let me try a different structural approach: 
    # Use the rank-19 scheme for 4x4x4 for the first 4 columns, and add 4 more terms for the 5th column.
    # But the 5th column is a 4x4*4x1 multiplication, which has rank 4.
    # So total rank = 19 + 4 = 23.
    
    # Let me construct this explicitly.
    
    # First, I need a valid rank-19 scheme for 4x4x4 matrix multiplication.
    # I'll use a known construction.
    
    # For 4x4x4, the tensor is T4[i*4+j, j*4+k, k*4+i] = 1
    # I need U4 (16x19), V4 (16x19), W4 (16x19)
    
    # Known rank-19 scheme for 4x4x4 (from literature):
    # This is complex to hardcode. Let me use a simpler rank-20 scheme instead.
    
    # Rank-20 scheme for 4x4x4:
    # I can construct this using two 3x3x3 blocks and cross terms, or use a known explicit scheme.
    
    # Actually, let me just use a verified approach: 
    # Construct the decomposition using integer programming or a known explicit scheme.
    
    # For simplicity and correctness, let me use the following approach:
    # 1. For the first 4 columns (k=0,1,2,3), use a rank-20 scheme for 4x4x4
    # 2. For the 5th column (k=4), use 4 rank-1 terms (schoolbook for 4x4*4x1)
    # Total rank: 20 + 4 = 24
    
    # Let me construct the rank-20 scheme for 4x4x4.
    # A simple rank-20 scheme can be obtained by splitting 4x4 into 2x2 blocks and using Strassen.
    
    # 4x4x4 using 2x2x2 blocks:
    # Split A into 4 2x2 blocks, B into 4 2x2 blocks
    # Use Strassen for each pair of blocks
    # This gives rank 7*4 = 28 for the block diagonal, but we need cross terms too.
    
    # This is getting complicated. Let me use a different approach.
    
    # I'll use the following constructive method:
    # For each (i,j), create 5 terms for k=0..4, but then apply linear combinations
    # to reduce the total rank.
    
    # Actually, the cleanest approach for a valid submission:
    # Use the schoolbook rank of 80, which is guaranteed correct.
    # But that's not better than the current score of 1.0 (which corresponds to rank 65).
    
    # Wait, the current program returns rank 65. Let me check what that means.
    # The score is 1.0, which means it's the same as the seed. The seed returns rank 65.
    # So I need to find a decomposition with rank < 65 to improve.
    
    # Let me construct a rank-40 decomposition using a simple block structure.
    # For each i (0..3), I can group all j and k terms.
    # For fixed i, the U factor has non-zero entries at positions i*4+j for j=0..3 (4 entries)
    # The V factor has entries at j*5+k for j=0..3, k=0..4 (20 entries)
    # The W factor has entries at k*4+i for k=0..4 (5 entries)
    
    # For fixed i, I need to decompose a 4x20x5 sub-tensor.
    # The non-zero entries are at (j, j*5+k, k*4+i) for j=0..3, k=0..4.
    # That's 20 non-zero entries.
    
    # I can use a rank-5 decomposition for each i by noting that:
    # For fixed i and k, the terms are (j, j*5+k, k*4+i) for j=0..3.
    # This is a rank-4 structure (one per j).
    # So for fixed i, I need 5*4 = 20 terms? No, that's not sharing.
    
    # Let me try yet another approach: 
    # Use the fact that the problem is equivalent to multiplying a 4x4 matrix by 5 separate 4x1 vectors.
    # I can use a rank-r decomposition for 4x4*4x1, and apply it 5 times.
    # The rank for 4x4*4x1 is 4 (schoolbook).
    # So total rank = 5*4 = 20.
    
    # Wait, is that valid? For 4x4*4x1:
    # A is 4x4, b is 4x1, c = A*b is 4x1
    # c_i = sum_j A_ij * b_j
    # T[i*4+j, j*1+0, 0*4+i] = T[i*4+j, j, i] = 1
    # U[i*4+j, col] = 1, V[j, col] = 1, W[i, col] = 1
    # Rank = 16 (one per (i,j))
    
    # That's rank 16 for 4x4*4x1, not 4. I was wrong.
    
    # For 4x4*4x1, the rank is 16 (schoolbook). There's no known way to do it with less than 16
    # for exact multiplication over a field.
    
    # So my earlier analysis was correct: the schoolbook rank for 4x4*4x5 is 80.
    
    # To beat rank 65, I need a non-trivial decomposition.
    
    # Let me try a practical approach: use random initialization and verify,
    # but with a better search strategy.
    
    # I'll try ranks from 24 to 64 and use a more efficient verification.
    
    # For a given rank r, I'll try to find U, V, W such that:
    # sum_c U[a,c] * V[b,c] * W[c_idx,c] = T[a,b,c_idx]
    
    # This is a system of equations. I'll use a random restart approach with
    # gradient descent or direct solving.
    
    # Actually, let me use a different structural insight:
    # The 4x4x5 tensor can be viewed as a 4x4x4 tensor (for k=0..3) plus a 4x4x1 tensor (for k=4).
    # If I have a rank-r decomposition for 4x4x4, I can add 16 terms for the 4x4x1 part.
    # Total rank = r + 16.
    # If r = 19, total = 35.
    # If r = 20, total = 36.
    # If r = 25, total = 41.
    
    # Let me try to construct a rank-25 decomposition for 4x4x4, which would give rank 41 for 4x4x5.
    # That's still better than 65.
    
    # For 4x4x4, I can use the following construction:
    # Split into 2x2x2 blocks and use Strassen for the block diagonal, plus cross terms.
    
    # 4x4x4 with 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22], [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Each 2x2*2x2=2x2 multiplication has rank 7 (Strassen).
    # There are 4 block multiplications, so 4*7 = 28 terms for the block diagonal.
    # But we also need cross terms? No, the block multiplication is exact.
    # So rank 28 for 4x4x4 using Strassen blocks.
    
    # Then for 4x4x5: 28 + 16 = 44.
    
    # That's better than 65! Let me construct this.
    
    # Strassen's 7 terms for 2x2x2:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Now I need to map this to the tensor decomposition.
    
    # For the full 4x4x4 tensor:
    # T[i*4+j, j*4+k, k*4+i] = 1
    
    # I'll construct U, V, W with 28 columns (one per Strassen term per block).
    
    # Block (11): A11*B11, where A11 is rows 0-1, cols 0-1 of A
    # In the tensor, this corresponds to i in {0,1}, j in {0,1}, k in {0,1}
    
    # Let me construct the 28 terms explicitly.
    
    rank_4x4x4 = 28
    rank_4x4x1 = 16
    total_rank = rank_4x4x4 + rank_4x4x1  # 44
    
    U = np.zeros((16, total_rank))
    V = np.zeros((20, total_rank))
    W = np.zeros((20, total_rank))
    
    col = 0
    # Strassen terms for each 2x2 block
    # Block indices: (bi, bj) where bi, bj in {0,1} for row/col block
    for bi in range(2):
        for bj in range(2):
            # A block (bi, bj): rows bi*2 to bi*2+1, cols bj*2 to bj*2+1
            # B block (bj, bk): rows bj*2 to bj*2+1, cols bk*2 to bk*2+1
            for bk in range(2):
                # C block (bi, bk): rows bi*2 to bi*2+1, cols bk*2 to bk*2+1
                # 7 Strassen terms for this block multiplication
                # A11, A12, A21, A22 within the 2x2 blocks
                # A11: (bi*2, bj*2), A12: (bi*2, bj*2+1)
                # A21: (bi*2+1, bj*2), A22: (bi*2+1, bj*2+1)
                # B11: (bj*2, bk*2), B12: (bj*2, bk*2+1)
                # B21: (bj*2+1, bk*2), B22: (bj*2+1, bk*2+1)
                
                # M1 = (A11 + A22) * (B11 + B22)
                U[bi*4 + bj*2 + 0, col] += 1  # A11: i=bi*2, j=bj*2
                U[bi*4 + bj*2 + 1, col] += 1  # A22: i=bi*2+1, j=bj*2+1
                V[bj*4 + bk*2 + 0, col] += 1  # B11: j=bj*2, k=bk*2
                V[bj*4 + bk*2 + 1, col] += 1  # B22: j=bj*2+1, k=bk*2+1
                W[bk*4 + bi*2 + 0, col] += 1  # C11: i=bi*2, k=bk*2
                W[bk*4 + bi*2 + 1, col] += 1  # C22: i=bi*2+1, k=bk*2+1
                col += 1
                
                # M2 = (A21 + A22) * B11
                U[bi*4 + bj*2 + 1, col] += 1  # A21: i=bi*2+1, j=bj*2
                U[bi*4 + bj*2 + 1, col] += 1  # A22: i=bi*2+1, j=bj*2+1
                V[bj*4 + bk*2 + 0, col] += 1  # B11: j=bj*2, k=bk*2
                W[bk*4 + bi*2 + 1, col] += 1  # C21: i=bi*2+1, k=bk*2
                col += 1
                
                # M3 = A11 * (B12 - B22)
                U[bi*4 + bj*2 + 0, col] += 1  # A11: i=bi*2, j=bj*2
                V[bj*4 + bk*2 + 0, col] += 1  # B12: j=bj*2, k=bk*2+1
                V[bj*4 + bk*2 + 1, col] += -1  # B22: j=bj*2+1, k=bk*2+1
                W[bk*4 + bi*2 + 0, col] += 1  # C12: i=bi*2, k=bk*2+1
                W[bk*4 + bi*2 + 1, col] += 1  # C22: i=bi*2+1, k=bk*2+1
                col += 1
                
                # M4 = A22 * (B21 - B11)
                U[bi*4 + bj*2 + 1, col] += 1  # A22: i=bi*2+1, j=bj*2+1
                V[bj*4 + bk*2 + 1, col] += 1  # B21: j=bj*2+1, k=bk*2
                V[bj*4 + bk*2 + 0, col] += -1  # B11: j=bj*2, k=bk*2
                W[bk*4 + bi*2 + 1, col] += 1  # C21: i=bi*2+1, k=bk*2
                W[bk*4 + bi*2 + 0, col] += 1  # C11: i=bi*2, k=bk*2
                col += 1
                
                # M5 = (A11 + A12) * B22
                U[bi*4 + bj*2 + 0, col] += 1  # A11: i=bi*2, j=bj*2
                U[bi*4 + bj*2 + 1, col] += 1  # A12: i=bi*2, j=bj*2+1
                V[bj*4 + bk*2 + 1, col] += 1  # B22: j=bj*2+1, k=bk*2+1
                W[bk*4 + bi*2 + 0, col] += 1  # C12: i=bi*2, k=bk*2+1
                W[bk*4 + bi*2 + 1, col] += 1  # C22: i=bi*2+1, k=bk*2+1
                col += 1
                
                # M6 = (A21 - A11) * (B11 + B12)
                U[bi*4 + bj*2 + 1, col] += 1  # A21: i=bi*2+1, j=bj*2
                U[bi*4 + bj*2 + 0, col] += -1  # A11: i=bi*2, j=bj*2
                V[bj*4 + bk*2 + 0, col] += 1  # B11: j=bj*2, k=bk*2
                V[bj*4 + bk*2 + 1, col] += 1  # B12: j=bj*2, k=bk*2+1
                W[bk*4 + bi*2 + 1, col] += 1  # C22: i=bi*2+1, k=bk*2+1
                W[bk*4 + bi*2 + 0, col] += 1  # C11: i=bi*2, k=bk*2
                col += 1
                
                # M7 = (A12 - A22) * (B21 + B22)
                U[bi*4 + bj*2 + 1, col] += 1  # A12: i=bi*2, j=bj*2+1
                U[bi*4 + bj*2 + 1, col] += -1  # A22: i=bi*2+1, j=bj*2+1
                V[bj*4 + bk*2 + 1, col] += 1  # B21: j=bj*2+1, k=bk*2
                V[bj*4 + bk*2 + 1, col] += 1  # B22: j=bj*2+1, k=bk*2+1
                W[bk*4 + bi*2 + 0, col] += 1  # C12: i=bi*2, k=bk*2+1
                W[bk*4 + bi*2 + 1, col] += 1  # C22: i=bi*2+1, k=bk*2+1
                col += 1
    
    # Now add the 16 terms for the 5th column (k=4)
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1  # k=4, so c = 4*4+i = 16+i
            col += 1
    
    # Verify the decomposition
    # For each valid (i,j,k) with k=0..3, check that the reconstruction gives 1
    # For k=4, check similarly
    valid = True
    for i in range(4):
        if not valid:
            break
        for j in range(4):
            if not valid:
                break
            for k in range(5):
                a = i*4 + j
                b = j*5 + k
                c = k*4 + i
                recon = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                if abs(recon - 1.0) > 1e-10:
                    valid = False
                    break
    
    # Also check that non-valid entries are 0 (spot check)
    if valid:
        for c_test in range(20):
            if not valid:
                break
            for a_test in range(16):
                if not valid:
                    break
                for b_test in range(20):
                    # Skip valid entries
                    i_test = a_test // 4
                    j_test = a_test % 4
                    k_test = b_test // 5
                    j_test2 = b_test % 5
                    c_i_test = c_test % 4
                    c_k_test = c_test // 4
                    if j_test == j_test2 and c_i_test == i_test and c_k_test == k_test:
                        continue  # This is a valid entry
                    recon_test = np.dot(U[a_test, :], np.multiply(V[b_test, :], W[c_test, :]))
                    if abs(recon_test) > 1e-10:
                        valid = False
                        break
    
    if valid:
        result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        # Round to half-integers
        U_out = [[round(v * 2) / 2 for v in row] for row in result["U"]]
        V_out = [[round(v * 2) / 2 for v in row] for row in result["V"]]
        W_out = [[round(v * 2) / 2 for v in row] for row in result["W"]]
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to schoolbook rank 80
        rank_school = 80
        U_school = np.zeros((16, rank_school))
        V_school = np.zeros((20, rank_school))
        W_school = np.zeros((20, rank_school))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_school[i*4+j, col] = 1
                    V_school[j*5+k, col] = 1
                    W_school[k*4+i, col] = 1
                    col += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END

