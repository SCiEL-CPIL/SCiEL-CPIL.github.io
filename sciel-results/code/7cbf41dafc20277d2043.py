"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Build Strassen 2x2x2 decomposition
    # U indexed by (i,k) -> i*2+k, V by (k,j) -> k*2+j, W by (j,i) -> j*2+i
    # T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify Strassen 2x2x2 correctness
    # T[i,k,j] = sum_r U2[i*2+k, r] * V2[k*2+j, r] * W2[j*2+i, r]
    # Should equal A[i,k]*B[k,j]
    for i in range(2):
        for k in range(2):
            for j in range(2):
                val = sum(U2[i*2+k, r] * V2[k*2+j, r] * W2[j*2+i, r] for r in range(7))
                if val != 1:
                    # Fallback to schoolbook if Strassen verification fails
                    rank = n * m * p
                    U = np.zeros((n * m, rank), dtype=int)
                    V = np.zeros((m * p, rank), dtype=int)
                    W = np.zeros((p * n, rank), dtype=int)
                    col = 0
                    for ii in range(n):
                        for kk in range(m):
                            for jj in range(p):
                                U[ii * m + kk, col] = 1
                                V[kk * p + jj, col] = 1
                                W[jj * n + ii, col] = 1
                                col += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Define the 7 outer Strassen products for 4x4x4
    # Each product: (A_blocks, B_blocks, C_blocks)
    # Block (bi, bk) of A: rows 2*bi to 2*bi+1, cols 2*bk to 2*bk+1
    # Block (bk, bj) of B: rows 2*bk to 2*bk+1, cols 2*bj to 2*bj+1
    # Block (bj, bi) of C: rows 2*bj to 2*bj+1, cols 2*bi to 2*bi+1
    
    big_mults = [
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((0,1), 1), ((1,1), -1)]},
        # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((1,0), 1), ((1,1), 1)]},
        # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((0,1), 1)]},
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((1,0), 1)]},
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]
    
    # Build rank-49 decomposition for 4x4x4
    # U: (16, 49), V: (16, 49), W: (16, 49)
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for big_idx, bm in enumerate(big_mults):
        for inner_idx in range(7):
            col = big_idx * 7 + inner_idx
            
            # For each inner Strassen term, determine which flat indices are involved
            # U2[ai, inner_idx] gives coefficient for A element at local position ai
            # ai = i_local * 2 + k_local where i_local, k_local in {0,1}
            # This corresponds to block positions that contribute to this inner term
            
            # We need to figure out which blocks of the 4x4 matrix map to each local 2x2 position
            # The virtual 2x2 A matrix has elements at positions (i2, k2) for i2,k2 in {0,1}
            # These come from linear combinations of the 4x4 blocks specified in bm['A']
            
            # Actually, the structure is:
            # For each big multiplication, the "virtual" 2x2 A matrix is formed by:
            # VA[i2, k2] = sum over blocks (bi, bk) in bm['A'] of sign * A_block(bi,bk)[i2, k2]
            # Similarly for B and C.
            
            # So for inner term inner_idx, U2[ai, inner_idx] tells us the coefficient
            # for VA element at flat position ai = i2*2 + k2
            # This VA element gets contributions from all blocks in bm['A']
            
            # Let's compute the contributions:
            for ai in range(4):
                coeff_a = U2[ai, inner_idx]
                if coeff_a == 0:
                    continue
                i2 = ai // 2
                k2 = ai % 2
                # VA[i2, k2] gets contribution from each block in bm['A']
                for (bi, bk), sign in bm['A']:
                    # Block (bi, bk) of A: flat index = bi*4 + bk*2 + i2*2 + k2
                    # Wait, block (bi,bk) occupies rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
                    # Element at local (i2, k2) within this block is at global (2*bi+i2, 2*bk+k2)
                    # Flat index in 4x4: (2*bi+i2)*4 + (2*bk+k2)
                    flat_a = (2*bi + i2) * 4 + (2*bk + k2)
                    U4[flat_a, col] += coeff_a * sign
            
            for bi_flat in range(4):
                coeff_b = V2[bi_flat, inner_idx]
                if coeff_b == 0:
                    continue
                k2 = bi_flat // 2
                j2 = bi_flat % 2
                for (bk, bj), sign in bm['B']:
                    # Block (bk, bj) of B: element at local (k2, j2) is at global (2*bk+k2, 2*bj+j2)
                    flat_b = (2*bk + k2) * 4 + (2*bj + j2)
                    V4[flat_b, col] += coeff_b * sign
            
            for wi_flat in range(4):
                coeff_c = W2[wi_flat, inner_idx]
                if coeff_c == 0:
                    continue
                j2 = wi_flat // 2
                i2 = wi_flat % 2
                for (bj, bi), sign in bm['C']:
                    # Block (bj, bi) of C: element at local (j2, i2) is at global (2*bj+j2, 2*bi+i2)
                    flat_c = (2*bj + j2) * 4 + (2*bi + i2)
                    W4[flat_c, col] += coeff_c * sign
    
    # Verify 4x4x4 decomposition
    # T[i,k,j] = sum_r U4[i*4+k, r] * V4[k*4+j, r] * W4[j*4+i, r] should be 1 if i==k==j... no
    # Actually T[i,k,j] should be 1 if the product A[i,k]*B[k,j] contributes to C[j,i]
    # For standard matrix mult: C[i,j] = sum_k A[i,k]*B[k,j]
    # So T[i,k,j] = 1 for all i,k,j (it's the identity tensor for matmul)
    # Wait no: the tensor is T[i,k,j] = delta_{i,k} * delta_{k,j}? No.
    # The matmul tensor: C[i,j] = sum_k A[i,k] B[k,j]
    # So the 3-tensor has T[i,k,j] = 1 for all i,k,j in the appropriate range
    # Actually: T[i,k,j] = 1 always (it's the full matmul tensor)
    
    # Let me verify: sum_r U4[i*4+k, r] * V4[k*4+j, r] * W4[j*4+i, r] should equal 1
    valid = True
    for i in range(4):
        for k in range(4):
            for j in range(4):
                val = sum(U4[i*4+k, r] * V4[k*4+j, r] * W4[j*4+i, r] for r in range(49))
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for ii in range(n):
            for kk in range(m):
                for jj in range(p):
                    U[ii * m + kk, col] = 1
                    V[kk * p + jj, col] = 1
                    W[jj * n + ii, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now build the full 4x4x9 decomposition
    # p = 9 = 4 + 4 + 1
    # First 4 columns: use U4, V4 (first 4 cols of B), W4
    # Next 4 columns: use U4, V4 (cols 4-7 of B), W4
    # Last 1 column: schoolbook 4x4x1, rank 16
    # Total rank: 49 + 49 + 16 = 114
    
    total_rank = 114
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((36, total_rank), dtype=int)  # m*p = 4*9 = 36
    W = np.zeros((36, total_rank), dtype=int)  # p*n = 9*4 = 36
    
    # First 4 columns of B (columns 0-3)
    U[:, 0:49] = U4
    V[:, 0:49] = V4  # V4 is (16, 49), maps to B elements for columns 0-3
    W[:, 0:49] = W4  # W4 is (16, 49), maps to C elements for rows 0-3
    
    # Next 4 columns of B (columns 4-7)
    U[:, 49:98] = U4
    V[4*4:8*4, 49:98] = V4  # B elements for rows 0-3, columns 4-7
    W[4*4:8*4, 49:98] = W4  # C elements for rows 4-7, all columns
    
    # Last 1 column of B (column 8): schoolbook 4x4x1
    # C[i, 8] = sum_k A[i,k] * B[k, 8]
    # Rank = 4*4 = 16
    col = 98
    for i in range(4):
        for k in range(4):
            U[i*4 + k, col] = 1
            V[k*4 + 8, col] = 1  # B[k, 8]: row k, column 8 -> flat index k*4+8
            W[8*4 + i, col] = 1  # C[8, i]: row 8, column i -> flat index 8*4+i
            col += 1
    
    # Final verification
    # Check that the decomposition is correct
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # In tensor form: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1 for all i,k,j
    
    # Spot check a few entries
    check_valid = True
    for i in range(4):
        for k in range(4):
            for j in range(9):
                val = sum(U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r] for r in range(total_rank))
                if val != 1:
                    check_valid = False
                    break
            if not check_valid:
                break
        if not check_valid:
            break
    
    if not check_valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for ii in range(n):
            for kk in range(m):
                for jj in range(p):
                    U[ii * m + kk, col] = 1
                    V[kk * p + jj, col] = 1
                    W[jj * n + ii, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
