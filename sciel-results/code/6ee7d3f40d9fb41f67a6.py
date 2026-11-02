import time
import random
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    U: 4x7 (index i*2+j), V: 4x7 (index j*2+k), W: 4x7 (index k*2+i)
    Convention: C[k*n+i, r] = W[k*n+i, r]
    """
    n = m = p = 2
    # Strassen:
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
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # Row indices for U: i*2+j where i in {0,1}, j in {0,1}
    # So: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Row indices for V: j*2+k where j in {0,1}, k in {0,1}
    # So: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Row indices for W: k*2+i where k in {0,1}, i in {0,1}
    # So: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    # M1 = (A11 + A22) * (B11 + B22)
    # A11 = U[0,0]=1, A22 = U[3,0]=1
    # B11 = V[0,0]=1, B22 = V[3,0]=1
    # C11 gets +1, C22 gets +1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C11 (k=0,i=0)
    W[3, 0] = 1  # C22 (k=1,i=1)
    
    # M2 = (A21 + A22) * B11
    # A21 = U[2,1]=1, A22 = U[3,1]=1
    # B11 = V[0,1]=1
    # C21 gets +1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21 (k=1,i=0)
    
    # M3 = A11 * (B12 - B22)
    # A11 = U[0,2]=1
    # B12 = V[1,2]=1, B22 = V[3,2]=-1
    # C12 gets +1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12 (k=0,i=1)
    
    # M4 = A22 * (B21 - B11)
    # A22 = U[3,3]=1
    # B21 = V[2,3]=1, B11 = V[0,3]=-1
    # C11 gets +1, C21 gets +1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5 = (A11 + A12) * B22
    # A11 = U[0,4]=1, A12 = U[1,4]=1
    # B22 = V[3,4]=1
    # C11 gets -1, C12 gets +1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C11
    W[1, 4] = 1   # C12
    
    # M6 = (A21 - A11) * (B11 + B12)
    # A21 = U[2,5]=1, A11 = U[0,5]=-1
    # B11 = V[0,5]=1, B12 = V[1,5]=1
    # C22 gets +1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C22
    
    # M7 = (A12 - A22) * (B21 + B22)
    # A12 = U[1,6]=1, A22 = U[3,6]=-1
    # B21 = V[2,6]=1, B22 = V[3,6]=1
    # C11 gets +1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    
    return U, V, W


def _schoolbook_block(n, m, p):
    """Return (U, V, W) for n x m x p matrix multiplication, schoolbook rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _block_diag_concat(blocks_U, blocks_V, blocks_W, row_sizes, col_sizes):
    """Concatenate block-diagonal factors.
    blocks_U: list of (row_size_i, rank_i) matrices
    blocks_V: list of (col_size_i, rank_i) matrices  
    blocks_W: list of (out_size_i, rank_i) matrices
    Returns (U, V, W) with appropriate shapes.
    """
    total_rank = sum(b.shape[1] for b in blocks_U)
    U = np.zeros((sum(row_sizes), total_rank))
    V = np.zeros((sum(col_sizes), total_rank))
    W = np.zeros((sum(row_sizes), total_rank))  # W has same row structure as U here (p*n)
    
    # Actually W shape is (p*n, rank), and U is (n*m, rank)
    # Let me be more careful.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # For block diagonal: each block has its own U_i, V_i, W_i
    # U_i: (n_i * m_i, r_i), V_i: (m_i * p_i, r_i), W_i: (p_i * n_i, r_i)
    
    # Let me just use the shapes directly from the blocks
    total_rank = sum(b.shape[1] for b in blocks_U)
    U = np.zeros((blocks_U[0].shape[0] if len(blocks_U) > 0 else 0, total_rank))
    # Actually shapes may differ, let me just stack
    U = np.zeros((sum(b.shape[0] for b in blocks_U), total_rank))
    V = np.zeros((sum(b.shape[0] for b in blocks_V), total_rank))
    W = np.zeros((sum(b.shape[0] for b in blocks_W), total_rank))
    
    offset = 0
    for i in range(len(blocks_U)):
        r_i = blocks_U[i].shape[1]
        U[offset:offset + blocks_U[i].shape[0], offset:offset + r_i] = blocks_U[i]
        V[offset:offset + blocks_V[i].shape[0], offset:offset + r_i] = blocks_V[i]
        W[offset:offset + blocks_W[i].shape[0], offset:offset + r_i] = blocks_W[i]
        offset += r_i
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Strategy: Decompose 3x4x6 into blocks and use Strassen where possible.
    # 
    # Approach 1: Split the middle dimension (m=4) into 2+2.
    # C = A[:, 0:2] @ B[0:2, :] + A[:, 2:4] @ B[2:4, :]
    # This is 3x2x6 + 3x2x6 = two 3x2x6 multiplications.
    # Schoolbook: 2 * 3*2*6 = 72. No gain.
    # 
    # Approach 2: Split n=3 into 1+2, and use structure.
    # C[0,:] = A[0,:] @ B  (1x4x6)
    # C[1:,:] = A[1:,:] @ B (2x4x6)
    # Schoolbook: 1*4*6 + 2*4*6 = 24 + 48 = 72. No gain.
    #
    # Approach 3: Use the fact that 4 = 2+2 and apply Strassen to the 2x2x2 sub-blocks.
    # For 3x4x6, split into:
    #   n=3, m=4=2+2, p=6
    # C[i, k] = sum_{j=0}^{3} A[i,j] B[j,k]
    #          = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    # 
    # For the first part (j=0,1): this is 3x2x6.
    # For the second part (j=2,3): this is also 3x2x6.
    # 
    # Now, 3x2x6: can we do better than schoolbook 36?
    # Split p=6 into 2+2+2:
    #   3x2x2 + 3x2x2 + 3x2x2
    # Each 3x2x2: schoolbook rank 12.
    # But 3x2x2 can be seen as: for each i in 0..2, k in 0..1:
    #   C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is like 3 copies of 1x2x2 multiplication.
    # 1x2x2: C[k] = A0*B[0,k] + A1*B[1,k], k=0,1
    # This has tensor T[i=0, j, k] with i fixed. Rank of 1x2x2 is 2 (just two terms).
    # Actually for 1x2x2: T[j,k] = 1 for all j,k in {0,1}. That's a 2x2 matrix, rank 2.
    # So 3x2x2 has rank 3*2 = 6? No...
    # 
    # Let me think differently. The tensor for n x m x p matrix mult has entries T[i,j,k]=1.
    # For 3x2x2: T[i,j,k] = 1 for i in {0,1,2}, j in {0,1}, k in {0,1}.
    # This is a 3x2x2 tensor. Its rank: we need U (3*2=6 x r), V (2*2=4 x r), W (2*3=6 x r).
    # T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*3+i, c]
    # 
    # Schoolbook: r = 3*2*2 = 12.
    # Can we do better? The tensor is a sum of 3 rank-1 slices (one per i).
    # For fixed i, T[i,:,:] is a 2x2 all-ones matrix, which has rank 2.
    # So the total tensor rank is at most 3*2 = 6.
    # And it's at least 3 (since there are 3 non-zero slices that are linearly independent in some sense).
    # Actually, let's construct rank 6:
    # For each i in {0,1,2} and each (j,k) in {(0,0),(0,1),(1,0),(1,1)}... no that's 12.
    # 
    # Wait, for fixed i, the 2x2 tensor T[i,j,k]=1 can be decomposed as:
    # T[i,j,k] = sum_{c=1}^{2} U[i,j,c] V[j,k,c] W[k,i,c]
    # But W depends on i, so we need to be careful.
    # 
    # Let me just use schoolbook for 3x2x2: rank 12.
    # Then 3x2x6 = 3 * (3x2x2) has rank 3*12 = 36. Same as schoolbook.
    #
    # Approach 4: More carefully combine Strassen.
    # Split 3x4x6 as: n=3, m=4, p=6.
    # Split m into 2+2: two 3x2x6 blocks.
    # Split each 3x2x6 into: n=3, m=2, p=6.
    # Split p into 2+2+2: three 3x2x2 blocks per 3x2x6.
    # Total: 2 * 3 = 6 blocks of 3x2x2.
    # Schoolbook rank of 3x2x2 is 12. Total: 6*12 = 72. No gain.
    #
    # Approach 5: Use Strassen on 2x2x2 sub-blocks within the 3x2x2.
    # 3x2x2: For i=0,1,2 and j,k in {0,1}.
    # This is NOT a 2x2x2 problem because n=3.
    # But we can split n=3 into 1+2.
    # 1x2x2 + 2x2x2.
    # 1x2x2: rank 2 (as discussed).
    # 2x2x2: rank 7 (Strassen).
    # Total: 2 + 7 = 9 < 12. Gain of 3!
    # 
    # So 3x2x2 can be done in rank 9.
    # Then 3x2x6 = 3 * (3x2x2) = 3 * 9 = 27.
    # And 3x4x6 = 2 * (3x2x6) = 2 * 27 = 54.
    # 
    # Wait, let me verify the 3x2x2 rank 9 construction.
    # 3x2x2: T[i,j,k] = 1 for i in {0,1,2}, j in {0,1}, k in {0,1}.
    # Split i: i=0 (1x2x2) and i=1,2 (2x2x2).
    # 
    # 1x2x2: T[0,j,k] = 1.
    # U (1*2=2 x r), V (2*2=4 x r), W (2*1=2 x r).
    # T[0,j,k] = sum_c U[j, c] V[j*2+k, c] W[k*1+0, c] = sum_c U[j,c] V[j*2+k,c] W[k,c]
    # We need this to equal 1 for all j,k in {0,1}.
    # Rank 2: 
    #   c=0: U[0,0]=1, U[1,0]=0; V[0,0]=1, V[1,0]=0, V[2,0]=0, V[3,0]=0; W[0,0]=1, W[1,0]=0
    #     This gives T[0,0,0] = 1*1*1 = 1. Others 0.
    #   c=1: U[0,1]=0, U[1,1]=1; V[0,1]=0, V[1,1]=0, V[2,1]=1, V[3,1]=0; W[0,1]=0, W[1,1]=1
    #     This gives T[0,1,1] = 1*1*1 = 1. Others 0.
    #   But we need T[0,0,1]=1 and T[0,1,0]=1 too.
    # 
    # Let me redo. For 1x2x2:
    # T[0,j,k] = 1 for all j,k.
    # We need: sum_c U[j,c] V[j*2+k,c] W[k,c] = 1 for all j,k in {0,1}.
    # 
    # Try rank 2:
    # c=0: U[0,0]=1, U[1,0]=1; V[0,0]=1, V[1,0]=1, V[2,0]=0, V[3,0]=0; W[0,0]=1, W[1,0]=0
    #   T[0,0,0] = U[0,0]*V[0,0]*W[0,0] + U[1,0]*V[2,0]*W[1,0] = 1*1*1 + 1*0*0 = 1. OK.
    #   T[0,0,1] = U[0,0]*V[1,0]*W[1,0] + U[1,0]*V[3,0]*W[1,0] = 1*1*0 + 1*0*0 = 0. Need 1. FAIL.
    # 
    # This is getting complicated. Let me just use schoolbook for the sub-blocks
    # and see if I can find a better global structure.
    #
    # Approach 6: Direct construction using the known result that 3x4x6 can be done
    # in rank less than 72.
    #
    # Let me try: split 3x4x6 into 4 blocks of 3x1x6 (splitting m=4 into four 1s).
    # 3x1x6: T[i,0,k] = 1 for i in {0,1,2}, k in {0..5}.
    # This is a 3x1x6 tensor. Rank: for each (i,k), it's a single term.
    # Schoolbook: 3*1*6 = 18.
    # Total: 4*18 = 72. No gain.
    #
    # Approach 7: Share structure between the m=1 blocks.
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k].
    # If we could compute S[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    # using fewer than 24 multiplications...
    #
    # For fixed i,k, this is a dot product of two 4-vectors. No obvious sharing.
    #
    # Let me try a different split: n=3, m=4, p=6.
    # Split p=6 into 3+3:
    #   C[:, 0:3] = A @ B[:, 0:3]  (3x4x3)
    #   C[:, 3:6] = A @ B[:, 3:6]  (3x4x3)
    # Schoolbook: 2 * 3*4*3 = 72. No gain.
    #
    # I think the key insight from the inspiration is that rank 68 was achieved.
    # 72 - 68 = 4 savings. This suggests saving 1 per some block.
    #
    # Let me try: 3x4x6 = (3x4x2) + (3x4x2) + (3x4x2) (splitting p=6 into three 2s).
    # Each 3x4x2: schoolbook rank 24. Total 72.
    # But 3x4x2: split m=4 into 2+2:
    #   3x2x2 + 3x2x2.
    # Each 3x2x2: schoolbook 12.
    # But can 3x2x2 be done in rank < 12?
    # 3x2x2: T[i,j,k] = 1 for i in {0,1,2}, j in {0,1}, k in {0,1}.
    # 
    # Let me try to find the rank of the 3x2x2 all-ones tensor.
    # The tensor is 3x2x2. We can flatten it as a 3x4 matrix (i, (j,k)) or 6x2, etc.
    # As a 3x4 matrix (rows i, cols (j,k)): all entries are 1. Rank 1.
    # As a 6x2 matrix (rows (i,j), cols k): all entries are 1. Rank 1.
    # As a 2x6 matrix (rows j, cols (i,k)): all entries are 1. Rank 1.
    # 
    # The tensor rank is at least the max of the matrix ranks = 1.
    # And it's at most... well, schoolbook is 12.
    # 
    # Can we do rank 3? 
    # T[i,j,k] = sum_{c=1}^{3} U[i*2+j, c] V[j*2+k, c] W[k*3+i, c]
    # 
    # Let me try rank 4:
    # For each (j,k) pair, we need sum_c U[i*2+j,c] W[k*3+i,c] to be proportional to V[j*2+k,c]...
    # This is hard to reason about abstractly.
    #
    # Let me just implement the schoolbook and see if I can find a simple improvement.
    # Given the time constraints, I'll return schoolbook rank 72.
    
    U, V, W = _schoolbook_block(n, m, p)
    
    # Convert to lists
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
