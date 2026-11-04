"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using vectorized operations.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    # Reshape for vectorized computation
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_88():
    """
    Construct a rank-88 decomposition.
    Split A (3x4) into A0 (1x4) and A12 (2x4).
    A0 x B (4x8): Schoolbook, rank 32.
    A12 x B (4x8): Split A12 into two 2x2 blocks (left/right).
                   Split B into four 2x2 blocks (top/bottom, left/right).
                   8 sub-problems of 2x2x2, each Strassen (rank 7).
                   Total rank 56.
    Grand total: 32 + 56 = 88.
    """
    n, m, p = 3, 4, 8
    rank = 88
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Part 1: i=0, schoolbook 1x4x8
    for j in range(4):
        for k in range(8):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
            
    # Part 2: i=1,2 using Strassen on 2x2 blocks
    # A rows 1-2, split cols into [0,1] and [2,3]
    # B split into 4 blocks of 2x2: cols [0,1], [2,3], [4,5], [6,7]
    
    strassen_U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]).T # shape (4, 7)
    
    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]).T # shape (4, 7)
    
    strassen_W = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]).T # shape (4, 7)
    
    # A blocks: 
    # Block 0: rows 1,2; cols 0,1. Indices in U: (1*4+0)..(1*4+1), (2*4+0)..(2*4+1)
    # Block 1: rows 1,2; cols 2,3. Indices in U: (1*4+2)..(1*4+3), (2*4+2)..(2*4+3)
    
    # B blocks:
    # Block 0: rows 0,1; cols 0,1.
    # Block 1: rows 0,1; cols 2,3.
    # Block 2: rows 2,3; cols 0,1.
    # Block 3: rows 2,3; cols 2,3.
    
    for k_block in range(4):
        k_start = k_block * 2
        
        # Sub-problem 1: A_L (cols 0-1) * B_TL (rows 0-1, cols k_start..k_start+1)
        for t in range(7):
            # U
            U[4, col] += strassen_U[0, t]
            U[5, col] += strassen_U[1, t]
            U[8, col] += strassen_U[2, t]
            U[9, col] += strassen_U[3, t]
            
            # V
            V[k_start, col] += strassen_V[0, t]
            V[k_start+1, col] += strassen_V[1, t]
            V[p+k_start, col] += strassen_V[2, t]
            V[p+k_start+1, col] += strassen_V[3, t]
            
            # W
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1
            
        # Sub-problem 2: A_R (cols 2-3) * B_BL (rows 2-3, cols k_start..k_start+1)
        for t in range(7):
            # U
            U[6, col] += strassen_U[0, t]
            U[7, col] += strassen_U[1, t]
            U[10, col] += strassen_U[2, t]
            U[11, col] += strassen_U[3, t]
            
            # V
            V[2*p+k_start, col] += strassen_V[0, t]
            V[2*p+k_start+1, col] += strassen_V[1, t]
            V[3*p+k_start, col] += strassen_V[2, t]
            V[3*p+k_start+1, col] += strassen_V[3, t]
            
            # W
            W[k_start*n + 1, col] += strassen_W[0, t]
            W[(k_start+1)*n + 1, col] += strassen_W[1, t]
            W[k_start*n + 2, col] += strassen_W[2, t]
            W[(k_start+1)*n + 2, col] += strassen_W[3, t]
            
            col += 1

    return U, V, W

def _try_rank_84():
    """
    Construct a rank-84 decomposition using a rank-6 algorithm for 2x2x4 sub-problems.
    
    Strategy:
    1. Handle i=0 (1x4x8) with rank 32 (Schoolbook).
    2. Handle i=1,2 (2x4x8) with rank 52.
       - For each of the 4 B-column-blocks (k=0,1,2,3), we compute A(2x4) * B_block(4x2).
       - This is a 2x4x2 problem. We can decompose it as:
         C = [A_L * B_top + A_R * B_bottom] where A_L,A_R are 2x2, B_top,B_bottom are 2x2.
       - Actually, let's think differently. For fixed k_block, B_block is 4x2.
         C(2x2) = A(2x4) * B_block(4x2)
                = A_L(2x2) * B_top(2x2) + A_R(2x2) * B_bot(2x2)
       - We need rank for 2x4x2. Known: R(2x4x2) = 6 (Baur-Strassen or similar).
       - 4 blocks * 6 = 24 ranks for the 2x4x8 part? No wait.
       
       Actually, let me reconsider. The 2x4x8 problem:
       C(2x8) = A(2x4) * B(4x8)
       Split B into 4 blocks of 4x2: B_0, B_1, B_2, B_3
       C = [C_0, C_1, C_2, C_3] where C_j = A * B_j (2x2)
       
       For each j, C_j = A(2x4) * B_j(4x2). This is a 2x4x2 tensor.
       R(2x4x2) = 6 (this is known; it's the same as 2x2x4 by symmetry).
       
       So total for 2x4x8 = 4 * 6 = 24? That seems too good.
       
       Wait, R(2x4x2): The tensor is sum over i,j,k of e_i (in R^2) ⊗ e_j (in R^4) ⊗ e_k (in R^2).
       Actually, by symmetry R(2x4x2) = R(2x2x4). And R(2x2x4) >= R(2x2x2) = 7? No.
       
       Let me just implement a known rank-6 algorithm for 2x2x4 and verify.
       
       For 2x2x4: C(2x2) = A(2x2) * B(2x4)
       We can write:
       C_00 = A_00*B_00 + A_01*B_10
       C_01 = A_00*B_01 + A_01*B_11
       C_10 = A_10*B_00 + A_11*B_10
       C_11 = A_10*B_01 + A_11*B_11
       
       Rank-6 algorithm (from literature):
       Use 6 products:
       P1 = (A_00+A_11) * (B_00+B_01)
       P2 = (A_10) * (B_00)
       P3 = (A_01) * (B_11)
       P4 = (A_00+A_11+A_01) * (B_00)
       P5 = (A_10+A_00) * (B_01)
       P6 = (A_11) * (B_00+B_01)
       
       Hmm, this is getting complicated. Let me just try to find a rank-6 decomposition numerically or use a known one.
       
       Actually, I recall that R(2x2x4) = 14, not 6. Let me reconsider.
       
       The tensor for 2x2x4 has 2*2*4 = 16 entries. The border rank is at least... 
       
       Let me just keep the rank-88 solution and not attempt rank-84 since I'm not confident about the rank-6 claim.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try Rank 88
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88, V_88, W_88):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass

    # Try Rank 84 (if implemented)
    if time.time() < deadline:
        try:
            res = _try_rank_84()
            if res:
                U_84, V_84, W_84 = res
                if _verify_safe(U_84, V_84, W_84):
                    if U_84.shape[1] < best_rank:
                        best_result = {"U": U_84.tolist(), "V": V_84.tolist(), "W": W_84.tolist()}
                        best_rank = U_84.shape[1]
        except Exception:
            pass

    return best_result
