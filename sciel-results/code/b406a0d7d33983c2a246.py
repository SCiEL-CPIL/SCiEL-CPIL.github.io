"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def _verify_exact(U, V, W, n=3, m=4, p=8):
    """
    Verify the decomposition exactly using rational arithmetic.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    # Check each (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    total += U_frac[i * m + j][c] * V_frac[j * p + k][c] * W_frac[k * n + i][c]
                if total != 1:
                    return False
    return True

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using vectorized operations (fast, approximate).
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

def _construct_rank_72():
    """
    Construct a rank-72 decomposition.
    Split A (3x4) into A0 (1x4) and A12 (2x4).
    A0 x B (4x8): Schoolbook, rank 32.
    A12 x B (4x8): For each of 4 column-blocks of B (size 4x2),
                   C_block = A12(2x4) * B_block(4x2).
                   Split into: A12 = [A_L(2x2) | A_R(2x2)], B_block = [B_T(2x2); B_B(2x2)]
                   C_block = A_L * B_T + A_R * B_B
                   Use rank-12 algorithm for 2x2x4 (two 2x2x2 Strassen combined).
                   Actually, we can do better: for 2x4x2, use a direct rank-10 or rank-12 method.
                   
                   Simpler: use rank-14 for each 2x4x2 (two Strassen 7+7), total 4*14=56, grand total 88.
                   OR: use a single rank-12 for 2x2x4 per block pair.
                   
                   Let's try: for each k_block (4 blocks), we have:
                   C_11 = A_L * B_T + A_R * B_B  (2x2 result for row 1)
                   C_21 = A_L * B_T + A_R * B_B  (same for row 2? No...)
                   
                   Wait, C is 2x2 for each k_block:
                   C[0,0] = sum_j A[0,j]*B_block[j,0]
                   C[0,1] = sum_j A[0,j]*B_block[j,1]
                   C[1,0] = sum_j A[1,j]*B_block[j,0]
                   C[1,1] = sum_j A[1,j]*B_block[j,1]
                   
                   With A = [A_L | A_R] (2x4), B_block = [B_T; B_B] (4x2):
                   C = A_L * B_T + A_R * B_B
                   
                   This is a 2x2x4 tensor (A is 2x2 for each of two B-parts, B is 2x2 for each of two A-parts).
                   Actually it's: C(2x2) = A1(2x2)*B1(2x2) + A2(2x2)*B2(2x2)
                   where A1=A_L, B1=B_T, A2=A_R, B2=B_B.
                   
                   This is the sum of two 2x2x2 matrix multiplications.
                   Rank of sum of two 2x2x2: at most 14 (two Strassen).
                   
                   But we can do better! The tensor for "sum of two 2x2x2" is actually
                   the 2x2x4 matrix multiplication tensor, which has rank 14.
                   
                   So 4 blocks * 14 = 56, total 32+56 = 88. Same as before.
                   
                   Let me try a different decomposition. What if we use rank-10 for 2x4x2?
                   R(2x4x2) = R(2x2x4) = 14 is the known value.
                   
                   Hmm, so we're stuck at 88 with this approach.
                   
                   Let me try yet another approach: don't split B into blocks.
                   Instead, treat the full 2x4x8 problem.
                   R(2x4x8) <= R(2x4x4) + R(2x4x4) = 2*R(2x4x4)
                   R(2x4x4) <= R(2x4x2)*2 ... circular.
                   
                   Actually R(2x4x4) = R(4x4x2) by symmetry. And R(4x4x2) <= 2*R(2x4x2) = 2*14 = 28?
                   No, R(4x4x2) is the rank of 4x4x2 which is at most 4*4*2=32 schoolbook, 
                   but can be done with Strassen: split 4x4 into 2x2 blocks, 4 subproblems of 2x2x2,
                   each rank 7, total 28.
                   
                   So R(2x4x8) <= 2*28 = 56. Total 32+56=88. Still 88.
                   
                   What about R(2x4x8) directly? 
                   Split 2x4x8 as: A(2x4) * B(4x8).
                   Use Strassen on the 4x4 part: split A into A_L(2x2) and A_R(2x2) [by columns],
                   split B into B_1(4x4) and B_2(4x4) [by columns].
                   C_1 = A * B_1 (2x4), C_2 = A * B_2 (2x4).
                   
                   For C_1 = A(2x4)*B_1(4x4):
                   Split B_1 into B_T(2x4) and B_B(2x4) [by rows].
                   C_1 = A_L(2x2)*B_T(2x4) + A_R(2x2)*B_B(2x4).
                   
                   A_L(2x2)*B_T(2x4): this is 2x2x4, rank 14.
                   A_R(2x2)*B_B(2x4): this is 2x2x4, rank 14.
                   But we can share Strassen products!
                   
                   Actually, let's just use the full Strassen decomposition:
                   2x4x4 = 2x(2x2x4) where each 2x2x4 uses rank 14.
                   Total for 2x4x4: 2*14 = 28.
                   Total for 2x4x8: 2*28 = 56.
                   Grand total: 32 + 56 = 88.
                   
                   I'm going in circles. Let me just keep rank 88 and focus on verification.
    """
    return None

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
    Placeholder for rank-84 decomposition attempt.
    R(2x2x4) = 14 (known), so 4 blocks * 14 = 56, total 32+56=88.
    Cannot achieve rank 84 with this block decomposition approach.
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
                # Confirm with exact verification
                if _verify_exact(U_88.tolist(), V_88.tolist(), W_88.tolist()):
                    if U_88.shape[1] < best_rank:
                        best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                        best_rank = U_88.shape[1]
        except Exception:
            pass

    # Try Rank 72 (if implemented)
    if time.time() < deadline:
        try:
            res = _construct_rank_72()
            if res:
                U_72, V_72, W_72 = res
                if _verify_safe(U_72, V_72, W_72):
                    if _verify_exact(U_72.tolist(), V_72.tolist(), W_72.tolist()):
                        if U_72.shape[1] < best_rank:
                            best_result = {"U": U_72.tolist(), "V": V_72.tolist(), "W": W_72.tolist()}
                            best_rank = U_72.shape[1]
        except Exception:
            pass

    # Try Rank 84 (if implemented)
    if time.time() < deadline:
        try:
            res = _try_rank_84()
            if res:
                U_84, V_84, W_84 = res
                if _verify_safe(U_84, V_84, W_84):
                    if _verify_exact(U_84.tolist(), V_84.tolist(), W_84.tolist()):
                        if U_84.shape[1] < best_rank:
                            best_result = {"U": U_84.tolist(), "V": V_84.tolist(), "W": W_84.tolist()}
                            best_rank = U_84.shape[1]
        except Exception:
            pass

    return best_result
