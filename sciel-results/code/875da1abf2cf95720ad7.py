"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast tensor-based verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U_r = U.T.reshape(rank, n, m)
    V_r = V.T.reshape(rank, m, p)
    W_r = W.T.reshape(rank, p, n)
    T = np.einsum('rab,rcd,rki->abcdik', U_r, V_r, W_r)
    target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for k in range(p):
                target[a, b, b, k, a, k] = 1.0
    return np.max(np.abs(T - target)) < tol

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _build_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, U, V, W, col_offset):
    """
    Adds a Strassen decomposition for a 2x2 block multiplication.
    a_rows, a_cols: indices in A (2x2 block)
    b_rows, b_cols: indices in B (2x2 block)
    c_rows, c_cols: indices in C (2x2 block)
    Returns new col_offset.
    """
    # Strassen terms for 2x2x2
    # U: [A11, A12, A21, A22]
    # V: [B11, B12, B21, B22]
    # W: [C11, C12, C21, C22]
    
    # M1 = (A11+A22)(B11+B22) -> C11+=1, C22+=1
    # M2 = (A21+A22)B11 -> C21+=1, C22-=1
    # M3 = A11(B12-B22) -> C12+=1, C22+=1
    # M4 = A22(B21-B11) -> C11+=1, C21+=1
    # M5 = (A11+A12)B22 -> C11-=1, C12+=1
    # M6 = (A21-A11)(B11+B12) -> C22+=1
    # M7 = (A12-A22)(B21+B22) -> C11+=1
    
    terms = [
        # (u_coeffs, v_coeffs, w_coeffs)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Local indices
    # A: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # B: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # C: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    
    a_idx_map = {
        (0,0): a_rows[0]*len(a_cols) + a_cols[0],
        (0,1): a_rows[0]*len(a_cols) + a_cols[1],
        (1,0): a_rows[1]*len(a_cols) + a_cols[0],
        (1,1): a_rows[1]*len(a_cols) + a_cols[1],
    }
    b_idx_map = {
        (0,0): b_rows[0]*len(b_cols) + b_cols[0],
        (0,1): b_rows[0]*len(b_cols) + b_cols[1],
        (1,0): b_rows[1]*len(b_cols) + b_cols[0],
        (1,1): b_rows[1]*len(b_cols) + b_cols[1],
    }
    c_idx_map = {
        (0,0): c_rows[0]*len(c_cols) + c_cols[0],
        (0,1): c_rows[0]*len(c_cols) + c_cols[1],
        (1,0): c_rows[1]*len(c_cols) + c_cols[0],
        (1,1): c_rows[1]*len(c_cols) + c_cols[1],
    }
    
    # Map local indices to global indices
    # A is n x m, so index is i*m + j.
    # Here a_rows are row indices, a_cols are col indices.
    # Global A index: i*m + j
    # Global B index: j*p + k
    # Global C index: k*n + i
    
    # Wait, the input to construct() is U, V, W matrices.
    # U shape (n*m, rank). Row index corresponds to A[i,j] -> i*m+j.
    # V shape (m*p, rank). Row index corresponds to B[j,k] -> j*p+k.
    # W shape (p*n, rank). Row index corresponds to C[k,i] -> k*n+i.
    # Note: The task says "output factor follows the official trace convention W[k*n+i,r]".
    # So W row index is k*n + i.
    
    # Let's re-verify the index maps.
    # A[i,j] -> U[i*m+j]
    # B[j,k] -> V[j*p+k]
    # C[i,k] -> W[k*n+i]
    
    # Local A (2x2) at rows a_rows, cols a_cols:
    # A(a_rows[0], a_cols[0]) -> U[a_rows[0]*m + a_cols[0]]
    # A(a_rows[0], a_cols[1]) -> U[a_rows[0]*m + a_cols[1]]
    # A(a_rows[1], a_cols[0]) -> U[a_rows[1]*m + a_cols[0]]
    # A(a_rows[1], a_cols[1]) -> U[a_rows[1]*m + a_cols[1]]
    
    # Local B (2x2) at rows b_rows, cols b_cols:
    # B(b_rows[0], b_cols[0]) -> V[b_rows[0]*p + b_cols[0]]
    # ...
    
    # Local C (2x2) at rows c_rows, cols c_cols:
    # C(c_rows[0], c_cols[0]) -> W[c_cols[0]*n + c_rows[0]]
    # ...
    
    m, p, n = DIMENSIONS[1], DIMENSIONS[2], DIMENSIONS[0]
    
    for term in terms:
        u_coeffs, v_coeffs, w_coeffs = term
        col = col_offset
        
        # U
        for l in range(4):
            if u_coeffs[l] != 0:
                # Map l to local (r, c)
                lr, lc = l // 2, l % 2
                global_idx = a_rows[lr] * m + a_cols[lc]
                U[global_idx][col] += u_coeffs[l]
        
        # V
        for l in range(4):
            if v_coeffs[l] != 0:
                lr, lc = l // 2, l % 2
                global_idx = b_rows[lr] * p + b_cols[lc]
                V[global_idx][col] += v_coeffs[l]
        
        # W
        for l in range(4):
            if w_coeffs[l] != 0:
                lr, lc = l // 2, l % 2
                # C row is lr (local), col is lc (local)
                # Global C row index: c_rows[lr]
                # Global C col index: c_cols[lc]
                # W index: c_cols[lc] * n + c_rows[lr]
                global_idx = c_cols[lc] * n + c_rows[lr]
                W[global_idx][col] += w_coeffs[l]
        
        col_offset += 1
        
    return col_offset

def _make_hybrid_strassen(n, m, p):
    """
    Constructs a hybrid decomposition using Strassen for the 2x4x4 part and schoolbook for the rest.
    Rank: 28 (Strassen 2x4x4) + 8 (2x4x1) + 8 (2x1x4) + 2 (2x1x1) = 46.
    """
    rank_total = 46
    U = [[0.0] * rank_total for _ in range(n * m)]
    V = [[0.0] * rank_total for _ in range(m * p)]
    W = [[0.0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    
    # Split A (2x5) into A_left (2x4) and A_right (2x1)
    # Split B (5x5) into B_tl (4x4), B_tr (4x1), B_bl (1x4), B_br (1x1)
    # C (2x5) = A_left * B_tl + A_right * B_bl  (cols 0..3)
    #           + A_left * B_tr + A_right * B_br  (col 4)
    
    # 1. A_left (2x4) * B_tl (4x4) -> C_left (2x4)
    # Use Strassen for 2x4x4.
    # Split 2x4 into four 2x2 blocks?
    # A_left rows: 0,1. Cols: 0,1,2,3.
    # B_tl rows: 0,1,2,3. Cols: 0,1,2,3.
    # C_left rows: 0,1. Cols: 0,1,2,3.
    
    # We can view this as two independent 2x2x2 multiplications?
    # No, A_left is 2x4. B_tl is 4x4.
    # We can split B_tl into B_tl_left (4x2) and B_tl_right (4x2).
    # Then A_left * B_tl_left (2x2) and A_left * B_tl_right (2x2).
    # But A_left is 2x4, so A_left * B_tl_left is (2x4)*(4x2) = 2x2.
    # This is not a standard 2x2x2 Strassen block because A is 2x4.
    
    # Alternative: Split A_left into A_ll (2x2) and A_lr (2x2).
    # A_ll * B_tl (2x4) -> 2x4.
    # A_lr * B_tl (2x4) -> 2x4.
    # Sum them.
    # A_ll * B_tl is (2x2)*(4x4). This can be decomposed into two 2x2x2 blocks?
    # Let B_tl be split into B_tl_1 (4x2) and B_tl_2 (4x2).
    # A_ll * B_tl_1 is (2x2)*(4x2). This is not square.
    
    # Let's stick to the 2x2x2 Strassen blocks.
    # We can cover the 2x4x4 multiplication with 4 Strassen blocks of 2x2x2?
    # A is 2x4. B is 4x4. C is 2x4.
    # Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    # Split B into B11 (2x2, rows 0-1, cols 0-1), B12 (2x2, rows 0-1, cols 2-3),
    #             B21 (2x2, rows 2-3, cols 0-1), B22 (2x2, rows 2-3, cols 2-3).
    # C = A1*(B11+B21) + A2*(B12+B22).
    # This is not a simple sum of independent 2x2x2 blocks because of the sums.
    
    # Actually, the rank of 2x4x4 is known to be 28.
    # We can construct it by applying Strassen to 2x2x2 blocks.
    # Consider the multiplication as a block matrix.
    # [A1 A2] * [B11 B12; B21 B22] = [A1B11 + A2B21, A1B12 + A2B22]
    # This is not separable.
    
    # Let's use a different splitting.
    # Pad A to 4x4? No, that increases rank.
    
    # Let's just use schoolbook for the 2x4x4 part to be safe, or find a 2x2x2 decomposition.
    # Wait, 2x2x2 rank is 7.
    # 2x4x4 can be seen as 4 instances of 2x2x2?
    # If we split B into 4 blocks of 2x2, and A into 2 blocks of 2x2.
    # C = A1 B11 + A1 B12 + A2 B21 + A2 B22? No.
    # C = [A1 A2] [B11 B12; B21 B22] = A1 B11 + A2 B21 (for left half of C)
    #                             + A1 B12 + A2 B22 (for right half of C)
    
    # So we need to compute:
    # C_left = A1 B11 + A2 B21
    # C_right = A1 B12 + A2 B22
    
    # A1 B11 is 2x2 * 2x2. Rank 7.
    # A2 B21 is 2x2 * 2x2. Rank 7.
    # Sum of two rank 7 tensors? The rank of the sum is at most 14.
    # So C_left can be done in rank 14.
    # Similarly C_right in rank 14.
    # Total for 2x4x4 is 28.
    
    # Let's implement this.
    
    # Block 1: A1 (rows 0-1, cols 0-1) * B11 (rows 0-1, cols 0-1) -> contributes to C (rows 0-1, cols 0-1)
    # Block 2: A2 (rows 0-1, cols 2-3) * B21 (rows 2-3, cols 0-1) -> contributes to C (rows 0-1, cols 0-1)
    # Block 3: A1 (rows 0-1, cols 0-1) * B12 (rows 0-1, cols 2-3) -> contributes to C (rows 0-1, cols 2-3)
    # Block 4: A2 (rows 0-1, cols 2-3) * B22 (rows 2-3, cols 2-3) -> contributes to C (rows 0-1, cols 2-3)
    
    # A1 rows: [0, 1], cols: [0, 1]
    # A2 rows: [0, 1], cols: [2, 3]
    # B11 rows: [0, 1], cols: [0, 1]
    # B21 rows: [2, 3], cols: [0, 1]
    # B12 rows: [0, 1], cols: [2, 3]
    # B22 rows: [2, 3], cols: [2, 3]
    # C_left rows: [0, 1], cols: [0, 1]
    # C_right rows: [0, 1], cols: [2, 3]
    
    col_offset = _build_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], U, V, W, col_offset)
    col_offset = _build_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1], U, V, W, col_offset)
    col_offset = _build_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], U, V, W, col_offset)
    col_offset = _build_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3], U, V, W, col_offset)
    
    # Now handle the remaining parts:
    # A_left (2x4) * B_tr (4x1) -> C (2x1)
    # A_right (2x1) * B_bl (1x4) -> C (2x4)
    # A_right (2x1) * B_br (1x1) -> C (2x1)
    
    # 2. A_left (2x4) * B_tr (4x1)
    # A_left cols 0-3. B_tr rows 0-3, col 4.
    # This is 2x4 * 4x1.
    # Split A_left into A1 (2x2) and A2 (2x2).
    # Split B_tr into B_tr1 (2x1) and B_tr2 (2x1).
    # C = A1 B_tr1 + A2 B_tr2.
    # A1 B_tr1 is 2x2 * 2x1. Rank 4 (schoolbook) or less?
    # 2x2x1 tensor. Rank is 4?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For 2x2x1, rank is 4. (A00B00 + A01B10 + A10B00 + A11B10).
    # So 4 + 4 = 8.
    
    # Let's just use schoolbook for these small parts to be safe and simple.
    # Schoolbook for 2x4x1:
    for i in range(2):
        for j in range(4):
            for k in range(1):
                U[i * m + j][col_offset] = 1
                V[j * p + (4 + k)][col_offset] = 1  # B col is 4
                W[(4 + k) * n + i][col_offset] = 1  # C col is 4
                col_offset += 1
    
    # 3. A_right (2x1) * B_bl (1x4)
    # A_right col 4. B_bl row 4, cols 0-3.
    # 2x1 * 1x4. Rank 8.
    for i in range(2):
        for j in range(1):
            for k in range(4):
                U[i * m + (4 + j)][col_offset] = 1  # A col is 4
                V[(4 + j) * p + k][col_offset] = 1  # B row is 4
                W[k * n + i][col_offset] = 1
                col_offset += 1
    
    # 4. A_right (2x1) * B_br (1x1)
    # A_right col 4. B_br row 4, col 4.
    # 2x1 * 1x1. Rank 2.
    for i in range(2):
        for j in range(1):
            for k in range(1):
                U[i * m + (4 + j)][col_offset] = 1
                V[(4 + j) * p + (4 + k)][col_offset] = 1
                W[(4 + k) * n + i][col_offset] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # 1. Try the hybrid Strassen construction (Rank 46)
    best = _make_hybrid_strassen(n, m, p)
    best_rank = 46
    
    # Verify it
    if not _verify_fast(best["U"], best["V"], best["W"], n, m, p, tol=1e-8):
        # If verification fails, fall back to schoolbook
        best = _make_schoolbook(n, m, p)
        best_rank = n * m * p
    else:
        # If it works, can we do better?
        # Try to optimize the rank down from 46?
        # The theoretical lower bound is 25.
        # But finding a rank < 46 decomposition is hard.
        # For now, return the valid 46.
        pass
    
    return best
# EVOLVE-BLOCK-END
