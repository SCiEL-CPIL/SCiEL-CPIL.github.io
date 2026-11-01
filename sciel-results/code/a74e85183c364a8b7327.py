"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)
    best_rank = n * m * p
    best = (U0, V0, W0)

    # Try structured decompositions
    # Strategy: split p=8 into blocks and use efficient sub-decompositions
    # For 2x3 * 3x2: minimum rank is 8 (not 11 as incorrectly assumed)
    # Actually, let's use a proper approach
    
    # Try 2-block decomposition: 2x3 * 3x4 for each half
    if time.time() < deadline - 30:
        try:
            U1, V1, W1, r1 = _two_block_decomposition(n, m, p)
            if U1 is not None and _verify_correct(n, m, p, U1, V1, W1):
                if r1 < best_rank:
                    best = (U1, V1, W1)
                    best_rank = r1
        except Exception:
            pass

    # Try 4-block decomposition: 2x3 * 3x2 for each quarter
    if time.time() < deadline - 20:
        try:
            U2, V2, W2, r2 = _four_block_decomposition_correct(n, m, p)
            if U2 is not None and _verify_correct(n, m, p, U2, V2, W2):
                if r2 < best_rank:
                    best = (U2, V2, W2)
                    best_rank = r2
        except Exception:
            pass

    # Try to find even lower rank via direct construction
    if time.time() < deadline - 10:
        try:
            U3, V3, W3, r3 = _structured_low_rank(n, m, p, deadline)
            if U3 is not None and _verify_correct(n, m, p, U3, V3, W3):
                if r3 < best_rank:
                    best = (U3, V3, W3)
                    best_rank = r3
        except Exception:
            pass

    U, V, W = best
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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


def _verify_correct(n, m, p, U, V, W):
    """
    Fast vectorized verification of tensor decomposition.
    
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1
    """
    rank = U.shape[1]
    
    # Build index arrays for broadcasting
    # For each (i,j,k), we need U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
    # Sum over r of the element-wise product
    
    # Use numpy to build the full tensor efficiently
    # Create index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    u_idx = i_idx * m + j_idx  # shape (n*m*p,)
    v_idx = j_idx * p + k_idx  # shape (n*m*p,)
    w_idx = k_idx * n + i_idx  # shape (n*m*p,)
    
    # Extract rows
    U_sel = U[u_idx, :]  # (n*m*p, rank)
    V_sel = V[v_idx, :]  # (n*m*p, rank)
    W_sel = W[w_idx, :]  # (n*m*p, rank)
    
    # Sum over rank dimension
    tensor_vals = np.sum(U_sel * V_sel * W_sel, axis=1)  # (n*m*p,)
    
    if np.max(np.abs(tensor_vals - 1.0)) > 1e-9:
        return False
    return True


def _two_block_decomposition(n, m, p):
    """
    Split p=8 into two blocks of size 4.
    For each 2x3 * 3x4 sub-problem, try to find a low-rank decomposition.
    
    For 2x3 * 3x4: schoolbook rank is 24.
    We can try to do better using structure.
    
    One approach: view 2x3 * 3x4 as computing C = A * B where:
    A is 2x3, B is 3x4, C is 2x4
    
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    We can try to use a Strassen-like approach by grouping.
    """
    n, m, p = 2, 3, 8
    block_size = p // 2  # 4
    
    # For 2x3 * 3x4, let's try rank 12 (half of schoolbook 24)
    # Actually, let's try to construct a valid decomposition
    
    # Approach: use the fact that we can write the 3 middle rows as 2+1
    # A = [A_2x2 | A_2x1], B = [B_2x4; B_1x4]
    # C = A_2x2 * B_2x4 + A_2x1 * B_1x4
    
    # A_2x2 * B_2x4: can use Strassen-like for 2x2 * 2x4
    # A_2x1 * B_1x4: rank 4 (outer product)
    
    # For 2x2 * 2x4: C[i,k] = sum_{j=0}^{1} A[i,j] * B[j,k]
    # This is like two 2x2 multiplications side by side
    # Each 2x2 * 2x2 can be done in rank 7 (Strassen)
    # So 2x2 * 2x4 can be done in rank 14 (two Strassen blocks)
    # Plus rank 4 for the outer product = total rank 18 per block
    # Total for 2 blocks: 36
    
    # But we can do better. Let's try a unified approach.
    
    # Actually, let's just return None for now and rely on other methods
    return None, None, None, 999


def _four_block_decomposition_correct(n, m, p):
    """
    Split p=8 into 4 blocks of size 2.
    For each 2x3 * 3x2 sub-problem, use a proper low-rank decomposition.
    
    For 2x3 * 3x2: C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    where A is 2x3, B is 3x2, C is 2x2.
    
    Schoolbook rank: 12.
    
    We can decompose as:
    A = [A_left (2x2) | A_right (2x1)]
    B = [B_top (2x2); B_bottom (1x2)]
    
    C = A_left * B_top + A_right * B_bottom
    
    A_left * B_top: 2x2 * 2x2, rank 7 (Strassen)
    A_right * B_bottom: 2x1 * 1x2, rank 4 (outer product)
    
    Total per block: 11
    Total for 4 blocks: 44
    """
    n, m, p = 2, 3, 8
    block_size = p // 4  # 2
    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44
    
    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44
    
    for b in range(4):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block
        
        # Strassen decomposition for 2x2 * 2x2
        # A_left = A[:, 0:2], B_top = B[0:2, k_offset:k_offset+2]
        
        # Strassen for 2x2 * 2x2:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # U indices for A_left (2x2 part of 2x3 matrix):
        # A[i, j] where i=0,1 and j=0,1 (the left 2 columns)
        # U index: i * m + j = i * 3 + j
        
        strassen_U = [
            {0 * 3 + 0: 1, 1 * 3 + 1: 1},      # M1: A00 + A11
            {1 * 3 + 0: 1, 1 * 3 + 1: 1},      # M2: A10 + A11
            {0 * 3 + 0: 1},                     # M3: A00
            {1 * 3 + 1: 1},                     # M4: A11
            {0 * 3 + 0: 1, 0 * 3 + 1: 1},      # M5: A00 + A01
            {1 * 3 + 0: 1, 0 * 3 + 0: -1},     # M6: A10 - A00
            {0 * 3 + 1: 1, 1 * 3 + 1: -1},     # M7: A01 - A11
        ]
        
        # V indices for B_top (2x2 part of 3x2 block):
        # B[j, k] where j=0,1 and k=k_offset, k_offset+1
        # V index: j * p + k = j * 8 + k
        
        strassen_V = [
            {0 * 8 + k_offset + 0: 1, 1 * 8 + k_offset + 1: 1},  # M1: B00 + B11
            {0 * 8 + k_offset + 0: 1},                           # M2: B00
            {1 * 8 + k_offset + 0: 1, 1 * 8 + k_offset + 1: -1}, # M3: B10 - B11
            {0 * 8 + k_offset + 1: 1, 0 * 8 + k_offset + 0: -1}, # M4: B01 - B00
            {1 * 8 + k_offset + 1: 1},                           # M5: B11
            {0 * 8 + k_offset + 0: 1, 0 * 8 + k_offset + 1: 1},  # M6: B00 + B01
            {1 * 8 + k_offset + 0: 1, 1 * 8 + k_offset + 1: 1},  # M7: B10 + B11
        ]
        
        # W indices for C (2x2 output):
        # C[i, k] where i=0,1 and k=k_offset, k_offset+1
        # W index: k * n + i = k * 2 + i
        
        strassen_W = [
            {(k_offset + 0) * 2 + 0: 1, (k_offset + 1) * 2 + 1: 1},  # M1: C00+1, C11+1
            {(k_offset + 0) * 2 + 1: 1, (k_offset + 1) * 2 + 1: -1}, # M2: C10+1, C11-1
            {(k_offset + 1) * 2 + 0: 1, (k_offset + 1) * 2 + 1: 1},  # M3: C01+1, C11+1
            {(k_offset + 0) * 2 + 0: 1, (k_offset + 0) * 2 + 1: 1},  # M4: C00+1, C10+1
            {(k_offset + 0) * 2 + 0: -1, (k_offset + 1) * 2 + 0: 1}, # M5: C00-1, C01+1
            {(k_offset + 1) * 2 + 1: 1},                             # M6: C11+1
            {(k_offset + 0) * 2 + 0: 1},                             # M7: C00+1
        ]
        
        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val
        
        # Outer product part: A_right (2x1) * B_bottom (1x2)
        # A_right: A[:, 2], B_bottom: B[2, k_offset:k_offset+2]
        # C[i, k] += A[i, 2] * B[2, k]
        
        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk  # 7, 8, 9, 10
                U[ii * 3 + 2, col_offset + c] = 1
                V[2 * 8 + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * 2 + ii, col_offset + c] = 1
    
    return U, V, W, total_rank


def _structured_low_rank(n, m, p, deadline):
    """
    Try to find a lower rank decomposition using a more global approach.
    Attempts a 2-block decomposition with rank-11 per 2x3*3x4 block.
    """
    if time.time() > deadline - 5:
        return None, None, None, 999
    
    # Try splitting p=8 into two blocks of size 4
    # For 2x3 * 3x4: C[i,k] = sum_j A[i,j]*B[j,k]
    # Split B into B_top (2x4) and B_bottom (1x4)
    # C = A_left(2x2)*B_top(2x4) + A_right(2x1)*B_bottom(1x4)
    # A_left*B_top: use two Strassen 2x2*2x2 blocks (rank 14)
    # A_right*B_bottom: outer product rank 4
    # Total per block: 18, total: 36
    # But we can potentially do better with a unified approach
    
    # Alternative: try rank 34 by sharing some structure
    # For now, try the 36-rank construction and see if it's valid
    total_rank = 36
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for b in range(2):
        k_offset = b * 4  # 0, 4
        col_offset = b * 18
        
        # Two Strassen blocks for A_left(2x2) * B_top(2x4)
        for sb in range(2):
            kk = k_offset + sb * 2  # 0,2 or 4,6
            sc = col_offset + sb * 7
            
            strassen_U = [
                {0 * 3 + 0: 1, 1 * 3 + 1: 1},
                {1 * 3 + 0: 1, 1 * 3 + 1: 1},
                {0 * 3 + 0: 1},
                {1 * 3 + 1: 1},
                {0 * 3 + 0: 1, 0 * 3 + 1: 1},
                {1 * 3 + 0: 1, 0 * 3 + 0: -1},
                {0 * 3 + 1: 1, 1 * 3 + 1: -1},
            ]
            
            strassen_V = [
                {0 * 8 + kk + 0: 1, 1 * 8 + kk + 1: 1},
                {0 * 8 + kk + 0: 1},
                {1 * 8 + kk + 0: 1, 1 * 8 + kk + 1: -1},
                {0 * 8 + kk + 1: 1, 0 * 8 + kk + 0: -1},
                {1 * 8 + kk + 1: 1},
                {0 * 8 + kk + 0: 1, 0 * 8 + kk + 1: 1},
                {1 * 8 + kk + 0: 1, 1 * 8 + kk + 1: 1},
            ]
            
            strassen_W = [
                {(kk + 0) * 2 + 0: 1, (kk + 1) * 2 + 1: 1},
                {(kk + 0) * 2 + 1: 1, (kk + 1) * 2 + 1: -1},
                {(kk + 1) * 2 + 0: 1, (kk + 1) * 2 + 1: 1},
                {(kk + 0) * 2 + 0: 1, (kk + 0) * 2 + 1: 1},
                {(kk + 0) * 2 + 0: -1, (kk + 1) * 2 + 0: 1},
                {(kk + 1) * 2 + 1: 1},
                {(kk + 0) * 2 + 0: 1},
            ]
            
            for c in range(7):
                for row, val in strassen_U[c].items():
                    U[row, sc + c] += val
                for row, val in strassen_V[c].items():
                    V[row, sc + c] += val
                for row, val in strassen_W[c].items():
                    W[row, sc + c] += val
        
        # Outer product: A_right(2x1) * B_bottom(1x4)
        for ii in range(2):
            for kk in range(4):
                c = 14 + ii * 4 + kk
                U[ii * 3 + 2, col_offset + c] = 1
                V[2 * 8 + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * 2 + ii, col_offset + c] = 1
    
    if _verify_correct(n, m, p, U, V, W):
        return U, V, W, total_rank
    return None, None, None, 999


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
