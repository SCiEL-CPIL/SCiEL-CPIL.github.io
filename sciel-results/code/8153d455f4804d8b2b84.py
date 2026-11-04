"""
Matrix multiplication 5x5x5 tensor decomposition.
Uses multiple block-based decompositions and selects the best valid one.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Construct the standard rank n*m*p decomposition."""
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


def _verify_float(n, m, p, U, V, W, tol=1e-10):
    """Fast float verification."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i * m + j]
                v_val = V_np[j * p + k]
                w_val = W_np[k * n + i]
                dot = np.dot(u_val, v_val * w_val)
                if abs(dot - 1.0) > tol:
                    return False
    return True


def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    U: shape (4, 7) - for 2x2 input A (indices: 0=a11, 1=a12, 2=a21, 3=a22)
    V: shape (4, 7) - for 2x2 input B (indices: 0=b11, 1=b12, 2=b21, 3=b22)
    W: shape (4, 7) - for 2x2 output C (indices: 0=c11, 1=c12, 2=c21, 3=c22)
    """
    U = [
        [1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _block_decomposition(n, m, p, split_a, split_b, split_c, deadline):
    """
    Construct a block-based decomposition for n×m×p matrix multiplication.
    
    split_a: how to split n rows into blocks (e.g., [2,3] means first 2, then 3)
    split_b: how to split m cols into blocks
    split_c: how to split p cols into blocks
    
    For each combination of blocks (A_block, B_block, C_block), we need a
    decomposition for that sub-problem. We use:
    - Strassen for 2×2×2 (rank 7)
    - Schoolbook for all other sizes
    """
    if time.time() > deadline:
        return None
    
    # Compute block boundaries
    a_blocks = []
    pos = 0
    for s in split_a:
        a_blocks.append((pos, pos + s))
        pos += s
    
    b_blocks = []
    pos = 0
    for s in split_b:
        b_blocks.append((pos, pos + s))
        pos += s
    
    c_blocks = []
    pos = 0
    for s in split_c:
        c_blocks.append((pos, pos + s))
        pos += s
    
    # Total rank is sum over all block combinations
    total_rank = 0
    block_decomps = []  # List of (a_block_idx, b_block_idx, c_block_idx, U_sub, V_sub, W_sub, rank_sub)
    
    for ai, (a_start, a_end) in enumerate(a_blocks):
        for bi, (b_start, b_end) in enumerate(b_blocks):
            for ci, (c_start, c_end) in enumerate(c_blocks):
                a_size = a_end - a_start
                b_size = b_end - b_start
                c_size = c_end - c_start
                
                # For the sub-problem: multiply a_size×b_size by b_size×c_size -> a_size×c_size
                if a_size == 2 and b_size == 2 and c_size == 2:
                    U_sub, V_sub, W_sub = _strassen_2x2()
                    rank_sub = 7
                else:
                    # Schoolbook for this sub-problem
                    rank_sub = a_size * b_size * c_size
                    U_sub = [[0] * rank_sub for _ in range(a_size * b_size)]
                    V_sub = [[0] * rank_sub for _ in range(b_size * c_size)]
                    W_sub = [[0] * rank_sub for _ in range(a_size * c_size)]
                    col = 0
                    for ii in range(a_size):
                        for jj in range(b_size):
                            for kk in range(c_size):
                                U_sub[ii * b_size + jj][col] = 1
                                V_sub[jj * c_size + kk][col] = 1
                                W_sub[kk * a_size + ii][col] = 1
                                col += 1
                
                total_rank += rank_sub
                block_decomps.append((ai, bi, ci, a_start, a_end, b_start, b_end, c_start, c_end, U_sub, V_sub, W_sub, rank_sub))
    
    if total_rank >= n * m * p:
        return None  # Not better than schoolbook
    
    # Build the full matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for (ai, bi, ci, a_start, a_end, b_start, b_end, c_start, c_end, U_sub, V_sub, W_sub, rank_sub) in block_decomps:
        a_size = a_end - a_start
        b_size = b_end - b_start
        c_size = c_end - c_start
        
        for c in range(rank_sub):
            # Fill U: rows are (a_start + ii, b_start + jj) -> index (a_start+ii)*m + (b_start+jj)
            for ii in range(a_size):
                for jj in range(b_size):
                    u_val = U_sub[ii * b_size + jj][c]
                    if u_val != 0:
                        global_u_idx = (a_start + ii) * m + (b_start + jj)
                        U[global_u_idx][col_offset + c] = u_val
            
            # Fill V: rows are (b_start + jj, c_start + kk) -> index (b_start+jj)*p + (c_start+kk)
            for jj in range(b_size):
                for kk in range(c_size):
                    v_val = V_sub[jj * c_size + kk][c]
                    if v_val != 0:
                        global_v_idx = (b_start + jj) * p + (c_start + kk)
                        V[global_v_idx][col_offset + c] = v_val
            
            # Fill W: rows are (c_start + kk, a_start + ii) -> index (c_start+kk)*n + (a_start+ii)
            for kk in range(c_size):
                for ii in range(a_size):
                    w_val = W_sub[kk * a_size + ii][c]
                    if w_val != 0:
                        global_w_idx = (c_start + kk) * n + (a_start + ii)
                        W[global_w_idx][col_offset + c] = w_val
        
        col_offset += rank_sub
    
    return {"U": U, "V": V, "W": W}


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try various block partitions
    # For 5, possible splits: [5], [1,4], [2,3], [3,2], [4,1], [1,1,3], [1,2,2], [2,1,2], [2,2,1], [1,1,1,2], etc.
    # We focus on splits that use 2s (for Strassen benefit) and 3s
    
    candidate_splits = [
        [2, 3],
        [3, 2],
        [1, 4],
        [4, 1],
        [1, 1, 3],
        [1, 3, 1],
        [3, 1, 1],
        [1, 2, 2],
        [2, 1, 2],
        [2, 2, 1],
        [1, 1, 1, 2],
        [1, 1, 2, 1],
        [1, 2, 1, 1],
        [2, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    
    for split in candidate_splits:
        if time.time() > deadline:
            break
        if sum(split) != 5:
            continue
        
        result = _block_decomposition(n, m, p, split, split, split, deadline)
        if result is None:
            continue
        
        rank = len(result['U'][0])
        if rank >= best_rank:
            continue
        
        # Verify the result
        if _verify_float(n, m, p, result['U'], result['V'], result['W'], tol=1e-10):
            best = result
            best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
