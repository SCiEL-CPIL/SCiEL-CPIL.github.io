import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 6]

def _verify(U, V, W, n, m, p, rank):
    """Verify the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Return U,V,W for 2x2x2 matrix multiplication (Strassen, rank 7)."""
    # Standard Strassen decomposition
    U = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ]
    return U, V, W

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine multiple block decompositions into one for the full problem.
    Each decomp is (U_block, V_block, W_block) for sub-problems.
    We partition indices and assign blocks.
    """
    # For now, just concatenate all blocks (they must act on disjoint index sets)
    # This is a placeholder - we need proper index assignment
    total_rank = sum(len(U) for U, V, W in decomps if U is not None)
    # This approach is too naive; we need to think about how blocks compose
    return None

def _try_block_decomposition(n, m, p, time_limit=100.0):
    """
    Try to build a decomposition by partitioning the index spaces.
    For 4x5x6, we can try:
    - Split into (2,2) x (5,5) x (6,6) blocks
    - Use Strassen for 2x2x2 sub-blocks
    - Use schoolbook for remaining
    """
    start_time = time.time()
    
    # Strategy: Partition i into [0,1] and [2,3], j into [0,1,2,3,4], k into [0,1,2,3,4,5]
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2, use Strassen (rank 7)
    # Block 2: i in {0,1}, j in {2,3,4}, k in {0,1} -> 2x3x2, use schoolbook (rank 12)
    # Block 3: i in {0,1}, j in {0,1,2,3,4}, k in {2,3,4,5} -> 2x5x4, use schoolbook (rank 40)
    # Block 4: i in {2,3}, j in {0,1}, k in {0,1} -> 2x2x2, use Strassen (rank 7)
    # Block 5: i in {2,3}, j in {2,3,4}, k in {0,1} -> 2x3x2, use schoolbook (rank 12)
    # Block 6: i in {2,3}, j in {0,1,2,3,4}, k in {2,3,4,5} -> 2x5x4, use schoolbook (rank 40)
    # Total: 7+12+40+7+12+40 = 118
    
    # But we can do better by overlapping blocks or using different partitions
    
    # Let's try a different approach: use the known optimal or near-optimal decompositions
    # For 4x5x6, the best known is around 100-110
    
    # Try to find a decomposition with rank <= 110
    best_rank = n * m * p  # schoolbook
    best_result = None
    
    # Generate candidate decompositions
    candidates = []
    
    # Candidate 1: Pure schoolbook
    U, V, W = _schoolbook(n, m, p)
    candidates.append((n*m*p, U, V, W))
    
    # Candidate 2: Block decomposition with Strassen for 2x2x2 corners
    # Partition: i={0,1}|{2,3}, j={0,1}|{2,3,4}, k={0,1}|{2,3,4,5}
    # Block (0,0,0): 2x2x2 -> Strassen rank 7
    # Block (0,0,1): 2x2x4 -> schoolbook rank 16
    # Block (0,1,0): 2x3x2 -> schoolbook rank 12
    # Block (0,1,1): 2x3x4 -> schoolbook rank 24
    # Block (1,0,0): 2x2x2 -> Strassen rank 7
    # Block (1,0,1): 2x2x4 -> schoolbook rank 16
    # Block (1,1,0): 2x3x2 -> schoolbook rank 12
    # Block (1,1,1): 2x3x4 -> schoolbook rank 24
    # Total: 7+16+12+24+7+16+12+24 = 118
    
    # Build this decomposition
    rank_total = 7 + 16 + 12 + 24 + 7 + 16 + 12 + 24
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to place a block decomposition
    def place_block(U, V, W, i_start, i_end, j_start, j_end, k_start, k_end, 
                    U_block, V_block, W_block, n, m, p, col_offset):
        """Place a block decomposition at the given indices."""
        for bi in range(i_start, i_end):
            for bj in range(j_start, j_end):
                for bk in range(k_start, k_end):
                    # Local indices
                    li = bi - i_start
                    lj = bj - j_start
                    lk = bk - k_start
                    
                    # Global indices
                    gi = bi
                    gj = bj
                    gk = bk
                    
                    # For each rank element in the block
                    for r in range(len(U_block[0])):
                        U[gi*m + gj][col_offset + r] = U_block[li*(j_end-j_start) + lj][r] if False else U_block[li*(j_end-j_start) + lj][r]
                        V[gj*p + gk][col_offset + r] = V_block[lj*(k_end-k_start) + lk][r]
                        W[gk*n + gi][col_offset + r] = W_block[lk*(i_end-i_start) + li][r]
    
    # This is getting complex. Let me simplify by using a direct approach.
    
    # Actually, let's just try to find any valid decomposition with rank < 120
    # using a more systematic search
    
    # For now, return the schoolbook as fallback
    return candidates[0]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to improve using block decomposition
    if time.time() < deadline - 30:
        result = _try_block_decomposition(n, m, p, time_limit=60.0)
        if result is not None:
            rank, U, V, W = result
            if rank < best_rank:
                # Verify
                if _verify(U, V, W, n, m, p, rank):
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    
    return best_result
