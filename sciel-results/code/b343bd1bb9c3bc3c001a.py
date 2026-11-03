"""Matrix multiplication 3x4x8 with optimized tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
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


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _build_88_decomposition():
    """Build the rank-88 decomposition using Strassen blocks."""
    n, m, p = 3, 4, 8
    total_rank = 88
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()

    col_offset = 0
    # 8 Strassen blocks for i=0,1
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            u_rows = [0 * m + j0, 0 * m + j1, 1 * m + j0, 1 * m + j1]
            v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
            w_rows = [k0 * n + 0, k1 * n + 0, k0 * n + 1, k1 * n + 1]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
    # 8 schoolbook blocks for i=2
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [j0 * p + k0, j1 * p + k0, j0 * p + k1, j1 * p + k1]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def _find_3x2x2_rank5(deadline):
    """Search for a rank-5 decomposition of 3x2x2 matmul tensor.
    
    Returns (U, V, W) if found, None otherwise.
    U: 6x5, V: 4x5, W: 6x5
    """
    def _check(U, V, W):
        """Check if decomposition is valid."""
        for i in range(3):
            for k in range(2):
                val = 0
                for c in range(5):
                    val += U[i * 2 + 0][c] * V[0 * 2 + k][c] * W[k * 3 + i][c]
                    val += U[i * 2 + 1][c] * V[1 * 2 + k][c] * W[k * 3 + i][c]
                if val != 1:
                    return False
        return True

    rng = random.Random(42)
    
    # Try random search with small integers/half-integers
    # Increase iterations and use a better distribution
    max_iters = 50000
    for it in range(max_iters):
        if time.time() > deadline:
            return None
        
        # Generate random U, V, W with entries from {-1, -0.5, 0, 0.5, 1}
        U = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(5)] for _ in range(6)]
        V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(5)] for _ in range(4)]
        W = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(5)] for _ in range(6)]
        
        if _check(U, V, W):
            return (U, V, W)
            
    return None


def _build_80_decomposition(U322, V322, W322):
    """Build a rank-80 decomposition using a rank-5 3x2x2 block.
    
    For each kp pair, use one 3x2x2 block (rank 5) for each jp pair.
    Wait, this doesn't quite work because the dimensions don't match up cleanly.
    
    Actually, the 3x2x2 tensor handles: A (3x2) * B (2x2) -> C (3x2)
    In our problem, for each kp pair, we have A (3x4) * B (4x2) -> C (3x2)
    We can split A into two 3x2 blocks (for jp=0 and jp=1) and B into one 2x2 block.
    So: C = A_left * B + A_right * B where A_left is 3x2 (columns 0,1) and A_right is 3x2 (columns 2,3)
    
    But that's two separate 3x2x2 multiplications, each with rank 5, total rank 10 per kp pair.
    4 kp pairs * 10 = 40. That's much better than 22 per kp pair!
    
    Wait, that can't be right. Let me reconsider.
    
    A (3x4) * B (4x2) = A_left (3x2) * B_left (2x2) + A_right (3x2) * B_right (2x2)
    where B_left = B[:, 0:2] and B_right = B[:, 2:4]... no wait, B is 4x2, so B_left and B_right don't make sense that way.
    
    Actually: A is 3x4, B is 4x2. 
    A = [A_left | A_right] where A_left is 3x2 (cols 0,1) and A_right is 3x2 (cols 2,3)
    B = [B_top; B_bottom] where B_top is 2x2 (rows 0,1) and B_bottom is 2x2 (rows 2,3)
    
    C = A_left * B_top + A_right * B_bottom
    
    Each term is a 3x2 * 2x2 = 3x2 multiplication, which is the 3x2x2 tensor!
    So if we have a rank-5 decomposition for 3x2x2, we can do:
    - For each kp pair: 2 * rank(3x2x2) = 2 * 5 = 10
    - 4 kp pairs: 4 * 10 = 40 total rank
    
    That would be a huge improvement from 88 to 40!
    """
    n, m, p = 3, 4, 8
    rank_322 = len(U322[0])  # should be 5
    total_rank = 4 * 2 * rank_322  # 4 kp pairs * 2 (left/right) * rank_322
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for kp in range(4):
        k0, k1 = 2 * kp, 2 * kp + 1
        for side in range(2):  # 0 = left (j=0,1), 1 = right (j=2,3)
            j0, j1 = 2 * side, 2 * side + 1
            col_offset = (kp * 2 + side) * rank_322
            
            # U rows: (i, j) for i in 0..2, j in {j0, j1}
            # In 3x2x2: U index is i*2 + j_local where j_local is 0 or 1
            for i in range(3):
                for j_local in range(2):
                    j = j0 + j_local
                    u_idx = i * 2 + j_local
                    for c in range(rank_322):
                        U[i * m + j][col_offset + c] += U322[u_idx][c]
            
            # V rows: (j, k) for j in {j0, j1}, k in {k0, k1}
            # In 3x2x2: V index is j_local * 2 + k_local
            for j_local in range(2):
                for k_local in range(2):
                    j = j0 + j_local
                    k = k0 + k_local
                    v_idx = j_local * 2 + k_local
                    for c in range(rank_322):
                        V[j * p + k][col_offset + c] += V322[v_idx][c]
            
            # W rows: (i, k) for i in 0..2, k in {k0, k1}
            # In 3x2x2: W index is i * 2 + k_local
            for i in range(3):
                for k_local in range(2):
                    k = k0 + k_local
                    w_idx = i * 2 + k_local
                    for c in range(rank_322):
                        W[k * n + i][col_offset + c] += W322[w_idx][c]
    
    return {"U": U, "V": V, "W": W}


def _construct_3x4x8(deadline):
    """Construct the decomposition."""
    # First, try to find a rank-5 3x2x2 decomposition
    result_322 = _find_3x2x2_rank5(deadline - 5)  # leave 5s for construction and verification
    
    if result_322 is not None:
        U322, V322, W322 = result_322
        result = _build_80_decomposition(U322, V322, W322)
        n, m, p = DIMENSIONS
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    
    # Fallback: rank-88 decomposition
    result = _build_88_decomposition()
    n, m, p = DIMENSIONS
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Last resort: schoolbook rank 96
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


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit
    
    result = _construct_3x4x8(deadline)
    
    return result
