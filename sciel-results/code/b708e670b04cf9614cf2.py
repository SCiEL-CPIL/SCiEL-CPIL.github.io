"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x):
    """Check if x is an integer or half-integer (multiple of 0.5)."""
    if abs(x - round(x)) < 1e-10:
        return True
    if abs(x * 2 - round(x * 2)) < 1e-10:
        return True
    return False

def _all_half_integer(U, V, W):
    """Check if all entries in U, V, W are half-integers."""
    for M in [U, V, W]:
        for row in M:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _construct_rank_36_decomposition():
    """
    Construct a rank-36 decomposition using Strassen's algorithm on 2x2 blocks
    for the 2x4x4 part and schoolbook for the 2x4x1 part.
    """
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen pattern for 2x2x2
    # Each entry is (u_offsets, v_offsets, w_offsets)
    # u_offsets: list of (local_idx, coeff)
    # v_offsets: list of (local_idx, coeff)
    # w_offsets: list of (local_idx, coeff)
    
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3
        ([(0, 1)], [(2, 1), (3, -1)], [(1, 1)]),
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, 1)])
    ]
    
    def apply_block(a_j_offset, b_j_offset, b_k_offset, c_k_offset, start_col):
        """
        Apply a 2x2x2 Strassen block.
        a_j_offset: offset for A's j index (0 or 2)
        b_j_offset: offset for B's j index (0 or 2)
        b_k_offset: offset for B's k index (0, 2)
        c_k_offset: offset for C's k index (0, 2)
        """
        for t in range(7):
            u_offs, v_offs, w_offs = strassen_terms[t]
            c_idx = start_col + t
            
            for local_j, coeff in u_offs:
                # Local A index: i (0,1) * 2 + local_j
                # Global A index: i * 4 + (local_j + a_j_offset)
                # U row: i*m + j
                for i in range(2):
                    local_a_idx = i * 2 + local_j
                    global_j = local_j + a_j_offset
                    global_i = i
                    U[global_i * m + global_j][c_idx] += coeff
            
            for local_j, local_k, coeff in v_offs:
                # Note: v_offs is flat list of (idx, coeff) for 2x2 B
                # B local index: local_j * 2 + local_k
                # But strassen_terms v_offs are (idx, coeff) where idx is 0..3
                pass 
            # Re-parse v_offs correctly
            for idx, coeff in v_offs:
                local_j = idx // 2
                local_k = idx % 2
                global_j = local_j + b_j_offset
                global_k = local_k + b_k_offset
                V[global_j * p + global_k][c_idx] += coeff
            
            for idx, coeff in w_offs:
                local_k = idx // 2
                local_i = idx % 2
                global_k = local_k + c_k_offset
                global_i = local_i
                W[global_k * n + global_i][c_idx] += coeff
                
        return start_col + 7

    # Block 1: A_L B_LL -> C_LL
    # A_L: j in 0,1. B_LL: j in 0,1, k in 0,1. C_LL: k in 0,1.
    col = apply_block(0, 0, 0, 0, col)
    
    # Block 2: A_R B_RL -> C_RL (adds to C_LL? No, C_L is cols 0,1)
    # A_R: j in 2,3. B_RL: j in 2,3, k in 0,1. C_L: k in 0,1.
    col = apply_block(2, 2, 0, 0, col)
    
    # Block 3: A_L B_LR -> C_LR (C_R is cols 2,3)
    # A_L: j in 0,1. B_LR: j in 0,1, k in 2,3. C_R: k in 2,3.
    col = apply_block(0, 0, 2, 2, col)
    
    # Block 4: A_R B_RR -> C_RR
    # A_R: j in 2,3. B_RR: j in 2,3, k in 2,3. C_R: k in 2,3.
    col = apply_block(2, 2, 2, 2, col)
    
    # Block 5: A_L B_L5 -> C_L5 (k=4)
    # This is 2x2x1. Rank 4.
    # Terms:
    # 1. A_00 B_00 -> C_00
    # 2. A_01 B_10 -> C_00
    # 3. A_10 B_00 -> C_10
    # 4. A_11 B_10 -> C_10
    
    # Local A idx 0,1,2,3. Local B idx 0,1 (j=0,1 k=0). Local C idx 0,1 (k=0 i=0,1)
    # Global A: j=0,1. Global B: j=0,1, k=4. Global C: k=4, i=0,1.
    
    # Term 5.1
    U[0*4+0][col] += 1 # A_00
    V[0*5+4][col] += 1 # B_04
    W[4*2+0][col] += 1 # C_40
    col += 1
    
    # Term 5.2
    U[0*4+1][col] += 1 # A_01
    V[1*5+4][col] += 1 # B_14
    W[4*2+0][col] += 1 # C_40
    col += 1
    
    # Term 5.3
    U[1*4+0][col] += 1 # A_10
    V[0*5+4][col] += 1 # B_04
    W[4*2+1][col] += 1 # C_41
    col += 1
    
    # Term 5.4
    U[1*4+1][col] += 1 # A_11
    V[1*5+4][col] += 1 # B_14
    W[4*2+1][col] += 1 # C_41
    col += 1
    
    # Block 6: A_R B_R5 -> C_L5 (k=4)
    # A_R: j=2,3. B_R5: j=2,3, k=4. C_L5: k=4.
    
    # Term 6.1
    U[0*4+2][col] += 1 # A_02
    V[2*5+4][col] += 1 # B_24
    W[4*2+0][col] += 1 # C_40
    col += 1
    
    # Term 6.2
    U[0*4+3][col] += 1 # A_03
    V[3*5+4][col] += 1 # B_34
    W[4*2+0][col] += 1 # C_40
    col += 1
    
    # Term 6.3
    U[1*4+2][col] += 1 # A_12
    V[2*5+4][col] += 1 # B_24
    W[4*2+1][col] += 1 # C_41
    col += 1
    
    # Term 6.4
    U[1*4+3][col] += 1 # A_13
    V[3*5+4][col] += 1 # B_34
    W[4*2+1][col] += 1 # C_41
    col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 145
    
    n, m, p = DIMENSIONS
    
    # Try rank 36 decomposition
    U, V, W = _construct_rank_36_decomposition()
    
    if _validate(U, V, W, n, m, p):
        if _all_half_integer(U, V, W):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook rank 40
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
