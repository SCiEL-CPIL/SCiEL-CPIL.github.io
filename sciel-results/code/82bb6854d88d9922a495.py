"""Rewritten program for matrix multiplication 4x4x6 with Strassen-based decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook rank-96 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_222(u_indices, v_indices, w_indices, U, V, W, col_start):
    """
    Apply Strassen's algorithm for 2x2x2 block multiplication.
    Returns the number of rank-1 terms added (7 for Strassen).
    
    u_indices: list of 4 U row indices [a00, a01, a10, a11]
    v_indices: list of 4 V row indices [b00, b01, b10, b11]
    w_indices: list of 4 W row indices [c00, c01, c10, c11]
    """
    a00, a01, a10, a11 = u_indices
    b00, b01, b10, b11 = v_indices
    c00, c01, c10, c11 = w_indices
    
    col = col_start
    
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Term 1: M1
    U[a00, col] += 1
    U[a11, col] += 1
    V[b00, col] += 1
    V[b11, col] += 1
    W[c00, col] += 1
    W[c11, col] += 1
    col += 1
    
    # Term 2: M2
    U[a10, col] += 1
    U[a11, col] += 1
    V[b00, col] += 1
    W[c10, col] += 1
    W[c11, col] -= 1
    col += 1
    
    # Term 3: M3
    U[a00, col] += 1
    V[b01, col] += 1
    V[b11, col] -= 1
    W[c01, col] += 1
    W[c11, col] += 1
    col += 1
    
    # Term 4: M4
    U[a11, col] += 1
    V[b10, col] += 1
    V[b00, col] -= 1
    W[c00, col] += 1
    W[c10, col] += 1
    col += 1
    
    # Term 5: M5
    U[a00, col] += 1
    U[a01, col] += 1
    V[b11, col] += 1
    W[c00, col] -= 1
    W[c01, col] += 1
    col += 1
    
    # Term 6: M6
    U[a10, col] += 1
    U[a00, col] -= 1
    V[b00, col] += 1
    V[b01, col] += 1
    W[c10, col] += 1
    col += 1
    
    # Term 7: M7
    U[a01, col] += 1
    U[a11, col] -= 1
    V[b10, col] += 1
    V[b11, col] += 1
    W[c00, col] -= 1
    W[c11, col] += 1
    col += 1
    
    return col - col_start


def _strassen_4x4x6():
    """
    Build a Strassen-based decomposition for 4x4x6.
    
    The 4x4x6 multiplication can be viewed as 4 independent 2x2x6 multiplications
    (rows of A in pairs, cols of C in pairs). Each 2x2x6 can be split into three
    2x2x2 Strassen multiplications.
    
    Actually, let's think of it differently:
    A is 4x4, B is 4x6, C is 4x6.
    We can partition A into four 2x2 blocks: A11, A12, A21, A22
    B into two 2x3 blocks top and bottom: B_top (2x6), B_bottom (2x6)
    C into two 2x6 blocks: C_top (2x6), C_bottom (2x6)
    
    C_top = A11 * B_top + A12 * B_bottom
    C_bottom = A21 * B_top + A22 * B_bottom
    
    For each 2x2 * 2x6 multiplication, we can further partition B_top into three 2x2 blocks:
    B_top = [Bt0, Bt1, Bt2] where each is 2x2
    Similarly B_bottom = [Bb0, Bb1, Bb2]
    
    C_top = [Ct0, Ct1, Ct2] where each is 2x2
    C_bottom = [Cb0, Cb1, Cb2]
    
    Ct0 = A11*Bt0 + A12*Bb0
    Ct1 = A11*Bt1 + A12*Bb1
    Ct2 = A11*Bt2 + A12*Bb2
    Cb0 = A21*Bt0 + A22*Bb0
    Cb1 = A21*Bt1 + A22*Bb1
    Cb2 = A21*Bt2 + A22*Bb2
    
    Each 2x2*2x2 = 2x2 product can use Strassen (7 terms).
    So we have 12 such products, each using 7 terms = 84 total.
    """
    n, m, p = 4, 4, 6
    total_rank = 12 * 7  # 84
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # A row indices: i in {0,1} for top, {2,3} for bottom
    # A col indices: j in {0,1} for left, {2,3} for right
    # B row indices: j in {0,1} for top, {2,3} for bottom
    # B col indices: k in {0,1}, {2,3}, {4,5} for the three 2x2 blocks
    # C row indices: i in {0,1} for top, {2,3} for bottom
    # C col indices: k in {0,1}, {2,3}, {4,5}
    
    # Define block index ranges
    # A blocks:
    # A11: rows 0,1 cols 0,1
    # A12: rows 0,1 cols 2,3
    # A21: rows 2,3 cols 0,1
    # A22: rows 2,3 cols 2,3
    
    a_blocks = {
        '11': [0, 1, 4, 5],      # a00, a01, a10, a11 (U row indices)
        '12': [2, 3, 6, 7],
        '21': [8, 9, 12, 13],
        '22': [10, 11, 14, 15],
    }
    
    # B blocks for top half (rows 0,1 of B, i.e., j=0,1)
    # Bt0: cols 0,1 -> V rows 0,1
    # Bt1: cols 2,3 -> V rows 2,3
    # Bt2: cols 4,5 -> V rows 4,5
    
    # B blocks for bottom half (rows 2,3 of B, i.e., j=2,3)
    # Bb0: cols 0,1 -> V rows 6,7
    # Bb1: cols 2,3 -> V rows 8,9
    # Bb2: cols 4,5 -> V rows 10,11
    
    # For each B 2x2 block, the V indices are [b00, b01, b10, b11]
    # Bt0: b00=V[0], b01=V[1], b10=V[4], b11=V[5]
    # Wait, V is indexed by (j, k) where j is row of B (0..3), k is col of B (0..5)
    # V row index = j * p + k = j * 6 + k
    
    # Bt0: j in {0,1}, k in {0,1}
    #   b00: j=0,k=0 -> V[0]
    #   b01: j=0,k=1 -> V[1]
    #   b10: j=1,k=0 -> V[6]
    #   b11: j=1,k=1 -> V[7]
    
    b_blocks_top = {
        0: [0, 1, 6, 7],    # Bt0
        1: [2, 3, 8, 9],    # Bt1
        2: [4, 5, 10, 11],  # Bt2
    }
    
    # Bb0: j in {2,3}, k in {0,1}
    #   b00: j=2,k=0 -> V[12]
    #   b01: j=2,k=1 -> V[13]
    #   b10: j=3,k=0 -> V[18]
    #   b11: j=3,k=1 -> V[19]
    
    b_blocks_bottom = {
        0: [12, 13, 18, 19],   # Bb0
        1: [14, 15, 20, 21],   # Bb1
        2: [16, 17, 22, 23],   # Bb2
    }
    
    # W is indexed by (k, i) where k is col of C (0..5), i is row of C (0..3)
    # W row index = k * n + i = k * 4 + i
    
    # C top: i in {0,1}
    # C bottom: i in {2,3}
    
    # For C block at position (top/bottom, block 0/1/2):
    # Ct0: i in {0,1}, k in {0,1}
    #   c00: k=0,i=0 -> W[0]
    #   c01: k=1,i=0 -> W[1]
    #   c10: k=0,i=1 -> W[4]
    #   c11: k=1,i=1 -> W[5]
    
    c_blocks_top = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
    }
    
    c_blocks_bottom = {
        0: [16, 17, 20, 21],
        1: [18, 19, 22, 23],
        2: [24, 25, 28, 29],
    }
    
    # Now build the decomposition
    # For each pair (A_block, B_block) and corresponding C_block:
    
    products = [
        # (a_block, b_block_type, b_block_key, c_block_type, c_block_key)
        ('11', 'top', 0, 'top', 0),
        ('11', 'top', 1, 'top', 1),
        ('11', 'top', 2, 'top', 2),
        ('12', 'bottom', 0, 'top', 0),
        ('12', 'bottom', 1, 'top', 1),
        ('12', 'bottom', 2, 'top', 2),
        ('21', 'top', 0, 'bottom', 0),
        ('21', 'top', 1, 'bottom', 1),
        ('21', 'top', 2, 'bottom', 2),
        ('22', 'bottom', 0, 'bottom', 0),
        ('22', 'bottom', 1, 'bottom', 1),
        ('22', 'bottom', 2, 'bottom', 2),
    ]
    
    for a_key, b_type, b_key, c_type, c_key in products:
        a_idx = a_blocks[a_key]
        if b_type == 'top':
            b_idx = b_blocks_top[b_key]
        else:
            b_idx = b_blocks_bottom[b_key]
        if c_type == 'top':
            c_idx = c_blocks_top[c_key]
        else:
            c_idx = c_blocks_bottom[c_key]
        
        col = _strassen_222(a_idx, b_idx, c_idx, U, V, W, col)
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based decomposition
    U, V, W = _strassen_4x4x6()
    
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                rounded = round(val * 2) / 2
                r.append(rounded)
            result.append(r)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}

# EVOLVE-BLOCK-END
