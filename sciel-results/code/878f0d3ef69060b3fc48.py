"""Tensor decomposition for 4x4 x 4x6 matrix multiplication.

Uses Strassen for 4x4*4x4 (rank 49) + schoolbook for 4x4*4x2 (rank 32) = rank 81.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 6]


def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' factor matrices.
    
    U: shape (16, rank) - for A (4x4)
    V: shape (24, rank) - for B (4x6)
    W: shape (24, rank) - for C (4x6), with W[k*n+i] convention
    """
    n, m, p = 4, 4, 6
    rank = 81  # 49 + 32
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index helpers
    def A_idx(i, j):
        return i * 4 + j
    
    def B_idx(j, k):
        return j * 6 + k
    
    def W_idx(i, k):
        return k * 4 + i
    
    col = 0
    
    # PART 1: 4x4 * 4x4 (B cols 0-3, C cols 0-3) using Strassen: rank 49
    # A blocks
    A11 = [A_idx(0,0), A_idx(0,1), A_idx(1,0), A_idx(1,1)]
    A12 = [A_idx(0,2), A_idx(0,3), A_idx(1,2), A_idx(1,3)]
    A21 = [A_idx(2,0), A_idx(2,1), A_idx(3,0), A_idx(3,1)]
    A22 = [A_idx(2,2), A_idx(2,3), A_idx(3,2), A_idx(3,3)]
    
    # B blocks (cols 0-3)
    B11 = [B_idx(0,0), B_idx(0,1), B_idx(1,0), B_idx(1,1)]
    B12 = [B_idx(0,2), B_idx(0,3), B_idx(1,2), B_idx(1,3)]
    B21 = [B_idx(2,0), B_idx(2,1), B_idx(3,0), B_idx(3,1)]
    B22 = [B_idx(2,2), B_idx(2,3), B_idx(3,2), B_idx(3,3)]
    
    # C blocks (cols 0-3)
    C11 = [W_idx(0,0), W_idx(0,1), W_idx(1,0), W_idx(1,1)]
    C12 = [W_idx(0,2), W_idx(0,3), W_idx(1,2), W_idx(1,3)]
    C21 = [W_idx(2,0), W_idx(2,1), W_idx(3,0), W_idx(3,1)]
    C22 = [W_idx(2,2), W_idx(2,3), W_idx(3,2), W_idx(3,3)]
    
    # Define linear combination maps for each Strassen mult
    def make_lc(blocks_signs):
        """blocks_signs: list of (block, sign). Returns dict (r,c) -> [(idx, coeff)]"""
        result = {}
        for r in range(2):
            for c in range(2):
                entries = []
                for block, sign in blocks_signs:
                    idx = block[r*2+c]
                    entries.append((idx, sign))
                result[(r,c)] = entries
        return result
    
    # The 7 Strassen mults
    # Each: (a_map, b_map, c_targets)
    # c_targets: dict (r,c) -> [(w_idx, coeff)]
    
    def make_c_targets(targets):
        """targets: list of (block, sign). Returns dict (r,c) -> [(w_idx, coeff)]"""
        result = {}
        for r in range(2):
            for c in range(2):
                entries = []
                for block, sign in targets:
                    idx = block[r*2+c]
                    entries.append((idx, sign))
                result[(r,c)] = entries
        return result
    
    strassen_mults = [
        # M1 = (A11+A22)*(B11+B22)
        (make_lc([(A11,1),(A22,1)]), make_lc([(B11,1),(B22,1)]),
         make_c_targets([(C11,1),(C12,-1),(C21,-1),(C22,1)])),
        # M2 = (A21+A22)*B11
        (make_lc([(A21,1),(A22,1)]), make_lc([(B11,1)]),
         make_c_targets([(C21,1),(C22,-1)])),
        # M3 = A11*(B21-B22)
        (make_lc([(A11,1)]), make_lc([(B21,1),(B22,-1)]),
         make_c_targets([(C12,1),(C22,-1)])),
        # M4 = A22*(B12-B11)
        (make_lc([(A22,1)]), make_lc([(B12,1),(B11,-1)]),
         make_c_targets([(C11,1),(C22,-1)])),
        # M5 = (A11+A12)*B22
        (make_lc([(A11,1),(A12,1)]), make_lc([(B22,1)]),
         make_c_targets([(C12,-1),(C22,1)])),
        # M6 = (A21-A11)*(B11+B12)
        (make_lc([(A21,1),(A11,-1)]), make_lc([(B11,1),(B12,1)]),
         make_c_targets([(C22,1)])),
        # M7 = (A12-A22)*(B21+B22)
        (make_lc([(A12,1),(A22,-1)]), make_lc([(B21,1),(B22,1)]),
         make_c_targets([(C11,1)])),
    ]
    
    # Apply Strassen to each mult
    for a_map, b_map, c_targets in strassen_mults:
        # 7 rank-1 terms for 2x2*2x2 Strassen
        # Each term: (a_positions, b_positions, c_contrib)
        # c_contrib: dict (r,c) -> coeff
        
        terms = [
            # T1: (a00+a11)*(b00+b11)
            ([(0,0),(1,1)], [(0,0),(1,1)], {(0,0):1, (1,1):1, (0,1):-1, (1,0):-1}),
            # T2: (a10+a11)*b00
            ([(1,0),(1,1)], [(0,0)], {(1,0):1, (1,1):-1}),
            # T3: a00*(b10-b11)
            ([(0,0)], [(1,0)], {(0,1):1, (1,1):-1}),
            ([(0,0)], [(1,1)], {(0,1):-1, (1,1):1}),
            # T4: a11*(b01-b00)
            ([(1,1)], [(0,1)], {(0,0):1, (1,1):-1}),
            ([(1,1)], [(0,0)], {(0,0):-1, (1,1):1}),
            # T5: (a00+a01)*b11
            ([(0,0),(0,1)], [(1,1)], {(0,1):-1, (1,1):1}),
            # T6: (a10-a00)*(b00+b01)
            ([(1,0)], [(0,0),(0,1)], {(1,1):1}),
            ([(0,0)], [(0,0),(0,1)], {(1,1):-1}),
            # T7: (a01-a11)*(b10+b11)
            ([(0,1)], [(1,0),(1,1)], {(0,0):1}),
            ([(1,1)], [(1,0),(1,1)], {(0,0):-1}),
        ]
        
        for a_pos, b_pos, c_contrib in terms:
            # Expand linear combinations
            a_entries = []
            for pos in a_pos:
                for (u_idx, cu) in a_map[pos]:
                    a_entries.append((u_idx, cu))
            
            b_entries = []
            for pos in b_pos:
                for (v_idx, cv) in b_map[pos]:
                    b_entries.append((v_idx, cv))
            
            # For each pair, add a rank-1 term
            for (u_idx, cu) in a_entries:
                for (v_idx, cv) in b_entries:
                    coeff_av = cu * cv
                    if coeff_av == 0:
                        continue
                    # Add to all C targets
                    for (r, c), w_coeff in c_contrib.items():
                        for (w_idx, wc) in c_targets[(r,c)]:
                            total_w = coeff_av * w_coeff * wc
                            if total_w == 0:
                                continue
                            U[u_idx, col] += 1
                            V[v_idx, col] += 1
                            W[w_idx, col] += total_w
                    col += 1
    
    # PART 2: 4x4 * 4x2 (B cols 4-5, C cols 4-5) using schoolbook: rank 32
    for i in range(4):
        for j in range(4):
            for k in range(2):
                k_actual = k + 4
                U[A_idx(i, j), col] = 1
                V[B_idx(j, k_actual), col] = 1
                W[W_idx(i, k_actual), col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
