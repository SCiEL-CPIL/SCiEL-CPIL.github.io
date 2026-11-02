"""Combined approach: structured search for low-rank exact tensor decomposition."""
import numpy as np
import time
import random
import itertools

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # Rank 84 decomposition using Strassen outer (7) * Schoolbook inner (12)
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to map global indices to block local indices
    # A: 4x4, blocks 2x2
    # B: 4x6, blocks 2x3
    # C: 4x6, blocks 2x3
    
    # Define the 7 Strassen steps
    # Each step i has:
    # A_blocks: list of (block_id, coeff)
    # B_blocks: list of (block_id, coeff)
    # C_blocks: list of (block_id, coeff)
    
    # Block IDs:
    # A: 00, 01, 10, 11
    # B: 00, 01, 10, 11
    # C: 00, 01, 10, 11
    
    steps = [
        # M1 = (A00 + A11)(B00 + B11)
        {
            'A': [(0, 1), (3, 1)], # 00, 11
            'B': [(0, 1), (3, 1)], # 00, 11
            'C': [(0, 1), (3, 1)]  # 00, 11
        },
        # M2 = (A10 + A11)B00
        {
            'A': [(2, 1), (3, 1)], # 10, 11
            'B': [(0, 1)],         # 00
            'C': [(2, 1), (3, -1)] # 10, 11
        },
        # M3 = A00(B01 - B11)
        {
            'A': [(0, 1)],         # 00
            'B': [(1, 1), (3, -1)],# 01, 11
            'C': [(1, 1), (3, 1)]  # 01, 11
        },
        # M4 = A11(B10 - B00)
        {
            'A': [(3, 1)],         # 11
            'B': [(2, 1), (0, -1)],# 10, 00
            'C': [(0, 1), (2, 1)]  # 00, 10
        },
        # M5 = (A00 + A01)B11
        {
            'A': [(0, 1), (1, 1)], # 00, 01
            'B': [(3, 1)],         # 11
            'C': [(0, -1), (1, 1)] # 00, 01
        },
        # M6 = (A10 - A00)(B00 + B01)
        {
            'A': [(2, 1), (0, -1)],# 10, 00
            'B': [(0, 1), (1, 1)], # 00, 01
            'C': [(3, 1)]          # 11
        },
        # M7 = (A01 - A11)(B10 + B11)
        {
            'A': [(1, 1), (3, -1)],# 01, 11
            'B': [(2, 1), (3, 1)], # 10, 11
            'C': [(0, 1)]          # 00
        }
    ]
    
    # Helper to get global indices for a block
    def get_A_indices(block_id):
        # block_id: 00, 01, 10, 11
        # 00: I in 0,1; J in 0,1
        # 01: I in 0,1; J in 2,3
        # 10: I in 2,3; J in 0,1
        # 11: I in 2,3; J in 2,3
        if block_id == 0: return [(i, j) for i in range(2) for j in range(2)]
        if block_id == 1: return [(i, j+2) for i in range(2) for j in range(2)]
        if block_id == 2: return [(i+2, j) for i in range(2) for j in range(2)]
        if block_id == 3: return [(i+2, j+2) for i in range(2) for j in range(2)]
        
    def get_B_indices(block_id):
        # block_id: 00, 01, 10, 11
        # 00: J in 0,1; K in 0,1,2
        # 01: J in 0,1; K in 3,4,5
        # 10: J in 2,3; K in 0,1,2
        # 11: J in 2,3; K in 3,4,5
        if block_id == 0: return [(j, k) for j in range(2) for k in range(3)]
        if block_id == 1: return [(j, k+3) for j in range(2) for k in range(3)]
        if block_id == 2: return [(j+2, k) for j in range(2) for k in range(3)]
        if block_id == 3: return [(j+2, k+3) for j in range(2) for k in range(3)]
        
    def get_C_indices(block_id):
        # block_id: 00, 01, 10, 11
        # 00: I in 0,1; K in 0,1,2
        # 01: I in 0,1; K in 3,4,5
        # 10: I in 2,3; K in 0,1,2
        # 11: I in 2,3; K in 3,4,5
        if block_id == 0: return [(i, k) for i in range(2) for k in range(3)]
        if block_id == 1: return [(i, k+3) for i in range(2) for k in range(3)]
        if block_id == 2: return [(i+2, k) for i in range(2) for k in range(3)]
        if block_id == 3: return [(i+2, k+3) for i in range(2) for k in range(3)]

    # Inner Schoolbook for 2x2 * 2x3
    # A_inner: 2x2 (indices r, s)
    # B_inner: 2x3 (indices s, k)
    # C_inner: 2x3 (indices r, k)
    # Rank 12
    # Term c_in = r*3 + s*? No.
    # Sum over s in 0..1, k in 0..2? No.
    # Schoolbook: C[r, k] = sum_s A[r, s] B[s, k]
    # So for each r, s, k:
    # U[r, s] = 1, V[s, k] = 1, W[r, k] = 1?
    # Wait, W index is k*n + r = k*2 + r.
    
    # Let's iterate through the 7 steps
    for i, step in enumerate(steps):
        # For each schoolbook term in this step
        # r in 0..1, s in 0..1, k in 0..2
        for r in range(2):
            for s in range(2):
                for k in range(3):
                    c_global = i * 12 + (r * 3 + s * 1 + k * 0) # Unique ID for this term?
                    # We need to map (r, s, k) to a unique c_in in 0..11
                    # Let c_in = r * 6 + s * 3 + k? 
                    # There are 2*2*3 = 12 terms.
                    c_in = r * 6 + s * 3 + k
                    c_global = i * 12 + c_in
                    
                    # Determine U contributions
                    # A_inner[r, s] is sum of A_global[I, J] * coeff
                    # A_inner is formed from blocks in step['A']
                    # A_inner[r, s] corresponds to local (r, s) in the block
                    # So for each block_id, coeff in step['A']:
                    # The local (r, s) maps to global (I, J) via get_A_indices
                    # But get_A_indices returns a list of all (I, J) in the block.
                    # We need the specific (I, J) corresponding to local (r, s).
                    
                    # Let's create a mapping from (block_id, r, s) to (I, J)
                    # A block is 2x2.
                    # block 00: I=r, J=s
                    # block 01: I=r, J=s+2
                    # block 10: I=r+2, J=s
                    # block 11: I=r+2, J=s+2
                    
                    for block_id, coeff in step['A']:
                        if block_id == 0: I, J = r, s
                        elif block_id == 1: I, J = r, s+2
                        elif block_id == 2: I, J = r+2, s
                        elif block_id == 3: I, J = r+2, s+2
                        else: continue
                        
                        u_idx = I * 4 + J
                        U[u_idx][c_global] += coeff
                        
                    # Determine V contributions
                    # B_inner[s, k] is sum of B_global[J, K] * coeff
                    # B block is 2x3.
                    # block 00: J=s, K=k
                    # block 01: J=s, K=k+3
                    # block 10: J=s+2, K=k
                    # block 11: J=s+2, K=k+3
                    
                    for block_id, coeff in step['B']:
                        if block_id == 0: J, K = s, k
                        elif block_id == 1: J, K = s, k+3
                        elif block_id == 2: J, K = s+2, k
                        elif block_id == 3: J, K = s+2, k+3
                        else: continue
                        
                        v_idx = J * 6 + K
                        V[v_idx][c_global] += coeff
                        
                    # Determine W contributions
                    # C_inner[r, k] contributes to C_global[I, K] * coeff
                    # C block is 2x3.
                    # block 00: I=r, K=k
                    # block 01: I=r, K=k+3
                    # block 10: I=r+2, K=k
                    # block 11: I=r+2, K=k+3
                    
                    for block_id, coeff in step['C']:
                        if block_id == 0: I, K = r, k
                        elif block_id == 1: I, K = r, k+3
                        elif block_id == 2: I, K = r+2, k
                        elif block_id == 3: I, K = r+2, k+3
                        else: continue
                        
                        w_idx = K * 4 + I
                        W[w_idx][c_global] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
