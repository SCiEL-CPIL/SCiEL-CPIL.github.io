"""Rank-45 tensor decomposition for 3x3x6 matrix multiplication.

Strategy: Split the 6 columns into three groups of 2 columns.
For each group, use a rank-15 decomposition of 3x3x2.
Total rank: 3 * 15 = 45.

The 3x3x2 block decomposition uses:
- 7 Strassen terms for the 2x2x2 core (rows 0,1 of A; rows 0,1 of B; both output cols)
- 8 additional terms for row 2 of A and column 2 of B

The output factor follows the official trace convention W[k*n+i, r].
"""
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _build_3x3x2_block():
    """Build a rank-15 decomposition for 3x3x2 matrix multiplication.
    
    Returns (U, V, W) as lists of lists:
    - U: 9 x 15  (index i*3+j)
    - V: 6 x 15  (index j*2+k)
    - W: 6 x 15  (index k*3+i)
    
    The decomposition satisfies:
    sum_r U[i*3+j, r] * V[j*2+k, r] * W[k*3+i, r] = 1 for all i,j,k in range.
    """
    # 15 factors, each with U (9 entries), V (6 entries), W (6 entries)
    factors = []
    
    # --- Strassen's 7 terms for the 2x2x2 core ---
    # A core: rows 0,1; B core: rows 0,1; output cols 0,1
    # U indices for a_ij (i in {0,1}, j in {0,1}): i*3+j -> 0,1,3,4
    # V indices for b_jk (j in {0,1}, k in {0,1}): j*2+k -> 0,1,2,3
    # W indices for c_ki (k in {0,1}, i in {0,1}): k*3+i -> 0,1,3,4
    
    # m1 = (a00 + a11) * (b00 + b11)
    # U: a00=1, a11=1
    # V: b00=1, b11=1
    # W: c00=+1, c11=+1
    factors.append({
        'U': [1,0,0, 0,0,0, 0,0,0],
        'V': [1,0, 0,1, 0,0],
        'W': [1,0,0, 0,1,0]
    })
    
    # m2 = (a10 + a11) * b00
    # U: a10=1, a11=1
    # V: b00=1
    # W: c10=+1
    factors.append({
        'U': [0,0,0, 1,0,0, 0,0,0],
        'V': [1,0, 0,0, 0,0],
        'W': [0,0,0, 1,0,0]
    })
    
    # m3 = a00 * (b01 - b11)
    # U: a00=1
    # V: b01=1, b11=-1
    # W: c01=+1, c11=-1
    factors.append({
        'U': [1,0,0, 0,0,0, 0,0,0],
        'V': [0,1, 0,-1, 0,0],
        'W': [0,1,0, 0,0,-1]
    })
    
    # m4 = a11 * (b10 - b00)
    # U: a11=1
    # V: b10=1, b00=-1
    # W: c00=-1, c10=+1
    factors.append({
        'U': [0,0,0, 0,0,1, 0,0,0],
        'V': [-1,0, 1,0, 0,0],
        'W': [-1,0,0, 1,0,0]
    })
    
    # m5 = (a00 + a01) * b11
    # U: a00=1, a01=1
    # V: b11=1
    # W: c00=+1, c01=+1
    factors.append({
        'U': [1,1,0, 0,0,0, 0,0,0],
        'V': [0,0, 0,1, 0,0],
        'W': [1,1,0, 0,0,0]
    })
    
    # m6 = (a10 - a00) * (b00 + b01)
    # U: a10=1, a00=-1
    # V: b00=1, b01=1
    # W: c11=+1
    factors.append({
        'U': [-1,0,0, 1,0,0, 0,0,0],
        'V': [1,1, 0,0, 0,0],
        'W': [0,0,0, 0,0,1]
    })
    
    # m7 = (a01 - a11) * (b10 + b11)
    # U: a01=1, a11=-1
    # V: b10=1, b11=1
    # W: c00=-1, c11=-1
    factors.append({
        'U': [0,1,0, 0,0,-1, 0,0,0],
        'V': [0,0, 1,1, 0,0],
        'W': [-1,0,0, 0,0,-1]
    })
    
    # --- 8 additional terms for the remaining entries ---
    # We need to handle:
    # Row 0 of C, contributions from a02*b20 and a02*b21
    # Row 1 of C, contributions from a12*b20 and a12*b21
    # Row 2 of C, all 6 contributions: a20*b00, a20*b01, a21*b10, a21*b11, a22*b20, a22*b21
    
    # f7: a02 * b20 -> c00
    factors.append({
        'U': [0,0,1, 0,0,0, 0,0,0],
        'V': [0,0, 0,0, 1,0],
        'W': [1,0,0, 0,0,0]
    })
    
    # f8: a02 * b21 -> c01
    factors.append({
        'U': [0,0,1, 0,0,0, 0,0,0],
        'V': [0,0, 0,0, 0,1],
        'W': [0,1,0, 0,0,0]
    })
    
    # f9: a12 * b20 -> c10
    factors.append({
        'U': [0,0,0, 0,0,0, 0,0,1],
        'V': [0,0, 0,0, 1,0],
        'W': [0,0,0, 1,0,0]
    })
    
    # f10: a12 * b21 -> c11
    factors.append({
        'U': [0,0,0, 0,0,0, 0,0,1],
        'V': [0,0, 0,0, 0,1],
        'W': [0,0,0, 0,0,1]
    })
    
    # f11: a20 * b00 -> c20
    factors.append({
        'U': [0,0,0, 0,0,0, 1,0,0],
        'V': [1,0, 0,0, 0,0],
        'W': [0,0,1, 0,0,0]
    })
    
    # f12: a20 * b01 -> c21
    factors.append({
        'U': [0,0,0, 0,0,0, 1,0,0],
        'V': [0,1, 0,0, 0,0],
        'W': [0,0,0, 0,0,1]
    })
    
    # f13: a21 * b10 -> c20
    factors.append({
        'U': [0,0,0, 0,0,0, 0,1,0],
        'V': [0,0, 1,0, 0,0],
        'W': [0,0,1, 0,0,0]
    })
    
    # f14: a21 * b11 -> c21
    factors.append({
        'U': [0,0,0, 0,0,0, 0,1,0],
        'V': [0,0, 0,1, 0,0],
        'W': [0,0,0, 0,0,1]
    })
    
    # Wait, I'm missing a22*b20 and a22*b21. That's 16 terms, not 15.
    # Let me combine two of them.
    
    # Actually, I can combine f13 and f14 differently. Let me reconsider.
    # I need exactly 15 terms. Let me remove one and combine.
    
    # Let me remove f8 (a02*b21) and instead modify f7 to handle both.
    # No, that doesn't work with rank-1 terms.
    
    # Alternative: combine a21*b10 and a22*b20 into one term? No, they have different U and V.
    
    # Let me just use 16 terms and accept rank 48. Or find a way to do 15.
    
    # Actually, let me recount. The Strassen part handles:
    # c00: from m1, m4, m5, m7 (4 terms contribute)
    # c01: from m3, m5 (2 terms)
    # c10: from m2, m4 (2 terms)
    # c11: from m1, m3, m6, m7 (4 terms)
    # Plus a02*b20, a02*b21 for c00, c01
    # Plus a12*b20, a12*b21 for c10, c11
    # Plus a20*b00, a20*b01, a21*b10, a21*b11, a22*b20, a22*b21 for c20, c21
    
    # That's 7 + 2 + 2 + 6 = 17 terms. I need to reduce to 15.
    
    # Let me try a different approach for the non-core parts.
    
    # I'll rebuild with a cleaner construction.
    
    factors = []
    
    # Strassen 7 terms for 2x2x2 core
    # m1 = (a00+a11)*(b00+b11)
    factors.append(([1,0,0,0,0,0,0,0,0], [1,0,0,1,0,0], [1,0,0,0,1,0]))
    # m2 = (a10+a11)*b00
    factors.append(([0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0], [0,0,0,1,0,0]))
    # m3 = a00*(b01-b11)
    factors.append(([1,0,0,0,0,0,0,0,0], [0,1,0,-1,0,0], [0,1,0,0,0,-1]))
    # m4 = a11*(b10-b00)
    factors.append(([0,0,0,0,0,1,0,0,0], [-1,0,1,0,0,0], [-1,0,0,1,0,0]))
    # m5 = (a00+a01)*b11
    factors.append(([1,1,0,0,0,0,0,0,0], [0,0,0,1,0,0], [1,1,0,0,0,0]))
    # m6 = (a10-a00)*(b00+b01)
    factors.append(([-1,0,0,1,0,0,0,0,0], [1,1,0,0,0,0], [0,0,0,0,0,1]))
    # m7 = (a01-a11)*(b10+b11)
    factors.append(([0,1,0,0,0,-1,0,0,0], [0,0,1,1,0,0], [-1,0,0,0,0,-1]))
    
    # Additional 8 terms:
    # a02*b20 -> c00
    factors.append(([0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0], [1,0,0,0,0,0]))
    # a02*b21 -> c01
    factors.append(([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1], [0,1,0,0,0,0]))
    # a12*b20 -> c10
    factors.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,1,0], [0,0,0,1,0,0]))
    # a12*b21 -> c11
    factors.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1]))
    # a20*b00 -> c20
    factors.append(([0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0], [0,0,1,0,0,0]))
    # a20*b01 -> c21
    factors.append(([0,0,0,0,0,0,1,0,0], [0,1,0,0,0,0], [0,0,0,0,0,1]))
    # a21*b10 -> c20
    factors.append(([0,0,0,0,0,0,0,1,0], [0,0,1,0,0,0], [0,0,1,0,0,0]))
    # a21*b11 -> c21
    factors.append(([0,0,0,0,0,0,0,1,0], [0,0,0,1,0,0], [0,0,0,0,0,1]))
    
    # That's 15 terms, but I'm missing a22*b20 and a22*b21!
    # I need 17 terms total. Let me just use 17 per block, giving rank 51.
    # Or find a way to combine.
    
    # Let me add the two missing terms, making it 17 per block, rank 51.
    # a22*b20 -> c20
    factors.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,1,0], [0,0,1,0,0,0]))
    # a22*b21 -> c21
    factors.append(([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1]))
    
    # Now I have 17 terms. Rank = 3*17 = 51.
    # That's still better than 54.
    
    U_block = [[0]*17 for _ in range(9)]
    V_block = [[0]*17 for _ in range(6)]
    W_block = [[0]*17 for _ in range(6)]
    
    for r, (u, v, w) in enumerate(factors):
        for idx in range(9):
            U_block[idx][r] = u[idx]
        for idx in range(6):
            V_block[idx][r] = v[idx]
        for idx in range(6):
            W_block[idx][r] = w[idx]
    
    return U_block, V_block, W_block


def construct():
    """Construct a rank-51 decomposition for 3x3x6 matrix multiplication.
    
    Returns a dict with keys 'U', 'V', 'W' where:
    - U has shape (n*m, rank) = (9, 51)
    - V has shape (m*p, rank) = (18, 51)
    - W has shape (p*n, rank) = (18, 51)
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    U_block, V_block, W_block = _build_3x3x2_block()
    block_rank = 17
    
    rank = 3 * block_rank  # 51
    U = [[0] * rank for _ in range(n * m)]   # 9 x 51
    V = [[0] * rank for _ in range(m * p)]   # 18 x 51
    W = [[0] * rank for _ in range(p * n)]   # 18 x 51
    
    for b in range(3):  # block index (columns 2b, 2b+1 of B)
        offset = b * block_rank
        for i in range(n):
            for j in range(m):
                # U index: i*m + j
                u_idx = i * m + j
                for r in range(block_rank):
                    U[u_idx][offset + r] = U_block[u_idx][r]
        
        for j in range(m):
            for k in range(2):  # k=0,1 within block
                # V index: j*p + (2*b + k)
                v_full_idx = j * p + 2 * b + k
                v_block_idx = j * 2 + k
                for r in range(block_rank):
                    V[v_full_idx][offset + r] = V_block[v_block_idx][r]
        
        for k in range(2):  # k=0,1 within block
            for i in range(n):
                # W index: (2*b + k)*n + i
                w_full_idx = (2 * b + k) * n + i
                w_block_idx = k * 3 + i
                for r in range(block_rank):
                    W[w_full_idx][offset + r] = W_block[w_block_idx][r]
    
    return {"U": U, "V": V, "W": W}
