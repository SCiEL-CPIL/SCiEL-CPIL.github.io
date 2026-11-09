import time
import numpy as np

def construct():
    n, m, p = 2, 4, 4
    # Rank 28 decomposition using 4x Strassen (7 terms each)
    # Blocks: (j_off, k_off) in {(0,0), (0,2), (2,0), (2,2)}
    
    U = [[0] * 28 for _ in range(n * m)]
    V = [[0] * 28 for _ in range(m * p)]
    W = [[0] * 28 for _ in range(p * n)]
    
    # Strassen terms for 2x2 matrix multiplication
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
    # Indices for B: 0,1,2,3
    # Indices for C: 0,1,2,3
    #
    # The 7 terms (u, v, w) where u,v,w are sparse vectors of length 4
    # Using 0-indexed flat representation for the 2x2 blocks
    
    # Term 1: (a00+a11)(b00+b11) -> c00(+1), c11(+1)
    # T1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
    # Term 2: (a10+a11)b00 -> c10(+1), c11(-1)
    # T2: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,-1]
    # Term 3: a00(b01-b11) -> c01(+1), c11(+1)
    # T3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1]
    # Term 4: a11(b10-b00) -> c00(+1), c10(+1)
    # T4: U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0]
    # Term 5: (a00+a01)b11 -> c00(-1), c01(+1)
    # T5: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
    # Term 6: (a10-a00)(b00+b01) -> c11(+1)
    # T6: U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    # Term 7: (a01-a11)(b10+b11) -> c00(+1)
    # T7: U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
    
    # Verify Term 6: (a10-a00)(b00+b01) = a10b00 + a10b01 - a00b00 - a00b01
    # W contribution is c11.
    # c11 should be a10b01 + a11b11.
    # Let's check the sum of W contributions for c11 (index 3 in 2x2 C).
    # T1: +1
    # T2: -1
    # T3: +1
    # T4: 0
    # T5: 0
    # T6: +1
    # T7: 0
    # Sum = 1 - 1 + 1 + 1 = 2.
    # This is incorrect. The sum of coefficients for c11 must be 1 for the correct monomials and 0 for others.
    # The Strassen decomposition is correct, but my manual mapping of indices or signs might be off.
    # The provided code in the prompt had a verified Strassen set. I will use that one.
    
    # Verified Strassen terms from prompt:
    strassen_terms = [
        # Term 1
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # Term 2
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        # Term 3
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        # Term 4
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        # Term 5
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        # Term 6
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # Term 7
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]},
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 2},
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            # Add to U
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            
            # Add to V
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            
            # Add to W
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
