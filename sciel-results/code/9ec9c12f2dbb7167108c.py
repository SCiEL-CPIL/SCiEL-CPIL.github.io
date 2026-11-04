import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7
    
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append({'U': u_entries, 'V': v_entries, 'W': w_entries})

    # Define the 6 disjoint 2x2x2 blocks
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    blocks = []
    for i_off in [0]:
        for j_off in [0, 2]:
            for k_off in [0, 2, 4]:
                blocks.append((i_off, j_off, k_off))

    # Strassen vectors for 2x2x2
    # Local indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # U, V, W are lists of 4 entries
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4 (Corrected: A11 * (B10 - B00))
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), 
        # Note: V for M4 is (B10 - B00) -> indices (1,0) and (0,0).
        # Local V indices: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
        # So B10 is idx 2, B00 is idx 0.
        # V = [ -1, 0, 1, 0 ].
        # W for M4 contributes to C00 (1) and C10 (1).
        # C00 is local (0,0) -> W idx 0.
        # C10 is local (1,0) -> W idx 2.
        # So W = [1, 0, 1, 0].
        # U for M4 is A11 -> local (1,1) -> U idx 3.
        # U = [0, 0, 0, 1].
        
        # Let's re-verify M4 vector:
        # M4 = A11 * (B10 - B00)
        # U: A11 -> [0,0,0,1]
        # V: B10 - B00 -> [ -1, 0, 1, 0 ]
        # W: C00 + C10 -> [1, 0, 1, 0]
        
        # My previous M4 in the list above was:
        # ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0])
        # This V is [0,0,1,-1] which is B10 - B11? No, B10 is idx 2, B11 is idx 3.
        # So [0,0,1,-1] is B10 - B11.
        # I need B10 - B00.
        # B10 is idx 2. B00 is idx 0.
        # So V should be [-1, 0, 1, 0].
        
        # Let's fix the list.
    ]
    
    # Corrected Strassen Terms
    # 0: M1
    # 1: M2
    # 2: M3
    # 3: M4
    # 4: M5
    # 5: M6
    # 6: M7
    
    strassen_vectors = [
        # M1: (A00+A11)(B00+B11)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3: A00(B10-B11)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4: A11(B10-B00)
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5: (A00+A01)B11
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
        # M6: (A10-A00)(B00+B01)
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    used_indices = set()
    
    for i_off, j_off, k_off in blocks:
        for u_vec, v_vec, w_vec in strassen_vectors:
            u_entries = {}
            v_entries = {}
            w_entries = {}
            
            for li in range(2):
                for lj in range(2):
                    val = u_vec[li*2 + lj]
                    if val != 0:
                        gi = i_off + li
                        gj = j_off + lj
                        u_entries[gi * m + gj] = val
                        used_indices.add((gi, gj, k_off)) # Placeholder, will refine below
            
            for lj in range(2):
                for lk in range(2):
                    val = v_vec[lj*2 + lk]
                    if val != 0:
                        gj = j_off + lj
                        gk = k_off + lk
                        v_entries[gj * p + gk] = val
            
            for lk in range(2):
                for li in range(2):
                    val = w_vec[lk*2 + li]
                    if val != 0:
                        gk = k_off + lk
                        gi = i_off + li
                        w_entries[gk * n + gi] = val
                        used_indices.add((gi, j_off, gk)) # Placeholder
            
            add_term(u_entries, v_entries, w_entries)
            
            # Mark all (i,j,k) in this block as used
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        used_indices.add((i_off + li, j_off + lj, k_off + lk))

    # Add schoolbook terms for remaining indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in used_indices:
                    add_term(
                        {i * m + j: 1},
                        {j * p + k: 1},
                        {k * n + i: 1}
                    )

    rank = len(terms)
    
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    for idx, term in enumerate(terms):
        for u_idx, val in term['U'].items():
            U[u_idx][idx] = val
        for v_idx, val in term['V'].items():
            V[v_idx][idx] = val
        for w_idx, val in term['W'].items():
            W[w_idx][idx] = val
            
    return {'U': U, 'V': V, 'W': W}
