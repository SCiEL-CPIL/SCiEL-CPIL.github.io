import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 terms
    # Each term is a tuple (u_vec, v_vec, w_vec)
    # u_vec: indices 0,1,2,3 for A00, A01, A10, A11
    # v_vec: indices 0,1 for B00, B01 (relative to block cols)
    # w_vec: indices 0,1,2,3 for C00, C01, C10, C11 (relative to block cols)
    
    terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 3: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 2: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {1: 1, 3: 1}, {0: -1, 2: 1})
    ]
    
    # Note: v_vec indices 0,1,2,3 correspond to:
    # 0: B_{0, col0}
    # 1: B_{0, col1}
    # 2: B_{1, col0}
    # 3: B_{1, col1}
    # Wait, my previous mapping was:
    # B_{00} -> 0, B_{01} -> 1, B_{10} -> 2, B_{11} -> 3.
    # Let's re-verify the terms with this mapping.
    
    # M1: (A00+A11)(B00+B11). B00 is B_{0,0}, B11 is B_{1,1}.
    # Indices: B_{0,0}=0, B_{1,1}=3. Correct.
    # M2: (A10+A11)B00. B00 is B_{0,0}=0. Correct.
    # M3: A00(B10-B11). B10 is B_{1,0}=2, B11 is B_{1,1}=3. Correct.
    # M4: A11(B01-B00). B01 is B_{0,1}=1, B00 is B_{0,0}=0. Correct.
    # M5: (A00+A01)B11. B11 is B_{1,1}=3. Correct.
    # M6: (A10-A00)(B00+B10). B00 is 0, B10 is 2. Correct.
    # M7: (A01-A11)(B01+B11). B01 is 1, B11 is 3. Correct.
    
    # w_vec indices 0,1,2,3 correspond to:
    # 0: C_{0, col0}
    # 1: C_{0, col1}
    # 2: C_{1, col0}
    # 3: C_{1, col1}
    
    # M1: C00 +1, C11 +1. Indices 0, 3. Correct.
    # M2: C10 +1, C11 -1. Indices 2, 3. Correct.
    # M3: C01 +1, C11 +1. Indices 1, 3. Correct.
    # M4: C00 +1, C11 +1. Indices 0, 3. Correct.
    # M5: C00 -1, C01 +1. Indices 0, 1. Correct.
    # M6: C11 +1. Index 3. Correct.
    # M7: C00 -1, C10 +1. Indices 0, 2. Correct.

    for s in range(3):
        for t in range(7):
            r = s * 7 + t
            u_dict, v_dict, w_dict = terms[t]
            
            # Fill U
            for idx, val in u_dict.items():
                U[idx][r] = val
                
            # Fill V
            # v_dict keys 0,1,2,3 map to B indices
            # 0 -> B_{0, 2s} -> index 0*6 + 2s = 2s
            # 1 -> B_{0, 2s+1} -> index 0*6 + 2s + 1 = 2s + 1
            # 2 -> B_{1, 2s} -> index 1*6 + 2s = 6 + 2s
            # 3 -> B_{1, 2s+1} -> index 1*6 + 2s + 1 = 6 + 2s + 1
            v_map = {
                0: 2 * s,
                1: 2 * s + 1,
                2: 6 + 2 * s,
                3: 6 + 2 * s + 1
            }
            for idx, val in v_dict.items():
                V[v_map[idx]][r] = val
                
            # Fill W
            # w_dict keys 0,1,2,3 map to C indices
            # 0 -> C_{0, 2s} -> index 2s*2 + 0 = 4s
            # 1 -> C_{0, 2s+1} -> index (2s+1)*2 + 0 = 4s + 2
            # 2 -> C_{1, 2s} -> index 2s*2 + 1 = 4s + 1
            # 3 -> C_{1, 2s+1} -> index (2s+1)*2 + 1 = 4s + 3
            w_map = {
                0: 4 * s,
                1: 4 * s + 2,
                2: 4 * s + 1,
                3: 4 * s + 3
            }
            for idx, val in w_dict.items():
                W[w_map[idx]][r] = val
                
    return {"U": U, "V": V, "W": W}
