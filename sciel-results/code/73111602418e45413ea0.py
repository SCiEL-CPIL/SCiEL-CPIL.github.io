import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen for 2x2x2 parts
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Each term is a list of (U_coeff, V_coeff, W_coeff)
    # U indices: (i, j_low) where i in 0..1, j_low in 0..1
    # V indices: (j_low, k_low) where j_low in 0..1, k_low in 0..1
    # W indices: (k_low, i) where k_low in 0..1, i in 0..1
    
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    strassen_terms = [
        # M0
        [(0,0,1), (1,1,1)], # U coeffs for A
        [(0,0,1), (1,1,1)], # V coeffs for B
        [(0,0,1), (1,1,1)]  # W coeffs for C (placeholder, will be updated)
    ]
    
    # This manual construction is error-prone. Let's generate the terms programmatically.
    
    def get_strassen_terms():
        # Returns list of 7 terms.
        # Each term is a dict with 'U', 'V', 'W' mappings.
        # U: {(i, j_low): coeff}
        # V: {(j_low, k_low): coeff}
        # W: {(k_low, i): coeff}
        
        terms = []
        
        # M0 = (A00 + A11)(B00 + B11)
        t = {'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {}}
        terms.append(t)
        
        # M1 = (A10 + A11)B00
        t = {'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {}}
        terms.append(t)
        
        # M2 = A00(B01 - B11)
        t = {'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {}}
        terms.append(t)
        
        # M3 = A11(B10 - B00)
        t = {'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {}}
        terms.append(t)
        
        # M4 = (A00 + A01)B11
        t = {'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {}}
        terms.append(t)
        
        # M5 = (A10 - A00)(B00 + B01)
        t = {'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {}}
        terms.append(t)
        
        # M6 = (A01 - A11)(B10 + B11)
        t = {'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {}}
        terms.append(t)
        
        # Now define W contributions for each C element
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5
        
        w_coeffs = [
            # M0
            {(0,0):1, (1,1):1}, # C00, C11
            # M1
            {(0,0):0, (1,0):1, (1,1):-1}, # C10, C11
            # M2
            {(0,1):1, (1,1):1}, # C01, C11
            # M3
            {(0,0):1, (1,0):1}, # C00, C10
            # M4
            {(0,0):-1, (0,1):1}, # C00, C01
            # M5
            {(1,1):1}, # C11
            # M6
            {(0,0):1}, # C00
        ]
        
        # Wait, the W mapping is (k_low, i).
        # C00 is i=0, k=0. Index (0,0).
        # C01 is i=0, k=1. Index (1,0).
        # C10 is i=1, k=0. Index (0,1).
        # C11 is i=1, k=1. Index (1,1).
        
        # Let's re-map W coeffs correctly.
        w_map = [
            # M0: C00 + C11
            {(0,0):1, (1,1):1},
            # M1: C10 - C11
            {(0,1):1, (1,1):-1},
            # M2: C01 + C11
            {(1,0):1, (1,1):1},
            # M3: C00 + C10
            {(0,0):1, (0,1):1},
            # M4: -C00 + C01
            {(0,0):-1, (1,0):1},
            # M5: C11
            {(1,1):1},
            # M6: C00
            {(0,0):1},
        ]
        
        for i, w in enumerate(w_map):
            terms[i]['W'] = w
            
        return terms

    terms = get_strassen_terms()
    
    col = 0
    for j_high in range(2):
        for l in range(4): # 4 blocks of 2 columns each
            for t_idx in range(7):
                term = terms[t_idx]
                
                # Update U
                for (i, j_low), coeff in term['U'].items():
                    j = j_high * 2 + j_low
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                for (j_low, k_low), coeff in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                for (k_low, i), coeff in term['W'].items():
                    k = l * 2 + k_low
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                    
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
