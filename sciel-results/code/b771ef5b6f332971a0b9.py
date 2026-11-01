"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target rank: 6 blocks of 2 (Strassen, rank 7) + 1 block of 1 (Schoolbook, rank 4)
    # Total rank = 6 * 7 + 4 = 46
    rank = 46
    U = np.zeros((nm, rank), dtype=np.int64)
    V = np.zeros((mp, rank), dtype=np.int64)
    W = np.zeros((pn, rank), dtype=np.int64)
    
    col = 0
    
    # Strassen formulas for 2x2 x 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+c)h
    # M6 = (c-a)(e+g)
    # M7 = (b-d)(f+h)
    
    # Indices for A (U): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (V): j*p + k
    # Indices for C (W): k*n + i
    
    # Define the 7 Strassen terms as (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs is a dict {idx: coeff}
    # v_coeffs is a dict {idx: coeff}
    # w_coeffs is a dict {idx: coeff}
    
    strassen_terms = [
        # M1: (a+d)(e+h)
        # C00 += M1, C01 += 0, C10 += 0, C11 += M1 ? No.
        # Reconstruction:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M5 + M7
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add contributions to W for each C index.
        # Term 1: M1
        # U: a(1), d(1) -> {0:1, 3:1}
        # V: e(1), h(1) -> {0:1, 3:1} (relative to block, e=B00, h=B11)
        # W: C00(1), C11(1) -> {0:1, 3:1} (relative to block, C00, C11)
        {'u': {0: 1, 3: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}},
        
        # Term 2: M2
        # U: c(1), d(1) -> {2:1, 3:1}
        # V: e(1) -> {0:1}
        # W: C10(1) -> {2:1}
        {'u': {2: 1, 3: 1}, 'v': {0: 1}, 'w': {2: 1}},
        
        # Term 3: M3
        # U: a(1) -> {0:1}
        # V: g(1), h(-1) -> {2:1, 3:-1}
        # W: C11(1) -> {3:1}
        {'u': {0: 1}, 'v': {2: 1, 3: -1}, 'w': {3: 1}},
        
        # Term 4: M4
        # U: d(1) -> {3:1}
        # V: f(1), e(-1) -> {1:1, 0:-1}
        # W: C00(1) -> {0:1}
        {'u': {3: 1}, 'v': {1: 1, 0: -1}, 'w': {0: 1}},
        
        # Term 5: M5
        # U: a(1), c(1) -> {0:1, 2:1}
        # V: h(1) -> {3:1}
        # W: C01(1), C10(-1) -> {1:1, 2:-1}
        {'u': {0: 1, 2: 1}, 'v': {3: 1}, 'w': {1: 1, 2: -1}},
        
        # Term 6: M6
        # U: c(1), a(-1) -> {2:1, 0:-1}
        # V: e(1), g(1) -> {0:1, 2:1}
        # W: C10(1), C11(1) -> {2:1, 3:1}
        {'u': {2: 1, 0: -1}, 'v': {0: 1, 2: 1}, 'w': {2: 1, 3: 1}},
        
        # Term 7: M7
        # U: b(1), d(-1) -> {1:1, 3:-1}
        # V: f(1), h(1) -> {1:1, 3:1}
        # W: C01(1), C00(-1) -> {1:1, 0:-1}
        {'u': {1: 1, 3: -1}, 'v': {1: 1, 3: 1}, 'w': {1: 1, 0: -1}}
    ]
    
    # Process 6 blocks of size 2
    for block_idx in range(6):
        k_start = block_idx * 2
        # For each of the 7 Strassen terms
        for term in strassen_terms:
            # Update U
            for u_idx, coeff in term['u'].items():
                U[u_idx, col] += coeff
                
            # Update V
            # V index: j*p + k
            # Block k indices: k_start, k_start+1
            # V local indices: 0:(0,k0), 1:(0,k1), 2:(1,k0), 3:(1,k1)
            # Mapping:
            # 0 -> 0*p + k_start
            # 1 -> 0*p + (k_start+1)
            # 2 -> 1*p + k_start
            # 3 -> 1*p + (k_start+1)
            for v_idx, coeff in term['v'].items():
                j = v_idx // 2
                k_offset = v_idx % 2
                global_v_idx = j * p + (k_start + k_offset)
                V[global_v_idx, col] += coeff
                
            # Update W
            # W index: k*n + i
            # Block k indices: k_start, k_start+1
            # W local indices: 0:(k0,0), 1:(k1,0), 2:(k0,1), 3:(k1,1)
            # Mapping:
            # 0 -> k_start*n + 0
            # 1 -> (k_start+1)*n + 0
            # 2 -> k_start*n + 1
            # 3 -> (k_start+1)*n + 1
            for w_idx, coeff in term['w'].items():
                k_offset = w_idx // 2
                i = w_idx % 2
                global_w_idx = (k_start + k_offset) * n + i
                W[global_w_idx, col] += coeff
                
            col += 1
            
    # Process 1 block of size 1 (k=12)
    k_start = 12
    # Schoolbook for 2x2 x 2x1
    # C0 = A00 B0 + A01 B1
    # C1 = A10 B0 + A11 B1
    # 4 terms
    schoolbook_terms = [
        {'u': {0: 1}, 'v': {0: 1}, 'w': {0: 1}}, # A00 B0 -> C0
        {'u': {1: 1}, 'v': {1: 1}, 'w': {0: 1}}, # A01 B1 -> C0
        {'u': {2: 1}, 'v': {0: 1}, 'w': {1: 1}}, # A10 B0 -> C1
        {'u': {3: 1}, 'v': {1: 1}, 'w': {1: 1}}, # A11 B1 -> C1
    ]
    
    for term in schoolbook_terms:
        for u_idx, coeff in term['u'].items():
            U[u_idx, col] += coeff
            
        for v_idx, coeff in term['v'].items():
            j = v_idx // 2
            k_offset = v_idx % 2
            global_v_idx = j * p + (k_start + k_offset)
            V[global_v_idx, col] += coeff
            
        for w_idx, coeff in term['w'].items():
            k_offset = w_idx // 2
            i = w_idx % 2
            global_w_idx = (k_start + k_offset) * n + i
            W[global_w_idx, col] += coeff
            
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
