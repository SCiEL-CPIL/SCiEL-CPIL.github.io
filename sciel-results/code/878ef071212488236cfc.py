import numpy as np

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A * B
    #
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B00)
    # M4 = A11 * (B01 - B11)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (U): i*m + j
    # A00: 0, A01: 1, A10: 2, A11: 3
    
    # Indices for B (V): j*p + k
    # For a specific block b (0, 1, 2), the columns are 2*b, 2*b+1
    # B00 (block b): row 0, col 2*b -> index 0*p + 2*b = 2*b
    # B01 (block b): row 0, col 2*b+1 -> index 0*p + 2*b+1 = 2*b+1
    # B10 (block b): row 1, col 2*b -> index 1*p + 2*b = 6 + 2*b
    # B11 (block b): row 1, col 2*b+1 -> index 1*p + 2*b+1 = 6 + 2*b+1
    
    # Indices for C (W): k*n + i
    # For a specific block b, the columns are 2*b, 2*b+1
    # C00 (block b): col 2*b, row 0 -> index 2*b*n + 0 = 4*b
    # C01 (block b): col 2*b+1, row 0 -> index (2*b+1)*n + 0 = 4*b + 2
    # C10 (block b): col 2*b, row 1 -> index 2*b*n + 1 = 4*b + 1
    # C11 (block b): col 2*b+1, row 1 -> index (2*b+1)*n + 1 = 4*b + 3
    
    strassen_terms = [
        # (U coeffs, V coeffs, W coeffs)
        # M1 = (A00 + A11) * (B00 + B11)
        ({0: 1, 3: 1}, {0: 1, 1: 1}, {0: 1, 1: 1}), # Note: V/W indices are local to block
        # M2 = (A10 + A11) * B00
        ({2: 1, 3: 1}, {0: 1}, {1: 1}), # C10 gets M2? No.
        # Let's re-verify C indices for M2.
        # C10 = M2 + M4.
        # C10 is row 1, col 0.
        # In local block indices:
        # C00 (r0, c0), C01 (r0, c1), C10 (r1, c0), C11 (r1, c1).
        # M2 contributes to C10.
        # So W local index for C10 is 1 (if C10 is index 1 in local 0..3? No, usually row-major: C00=0, C01=1, C10=2, C11=3? Or C00=0, C10=1, C01=2, C11=3?)
        # Task says W[k*n+i].
        # Local k=0, i=1 -> index 1.
        # Local k=1, i=0 -> index 2.
        # Local k=0, i=0 -> index 0.
        # Local k=1, i=1 -> index 3.
        # So:
        # C00 (k=0, i=0) -> 0
        # C10 (k=0, i=1) -> 1
        # C01 (k=1, i=0) -> 2
        # C11 (k=1, i=1) -> 3
        
        # Let's map Strassen terms to these local W indices.
        # M1 -> C00 (0), C11 (3). C00 = M1 + ... C11 = M1 - ...
        # So M1 has W coeffs {0: 1, 3: 1}.
        
        # M2 -> C10 (1).
        # So M2 has W coeffs {1: 1}.
        
        # M3 -> C01 (2), C11 (3).
        # C01 = M3 + M5. C11 = ... + M3 + ...
        # So M3 has W coeffs {2: 1, 3: 1}.
        
        # M4 -> C00 (0), C10 (1).
        # C00 = ... + M4 ... C10 = M2 + M4.
        # So M4 has W coeffs {0: 1, 1: 1}.
        
        # M5 -> C00 (0), C01 (2).
        # C00 = ... - M5 ... C01 = M3 + M5.
        # So M5 has W coeffs {0: -1, 2: 1}.
        
        # M6 -> C11 (3).
        # C11 = ... + M6.
        # So M6 has W coeffs {3: 1}.
        
        # M7 -> C00 (0), C11 (3).
        # C00 = ... + M7. C11 = ... + M7.
        # So M7 has W coeffs {0: 1, 3: 1}.
        
        # Let's re-list U and V coeffs.
        # M1: U {0:1, 3:1}, V {0:1, 3:1} (B00, B11)
        # M2: U {2:1, 3:1}, V {0:1} (B00)
        # M3: U {0:1}, V {2:1, 0:-1} (B10, B00) -> B10 - B00
        # M4: U {3:1}, V {1:1, 3:-1} (B01, B11) -> B01 - B11
        # M5: U {0:1, 1:1}, V {3:1} (B11)
        # M6: U {2:1, 0:-1}, V {0:1, 1:1} (B00, B01)
        # M7: U {1:1, 3:-1}, V {2:1, 3:1} (B10, B11)
    ]
    
    # Corrected List of Terms:
    terms = [
        # (U, V, W)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {1: 1}),             # M2
        ({0: 1}, {2: 1, 0: -1}, {2: 1, 3: 1}),      # M3
        ({3: 1}, {1: 1, 3: -1}, {0: 1, 1: 1}),      # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 2: 1}),      # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 3: 1}) # M7
    ]
    
    r = 0
    for b in range(3):
        for u_coeffs, v_coeffs, w_coeffs in terms:
            # Fill U
            for idx, val in u_coeffs.items():
                U[idx, r] = val
            
            # Fill V
            # Local B indices:
            # 0: B00 (row 0, col 0 of block) -> Global: 0*p + (2*b + 0) = 2*b
            # 1: B01 (row 0, col 1 of block) -> Global: 0*p + (2*b + 1) = 2*b + 1
            # 2: B10 (row 1, col 0 of block) -> Global: 1*p + (2*b + 0) = 6 + 2*b
            # 3: B11 (row 1, col 1 of block) -> Global: 1*p + (2*b + 1) = 6 + 2*b + 1
            for l_idx, val in v_coeffs.items():
                if l_idx == 0:
                    v_idx = 2 * b
                elif l_idx == 1:
                    v_idx = 2 * b + 1
                elif l_idx == 2:
                    v_idx = 6 + 2 * b
                elif l_idx == 3:
                    v_idx = 6 + 2 * b + 1
                V[v_idx, r] = val
            
            # Fill W
            # Local C indices:
            # 0: C00 (col 0 of block, row 0) -> Global: (2*b + 0)*n + 0 = 4*b
            # 1: C10 (col 0 of block, row 1) -> Global: (2*b + 0)*n + 1 = 4*b + 1
            # 2: C01 (col 1 of block, row 0) -> Global: (2*b + 1)*n + 0 = 4*b + 2
            # 3: C11 (col 1 of block, row 1) -> Global: (2*b + 1)*n + 1 = 4*b + 3
            for c_idx, val in w_coeffs.items():
                if c_idx == 0:
                    w_idx = 4 * b
                elif c_idx == 1:
                    w_idx = 4 * b + 1
                elif c_idx == 2:
                    w_idx = 4 * b + 2
                elif c_idx == 3:
                    w_idx = 4 * b + 3
                W[w_idx, r] = val
            
            r += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
