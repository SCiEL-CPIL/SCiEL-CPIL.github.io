    def construct():
        n, m, p = 2, 4, 6
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen steps definition
        # A coeffs: list of (row, col, val)
        # B coeffs: list of (row, col, val)
        steps = [
            {
                'A': [(0,0,1), (1,1,1)],
                'B': [(0,0,1), (1,1,1)],
                'C': { (0,0): 1, (1,1): 1, (0,1): 0, (1,0): 0 } # This is M1 contribution to C
            },
            # ... this is getting complicated to map C coeffs per step.
        ]
        
        # Better: Define A and B coeffs for each step.
        # Then define how each step contributes to C00, C01, C10, C11.
        
        A_coeffs = [
            [(0,0,1), (1,1,1)], # M1
            [(1,0,1), (1,1,1)], # M2
            [(0,0,1)],          # M3
            [(1,1,1)],          # M4
            [(0,0,1), (0,1,1)], # M5
            [(1,0,1), (0,0,-1)],# M6
            [(0,1,1), (1,1,-1)] # M7
        ]
        
        B_coeffs = [
            [(0,0,1), (1,1,1)], # M1
            [(0,0,1)],          # M2
            [(1,0,1), (1,1,-1)],# M3
            [(0,1,1), (0,0,-1)],# M4
            [(1,1,1)],          # M5
            [(0,0,1), (0,1,1)], # M6
            [(1,0,1), (1,1,1)]  # M7
        ]
        
        # C reconstruction: C_local[i,k] = sum_step coeff[step] * M_step
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        C_recon = {
            (0,0): {0: 1, 3: 1, 4: -1, 6: 1},
            (0,1): {2: 1, 4: 1},
            (1,0): {1: 1, 3: 1},
            (1,1): {0: 1, 1: -1, 2: 1, 5: 1}
        }
        
        t = 0
        for b in range(2): # Block index for A and B middle dimension
            for s in range(3): # Column block index for B (and C)
                for k in range(7): # Strassen step
                    # Fill U
                    for (r, c, val) in A_coeffs[k]:
                        i_global = r
                        j_global = 2*b + c
                        U[i_global * m + j_global][t] += val
                    
                    # Fill V
                    for (r, c, val) in B_coeffs[k]:
                        j_global = 2*b + r
                        k_global = 2*s + c
                        V[j_global * p + k_global][t] += val
                    
                    # Fill W
                    # We need to add the contribution of M_k to the C cells.
                    # Iterate over all C cells that M_k affects.
                    for (ci, ck), coeffs in C_recon.items():
                        if k in coeffs:
                            val = coeffs[k]
                            i_global = ci
                            k_global = 2*s + ck
                            # W index is k*n + i
                            W[k_global * n + i_global][t] += val
                    
                    t += 1
        
        return {"U": U, "V": V, "W": W}
    