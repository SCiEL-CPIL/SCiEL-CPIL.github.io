# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 145

def construct():
    """
    Decompose 4x4 * 4x8 using recursive Strassen.
    Rank 49 per 4x4x4 block, total rank 98.
    """
    n, m, p = 4, 4, 8
    
    # Inner Strassen 2x2x2 decomposition
    # 7 terms, each (u, v, w) vectors of length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    inner_strassen = [
        # M1: (A00+A11)(B00+B11) -> C00+C11
        (([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])),
        # M2: (A10+A11)B00 -> C10
        (([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])),
        # M3: A00(B01-B11) -> C01
        (([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0])),
        # M4: A11(B10-B00) -> C00+C10? No, C00 and C10.
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # So M4 contributes to C00 (coeff 1) and C10 (coeff 1).
        # Wait, standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Let's check M4: A11(B10-B00).
        # A11 is (1,1) block. B10 is (1,0), B00 is (0,0).
        # Product is (1,1)x(1,0) - (1,1)x(0,0).
        # This contributes to C10 and C11?
        # C10 = sum A1j Bjk. j=0,1. A10 B00 + A11 B10.
        # C11 = A10 B01 + A11 B11.
        # M4 = A11(B10 - B00) = A11 B10 - A11 B00.
        # This is not directly a term in C10 or C11.
        # The reconstruction formula handles the combination.
        # So w vector for M4 should indicate where it goes in the reconstruction.
        # But the rank-1 tensor is defined by the multiplication.
        # The w vector in the decomposition $C = \sum u v w^T$ (or similar) 
        # must be such that $\sum w_{t,k} (u_t \cdot A) (v_t \cdot B) = C_k$.
        
        # Let's use the standard rank-1 decomposition vectors for 2x2x2.
        # T = sum u_i v_i w_i
        # u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [1,0,0,1]
        # u2 = [0,0,1,1], v2 = [1,0,0,0], w2 = [0,0,1,0]
        # u3 = [1,0,0,0], v3 = [0,1,0,-1], w3 = [0,1,0,0]
        # u4 = [0,0,0,1], v4 = [0,0,1,-1], w4 = [1,0,1,0]  # Check: A11(B10-B00)
        # u5 = [1,1,0,0], v5 = [0,0,0,1], w5 = [0,1,0,0]  # Check: (A00+A01)B11
        # u6 = [0,0,1,-1], v6 = [1,1,0,0], w6 = [0,0,0,1]  # Check: (A10-A00)(B00+B01)
        # u7 = [0,1,-1,0], v7 = [0,0,1,1], w7 = [1,0,0,1]  # Check: (A01-A11)(B10+B11)
        
        # Let's verify w4.
        # M4 = A11(B10-B00).
        # Contribution to C:
        # C00 gets M4? C00 = M1+M4-M5+M7. Yes.
        # C10 gets M4? C10 = M2+M4. Yes.
        # So w4 should have 1 at (0,0) and 1 at (1,0).
        # My w4 above: [1,0,1,0] -> (0,0) and (1,0). Correct.
        
        # Let's verify w7.
        # M7 = (A01-A11)(B10+B11).
        # C00 gets M7? Yes.
        # C11 gets M7? No, C11 = M1-M2+M3+M6.
        # Wait, standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M7 contributes to C00 only?
        # Let's check M7 expansion: (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
        # C00 = A00B00 + A01B10.
        # C01 = A00B01 + A01B11.
        # C10 = A10B00 + A11B10.
        # C11 = A10B01 + A11B11.
        
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # M4 = A11(B10-B00) = A11B10 - A11B00.
        # M5 = (A00+A01)B11 = A00B11 + A01B11.
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
        
        # C00 = M1 + M4 - M5 + M7
        # = (A00B00 + A00B11 + A11B00 + A11B11) + (A11B10 - A11B00) - (A00B11 + A01B11) + (A01B10 + A01B11 - A11B10 - A11B11)
        # = A00B00 + A00B11 - A00B11 + A11B00 - A11B00 + A11B11 - A11B11 + A11B10 - A11B10 - A01B11 + A01B11 + A01B10
        # = A00B00 + A01B10. Correct.
        
        # So w7 should be [1,0,0,0] (only C00).
        # My w7 above: [1,0,0,1]. Incorrect.
        # Let's re-derive w vectors from reconstruction.
        
        # w1: C00(1), C11(1) -> [1,0,0,1]
        # w2: C10(1) -> [0,0,1,0]
        # w3: C01(1) -> [0,1,0,0]
        # w4: C00(1), C10(1) -> [1,0,1,0]
        # w5: C00(-1), C01(1) -> [-1,1,0,0]
        # w6: C11(1) -> [0,0,0,1]
        # w7: C00(1) -> [1,0,0,0]
        
        # Let's check w5.
        # M5 = (A00+A01)B11.
        # C00 = ... - M5.
        # C01 = M3 + M5.
        # So w5 = [-1, 1, 0, 0].
        
        # Let's check w6.
        # M6 = (A10-A00)(B00+B01).
        # C11 = ... + M6.
        # So w6 = [0, 0, 0, 1].
        
        # Let's check w7.
        # M7 = (A01-A11)(B10+B11).
        # C00 = ... + M7.
        # So w7 = [1, 0, 0, 0].
        
        # Corrected Inner Strassen:
        (([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1])),
        (([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0])),
        (([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0])),
        (([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0])),
        (([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0])),
        (([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1])),
        (([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0])),
    ]
    
    # Outer Strassen 4x4x4 (2x2 blocks of 2x2)
    # 7 multipliers
    # Each: (a_blocks, b_blocks)
    # a_blocks: list of (block_idx, coeff)
    # block_idx: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    outer_strassen = [
        # M1: (A00+A11)(B00+B11)
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        # M2: (A10+A11)B00
        (([(2, 1), (3, 1)], [(0, 1)])),
        # M3: A00(B01-B11)
        (([(0, 1)], [(1, 1), (3, -1)])),
        # M4: A11(B10-B00)
        (([(3, 1)], [(2, 1), (0, -1)])),
        # M5: (A00+A01)B11
        (([(0, 1), (1, 1)], [(3, 1)])),
        # M6: (A10-A00)(B00+B01)
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)])),
        # M7: (A01-A11)(B10+B11)
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)])),
    ]
    
    # Output reconstruction for 4x4x4
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    output_recon = {
        0: {0: 1, 3: 1, 4: -1, 6: 1},   # C00
        1: {2: 1, 4: 1},                # C01
        2: {1: 1, 3: 1},                # C10
        3: {0: 1, 1: -1, 2: 1, 5: 1},   # C11
    }
    
    U = []
    V = []
    W = []
    
    # 2 blocks for the 8 columns
    for block in range(2):
        k_off = block * 4
        
        for mi in range(7):
            a_blocks, b_blocks = outer_strassen[mi]
            
            # For each inner Strassen term
            for ti in range(7):
                u_in, v_in, w_in = inner_strassen[ti]
                
                # Construct global U (size 16)
                u_global = [0] * 16
                for bi, coeff in a_blocks:
                    # bi is block index (0-3)
                    # Map to global indices
                    br = bi // 2
                    bc = bi % 2
                    for k in range(4):
                        idx_in = k
                        # u_in[k] is coefficient for A_block[br, bc][k]
                        # Global index: (2*br + r, 2*bc + c)
                        # k corresponds to (r, c) in 2x2
                        r = k // 2
                        c = k % 2
                        g_idx = (2*br + r) * 4 + (2*bc + c)
                        u_global[g_idx] += coeff * u_in[k]
                
                # Construct global V (size 32)
                v_global = [0] * 32
                for bi, coeff in b_blocks:
                    br = bi // 2
                    bc = bi % 2
                    for k in range(4):
                        r = k // 2
                        c = k % 2
                        # V index: (2*br + r) * 8 + (2*bc + c + k_off)
                        # Note: k_off is 0 or 4
                        g_idx = (2*br + r) * 8 + (2*bc + c + k_off)
                        v_global[g_idx] += coeff * v_in[k]
                
                # Construct global W (size 32)
                w_global = [0] * 32
                # w_in is for the 2x2 block output
                # We need to scale by output_recon
                # The output block is determined by the outer Strassen multiplier mi
                # Wait, the outer Strassen multiplier mi produces a 2x2 block of C?
                # No, the outer Strassen multipliers produce the 4 2x2 blocks of C.
                # The reconstruction `output_recon` maps mi to C blocks.
                # For a given mi, it contributes to specific C blocks with specific coeffs.
                # The inner Strassen term ti produces the 4 entries of that 2x2 block.
                # So for each C block cj that mi contributes to:
                #   We add coeff * w_in to the corresponding part of W.
                
                for cj, coeff in output_recon[cj].items() if mi in output_recon[cj] else []:
                    # cj is 0,1,2,3 (C00, C01, C10, C11)
                    cr = cj // 2
                    cc = cj % 2
                    for k in range(4):
                        r = k // 2
                        c = k % 2
                        # W index: (2*cc + c + k_off) * 4 + (2*cr + r)
                        # Wait, W is (p*n, rank). p=8, n=4.
                        # Index: (col, row)
                        # col = 2*cc + c + k_off
                        # row = 2*cr + r
                        g_idx = (2*cc + c + k_off) * 4 + (2*cr + r)
                        w_global[g_idx] += coeff * w_in[k]
                
                U.append(u_global)
                V.append(v_global)
                W.append(w_global)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
