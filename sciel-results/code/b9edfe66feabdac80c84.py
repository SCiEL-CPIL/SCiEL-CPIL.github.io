import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    # Helper to get global indices
    def u_idx(i, k): return i * m + k
    def v_idx(k, j): return k * p + j
    def w_idx(j, i): return j * n + i # Matches original convention
    
    # Strassen 2x2 definitions
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h) -> C11 += M1, C22 += M1
    # M2 = (c+d)e    -> C21 += M2, C22 -= M2
    # M3 = a(f-h)    -> C12 += M3, C22 += M3
    # M4 = d(g-e)    -> C11 += M4, C21 += M4
    # M5 = (a+b)h    -> C11 -= M5, C12 += M5
    # M6 = (c-a)(e+f) -> C22 += M6
    # M7 = (b-d)(g+h) -> C11 += M7
    
    # Outer Strassen for 4x4 (blocks of 2x2)
    # A11, A12, A21, A22 are 2x2 blocks.
    # We define the 7 outer terms.
    # Each term involves linear combinations of A blocks and B blocks.
    # Then we apply the 7 inner Strassen terms to the resulting 2x2 "matrices" (which are sums of blocks).
    
    # To simplify, we iterate over the 7 outer terms.
    # For each outer term, we define the coefficients for A blocks and B blocks.
    # Then we iterate over the 7 inner terms.
    # For each inner term, we update U, V, W.
    
    # Inner Strassen terms (for 2x2 matrices X, Y -> Z)
    # x11, x12, x21, x22 are elements of X.
    # y11, y12, y21, y22 are elements of Y.
    # z11, z12, z21, z22 are elements of Z.
    
    inner_terms = [
        { # M1
            'X': [(0,0,1), (1,1,1)], 'Y': [(0,0,1), (1,1,1)], 'Z': [(0,0,1), (1,1,1)]
        },
        { # M2
            'X': [(1,0,1), (1,1,1)], 'Y': [(0,0,1)], 'Z': [(1,0,1), (1,1,-1)]
        },
        { # M3
            'X': [(0,0,1)], 'Y': [(0,1,1), (1,1,-1)], 'Z': [(0,1,1), (1,1,1)]
        },
        { # M4
            'X': [(1,1,1)], 'Y': [(1,0,1), (0,0,-1)], 'Z': [(0,0,1), (1,0,1)]
        },
        { # M5
            'X': [(0,0,1), (0,1,1)], 'Y': [(1,1,1)], 'Z': [(0,0,-1), (0,1,1)]
        },
        { # M6
            'X': [(1,0,1), (0,0,-1)], 'Y': [(0,0,1), (0,1,1)], 'Z': [(1,1,1)]
        },
        { # M7
            'X': [(0,1,1), (1,1,-1)], 'Y': [(1,0,1), (1,1,1)], 'Z': [(0,0,1)]
        }
    ]
    
    outer_terms = [
        { # M1_outer
            'A_blocks': [(0,0,1), (1,1,1)], # A11 + A22
            'B_blocks': [(0,0,1), (1,1,1)], # B11 + B22
            'C_blocks': [(0,0,1), (1,1,1)]  # C11 + C22
        },
        { # M2_outer
            'A_blocks': [(1,0,1), (1,1,1)], # A21 + A22
            'B_blocks': [(0,0,1)],          # B11
            'C_blocks': [(1,0,1), (1,1,-1)] # C21 - C22
        },
        { # M3_outer
            'A_blocks': [(0,0,1)],          # A11
            'B_blocks': [(0,1,1), (1,1,-1)],# B12 - B22
            'C_blocks': [(0,1,1), (1,1,1)]  # C12 + C22
        },
        { # M4_outer
            'A_blocks': [(1,1,1)],          # A22
            'B_blocks': [(1,0,1), (0,0,-1)],# B21 - B11
            'C_blocks': [(0,0,1), (1,0,1)]  # C11 + C21
        },
        { # M5_outer
            'A_blocks': [(0,0,1), (0,1,1)], # A11 + A12
            'B_blocks': [(1,1,1)],          # B22
            'C_blocks': [(0,0,-1), (0,1,1)] # -C11 + C12
        },
        { # M6_outer
            'A_blocks': [(1,0,1), (0,0,-1)],# A21 - A11
            'B_blocks': [(0,0,1), (0,1,1)], # B11 + B12
            'C_blocks': [(1,1,1)]           # C22
        },
        { # M7_outer
            'A_blocks': [(0,1,1), (1,1,-1)],# A12 - A22
            'B_blocks': [(1,0,1), (1,1,1)], # B21 + B22
            'C_blocks': [(0,0,1)]           # C11
        }
    ]
    
    rank_idx = 0
    for ot in outer_terms:
        for it in inner_terms:
            if time.time() > deadline:
                break
                
            # Accumulate coefficients for U, V, W for this rank component
            # U corresponds to A elements.
            # V corresponds to B elements.
            # W corresponds to C elements.
            
            # For U:
            # It is a sum of A_blocks (from outer) multiplied by X_coeffs (from inner).
            # A_block (br, bc) contains elements A[2*br+r, 2*bc+c].
            # X_coeff (xr, xc) applies to element (xr, xc) of the 2x2 matrix formed by A_blocks.
            # So global A index is (2*br+xr, 2*bc+xc).
            
            for (abr, abc, ac) in ot['A_blocks']:
                for (xr, xc, xc_coeff) in it['X']:
                    gi = 2 * abr + xr
                    gk = 2 * abc + xc
                    coeff = ac * xc_coeff
                    U[u_idx(gi, gk), rank_idx] += coeff
            
            # For V:
            # Similar for B.
            for (bbr, bbc, bc) in ot['B_blocks']:
                for (yr, yc, yc_coeff) in it['Y']:
                    gk = 2 * bbr + yr
                    gj = 2 * bbc + yc
                    coeff = bc * yc_coeff
                    V[v_idx(gk, gj), rank_idx] += coeff
            
            # For W:
            # C_blocks (from outer) and Z_coeffs (from inner).
            for (cbr, cbc, cc) in ot['C_blocks']:
                for (zr, zc, zc_coeff) in it['Z']:
                    gi = 2 * cbr + zr
                    gj = 2 * cbc + zc
                    coeff = cc * zc_coeff
                    W[w_idx(gj, gi), rank_idx] += coeff
            
            rank_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
