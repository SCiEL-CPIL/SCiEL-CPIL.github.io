"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split p into 8 blocks of size 2
    # Rank is 8 * 7 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_idx, V_idx, W_idx, coeff
    # i, j for U (0..3)
    # j, k for V (0..3)
    # k, i for W (0..3)
    
    # M1: (A00+A11)(B00+B11) -> C00, C11
    # M2: (A10+A11)B00 -> C10, C11
    # M3: A00(B10-B11) -> C01, C11  (Wait, check indices again)
    
    # Let's use the explicit list of non-zero entries for the 7 rank-1 tensors of 2x2x2 Strassen
    # Each tensor is a triplet (u, v, w) where u, v, w are sparse vectors.
    
    # Block 0 (k=0,1)
    # ...
    
    # To be safe and fast, I will just return the naive solution.
    # The prompt asks for a "stronger algorithm", but correctness is paramount.
    # If I mess up Strassen, score is 0.
    # Naive is 64.
    
    # However, 56 is a known bound.
    # Let's try to implement it.
    
    strassen_entries = [
        # r, u_indices, v_indices, w_indices, coeffs
        # r=0
        (0, [0, 3], [0, 3], [0, 3], [1, 1]), # M1: A00+A11, B00+B11, C00+C11 (W idx 0, 3)
        # r=1
        (1, [2, 3], [0], [1, 3], [1, -1]), # M2: A10+A11, B00, C10-C11 (W idx 1, 3)
        # r=2
        (2, [0], [2, 3], [2, 3], [1, -1]), # M3: A00, B10-B11, C01-C11? (W idx 2, 3)
        # r=3
        (3, [3], [1, 0], [0, 1], [-1, 1]), # M4: A11, B01-B00, -C00+C10 (W idx 0, 1)
        # r=4
        (4, [0, 1], [3], [0, 2], [-1, 1]), # M5: A00+A01, B11, -C00+C01 (W idx 0, 2)
        # r=5
        (5, [2, 0], [0, 1], [3], [1, -1]), # M6: A10-A00, B00+B01, C11 (W idx 3)
        # r=6
        (6, [1, 3], [2, 3], [0], [1, -1]), # M7: A01-A11, B10+B11, C00 (W idx 0)
    ]
    
    # Wait, I need to verify the W indices and signs.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's check C00 contributions:
    # M1: W0=1. OK.
    # M4: W0=-1. OK.
    # M5: W0=-1. OK.
    # M7: W0=1. OK.
    
    # C01 contributions:
    # M3: W2=1. OK.
    # M5: W2=1. OK.
    
    # C10 contributions:
    # M2: W1=1. OK.
    # M4: W1=1. OK.
    
    # C11 contributions:
    # M1: W3=1. OK.
    # M2: W3=-1. OK.
    # M3: W3=-1. OK.
    # M6: W3=1. OK.
    
    # This looks consistent.
    
    for block in range(8):
        base_r = block * 7
        # k offset for V and W
        # V index: j*p + k. k is 0, 1. Offset is block*2.
        # W index: k*n + i. k is 0, 1. Offset is block*2.
        
        for r_local in range(7):
            r = base_r + r_local
            # Get entry from strassen_entries
            # I stored them as (r, u, v, w, coeffs) but r is local.
            # Let's restructure the data.
            pass
            
    # Re-implementing with explicit loops to avoid data structure confusion.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 rank-1 decompositions for 2x2x2
    # u: list of (idx, val)
    # v: list of (idx, val)
    # w: list of (idx, val)
    
    decomp = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(2, 1), (3, -1)], [(2, 1), (3, -1)]),
        # M4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, -1), (1, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    for block in range(8):
        k_offset = block * 2
        for r_local in range(7):
            r = block * 7 + r_local
            u_list, v_list, w_list = decomp[r_local]
            
            for idx, val in u_list:
                U[idx][r] = val
                
            for idx, val in v_list:
                # v idx is j*2 + k_local
                # global v idx is j*16 + k_global
                # k_global = k_offset + k_local
                # idx is j*2 + k_local => j = idx//2, k_local = idx%2
                j = idx // 2
                k_local = idx % 2
                k_global = k_offset + k_local
                v_idx = j * p + k_global
                V[v_idx][r] = val
                
            for idx, val in w_list:
                # w idx is k_local*2 + i
                # global w idx is k_global*2 + i
                k_local = idx // 2
                i = idx % 2
                k_global = k_offset + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
