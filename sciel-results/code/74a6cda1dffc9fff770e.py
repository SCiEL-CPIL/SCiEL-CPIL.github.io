"""Matrix multiplication 2x4x4 via 2x2x2 Strassen blocks (rank-28)."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Tensor T[i, j, k] = 1 for i in 0..1, j in 0..3, k in 0..3
    # We split j into two halves of size 2: j1 in {0,1}, j2 in {2,3}
    # We split k into two halves of size 2: k1 in {0,1}, k2 in {2,3}
    # This gives 4 blocks, each a 2x2x2 multiplication.
    # Each 2x2x2 block uses 7 Strassen factors.
    # Total rank = 4 * 7 = 28.
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # (8, 28)
    
    # Strassen 2x2x2 decomposition:
    # For matrices A (2x2), B (2x2), C (2x2):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Strassen factors as (U_coeff, V_coeff, W_coeff) for each output position
    # U index: local i (0,1) * 2 + local j (0,1)
    # V index: local j (0,1) * 2 + local k (0,1)
    # W index: local k (0,1) * 2 + local i (0,1)
    
    # Each factor r has:
    #   U contributions: list of (u_idx, coeff)
    #   V contributions: list of (v_idx, coeff)
    #   W contributions: list of (w_idx, coeff)
    
    strassen_factors = [
        # M1: (A00+A11)(B00+B11)
        {
            'U': [(0, 1), (3, 1)],  # A00, A11
            'V': [(0, 1), (3, 1)],  # B00, B11
            'W': [(0, 1), (2, -1), (4, 1), (6, 1)],  # C00, C10, C01, C11
        },
        # M2: (A10+A11)B00
        {
            'U': [(2, 1), (3, 1)],  # A10, A11
            'V': [(0, 1)],  # B00
            'W': [(2, 1), (6, -1)],  # C10, C11
        },
        # M3: A00(B10-B11)
        {
            'U': [(0, 1)],  # A00
            'V': [(2, 1), (3, -1)],  # B10, B11
            'W': [(4, 1), (6, 1)],  # C01, C11
        },
        # M4: A11(B01-B10)
        {
            'U': [(3, 1)],  # A11
            'V': [(1, 1), (2, -1)],  # B01, B10
            'W': [(0, -1), (2, 1)],  # C00, C10
        },
        # M5: (A00+A01)B11
        {
            'U': [(0, 1), (1, 1)],  # A00, A01
            'V': [(3, 1)],  # B11
            'W': [(0, 1), (4, 1)],  # C00, C01
        },
        # M6: (A10-A00)(B00+B10)
        {
            'U': [(2, 1), (0, -1)],  # A10, A00
            'V': [(0, 1), (2, 1)],  # B00, B10
            'W': [(6, 1)],  # C11
        },
        # M7: (A01-A11)(B01+B11)
        {
            'U': [(1, 1), (3, -1)],  # A01, A11
            'V': [(1, 1), (3, 1)],  # B01, B11
            'W': [(0, 1)],  # C00
        },
    ]
    
    # 4 blocks: (j_half, k_half) where j_half in {0,1}, k_half in {0,1}
    # j_half=0: j in {0,1}, j_half=1: j in {2,3}
    # k_half=0: k in {0,1}, k_half=1: k in {2,3}
    
    for jh in range(2):
        for kh in range(2):
            base_col = (jh * 2 + kh) * 7  # 0, 7, 14, 21
            
            for r in range(7):
                col = base_col + r
                factor = strassen_factors[r]
                
                # U contributions
                for u_local, coeff in factor['U']:
                    # u_local is index in 2x2: i_local * 2 + j_local
                    i_local = u_local // 2
                    j_local = u_local % 2
                    # Global U index: i * m + j = i * 4 + (jh*2 + j_local)
                    u_global = i_local * m + (jh * 2 + j_local)
                    U[u_global][col] += coeff
                
                # V contributions
                for v_local, coeff in factor['V']:
                    # v_local is index in 2x2: j_local * 2 + k_local
                    j_local = v_local // 2
                    k_local = v_local % 2
                    # Global V index: j * p + k = (jh*2 + j_local) * 4 + (kh*2 + k_local)
                    v_global = (jh * 2 + j_local) * p + (kh * 2 + k_local)
                    V[v_global][col] += coeff
                
                # W contributions
                for w_local, coeff in factor['W']:
                    # w_local is index in 2x2 output: k_local * 2 + i_local
                    k_local = w_local // 2
                    i_local = w_local % 2
                    # Global W index: k * n + i = (kh*2 + k_local) * 2 + i_local
                    w_global = (kh * 2 + k_local) * n + i_local
                    W[w_global][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
