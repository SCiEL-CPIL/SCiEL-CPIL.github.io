"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen for 7 blocks of 2x2
    # Total rank = 7 * 7 = 49
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U row index: i * m + j
    # 0: 0*2+0=0 (a)
    # 1: 0*2+1=1 (b)
    # 2: 1*2+0=2 (c)
    # 3: 1*2+1=3 (d)
    
    # Strassen linear forms for B (2x2)
    # B indices in a block: e:(0,0), f:(0,1), g:(1,0), h:(1,1)
    # V row index: j * p + k
    
    # Strassen linear forms for C (2x2)
    # C indices in a block: c00:(0,0), c01:(0,1), c10:(1,0), c11:(1,1)
    # W row index: k * n + i
    
    # Define the 7 Strassen steps
    # Each step is a tuple of (coeffs_A, coeffs_B, coeffs_C)
    # coeffs_A is dict {index: coeff}
    # coeffs_B is dict {index: coeff}
    # coeffs_C is dict {index: coeff}
    
    steps = [
        # M1 = (a+d)(e+h)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (c+d)e
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = a(f-h)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = d(g-f)
        ({3: 1}, {2: 1, 1: -1}, {0: 1, 2: 1}),
        # M5 = (a+b)h
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (c-a)(e+f)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (b-d)(g+h)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Local B indices: 0:e, 1:f, 2:g, 3:h
    # Local C indices: 0:c00, 1:c01, 2:c10, 3:c11
    
    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            cA, cB, cC = steps[t]
            
            # Fill U
            for idx, coeff in cA.items():
                U[idx][r] = coeff
                
            # Fill V
            # Map local B index to global V index
            # local 0 (e) -> j=0, k=2s -> 0*p + 2s
            # local 1 (f) -> j=0, k=2s+1 -> 0*p + 2s+1
            # local 2 (g) -> j=1, k=2s -> 1*p + 2s
            # local 3 (h) -> j=1, k=2s+1 -> 1*p + 2s+1
            for idx, coeff in cB.items():
                if idx == 0:
                    v_idx = 0 * p + 2 * s
                elif idx == 1:
                    v_idx = 0 * p + 2 * s + 1
                elif idx == 2:
                    v_idx = 1 * p + 2 * s
                elif idx == 3:
                    v_idx = 1 * p + 2 * s + 1
                else:
                    continue
                V[v_idx][r] = coeff
                
            # Fill W
            # Map local C index to global W index
            # local 0 (c00) -> i=0, k=2s -> 2s*n + 0
            # local 1 (c01) -> i=0, k=2s+1 -> (2s+1)*n + 0
            # local 2 (c10) -> i=1, k=2s -> 2s*n + 1
            # local 3 (c11) -> i=1, k=2s+1 -> (2s+1)*n + 1
            for idx, coeff in cC.items():
                if idx == 0:
                    w_idx = 2 * s * n + 0
                elif idx == 1:
                    w_idx = (2 * s + 1) * n + 0
                elif idx == 2:
                    w_idx = 2 * s * n + 1
                elif idx == 3:
                    w_idx = (2 * s + 1) * n + 1
                else:
                    continue
                W[w_idx][r] = coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
