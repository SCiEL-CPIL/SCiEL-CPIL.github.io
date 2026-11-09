"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import itertools
from fractions import Fraction

def _build_rank28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7 terms for 2x2 matrix multiplication
    # Term structure: (U_terms, V_terms, W_terms)
    # Each term: list of (local_index, coeff)
    # Local indices for 2x2: a=0, b=1, c=2, d=3
    # For V: e=0, f=1, g=2, h=3
    # For W: k0i0=0, k0i1=1, k1i0=2, k1i1=3
    
    S = [
        # 1: (a+d)(e+h) -> C00 +=, C11 +=
        ({(0,1),(3,1)}, {(0,1),(3,1)}, {(0,1),(3,1)}),
        # 2: (c+d)e -> C10 +=, C11 -=
        ({(2,1),(3,1)}, {(0,1)}, {(1,1),(3,-1)}),
        # 3: a(f-h) -> C01 +=, C11 +=
        ({(0,1)}, {(1,1),(3,-1)}, {(2,1),(3,1)}),
        # 4: d(g-e) -> C00 +=, C10 +=
        ({(3,1)}, {(2,1),(0,-1)}, {(0,1),(1,1)}),
        # 5: (a+b)h -> C00 -=, C01 +=
        ({(0,1),(1,1)}, {(3,1)}, {(0,-1),(2,1)}),
        # 6: (c-a)(e+f) -> C11 +=
        ({(2,1),(0,-1)}, {(0,1),(1,1)}, {(3,1)}),
        # 7: (b-d)(g+h) -> C00 +=
        ({(1,1),(3,-1)}, {(2,1),(3,1)}, {(0,1)}),
    ]
    
    # 4 block combinations of (j_block, k_block) each of size 2
    j_offsets = [0, 2]  # j in {0,1} or {2,3}
    k_offsets = [0, 2]  # k in {0,1} or {2,3}
    
    col = 0
    for j_off in j_offsets:
        for k_off in k_offsets:
            for u_terms, v_terms, w_terms in S:
                for (li, coeff) in u_terms:
                    i, j = li // 2, li % 2
                    gi, gj = i, j + j_off
                    U[gi * m + gj][col] += coeff
                
                for (li, coeff) in v_terms:
                    j, k = li // 2, li % 2
                    gj, gk = j + j_off, k + k_off
                    V[gj * p + gk][col] += coeff
                
                for (li, coeff) in w_terms:
                    k, i = li // 2, li % 2
                    gk, gi = k + k_off, i
                    W[gk * n + gi][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    
    # Try to find a better decomposition using randomized LP relaxation
    # But for safety, start with the known rank-31 solution
    
    result = _build_rank28()
    
    # Try to improve using simple random search with time limit
    # This is a heuristic approach - we'll try to find rank 29 or 30
    # by looking for structure that can share multiplications
    
    elapsed = time.time() - start
    if elapsed < 10:  # If we have time, try some optimization
        # For now, just return the safe rank-31 solution
        # A proper optimization would involve solving a larger LP
        pass
    
    return result
# EVOLVE-BLOCK-END
