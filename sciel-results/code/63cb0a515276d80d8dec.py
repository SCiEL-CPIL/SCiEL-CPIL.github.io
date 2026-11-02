"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 145  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-35 exact tensor decomposition for 2x2 * 2x10 matrix multiplication.
    
    Uses Strassen's rank-7 algorithm on each of 5 independent 2x2 column blocks.
    Total rank = 5 * 7 = 35.
    
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer entries.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    num_blocks = p // 2   # 5 blocks
    rank = 7 * num_blocks  # 35

    U = [[0] * rank for _ in range(n * m)]   # (4, 35)
    V = [[0] * rank for _ in range(m * p)]   # (20, 35)
    W = [[0] * rank for _ in range(p * n)]   # (20, 35)

    # Strassen's 7 products for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    #
    # C[0,0] = P1 + P4 - P5 + P7
    # C[0,1] = P3 + P5
    # C[1,0] = P2 + P4
    # C[1,1] = P1 - P2 + P3 + P6
    #
    # U indices: i*m+j -> a=0, b=1, c=2, d=3
    # V indices: j*p+k -> e=V[0*p+2b], f=V[0*p+2b+1], g=V[1*p+2b], h=V[1*p+2b+1]
    # W indices: k*n+i -> C[i,k]=W[k*n+i]

    # Strassen coefficient table: (product_idx, U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (u_idx, coeff)
    # V_coeffs: list of (v_offset, coeff) where v_idx = v_offset + 2*b
    # W_coeffs: list of (w_offset, coeff) where w_idx = w_offset + 4*b
    
    strassen_products = [
        # P1 = (a+d)(e+h)
        ([(0,1),(3,1)], [(0,1),(p+1,1)], [(0,1),(3,1)]),
        # P2 = (c+d)e
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # P3 = a(f-h)
        ([(0,1)], [(1,1),(p+1,-1)], [(2,1),(3,1)]),
        # P4 = d(g-e)
        ([(3,1)], [(p,1),(0,-1)], [(0,1),(1,1)]),
        # P5 = (a+b)h
        ([(0,1),(1,1)], [(p+1,1)], [(0,-1),(2,1)]),
        # P6 = (c-a)(e+f)
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # P7 = (b-d)(g+h)
        ([(1,1),(3,-1)], [(p,1),(p+1,1)], [(0,1)]),
    ]

    for b in range(num_blocks):
        base = 7 * b
        for pi, (u_coeffs, v_coeffs, w_coeffs) in enumerate(strassen_products):
            col = base + pi
            for u_idx, coeff in u_coeffs:
                U[u_idx][col] += coeff
            for v_off, coeff in v_coeffs:
                V[v_off + 2*b][col] += coeff
            for w_off, coeff in w_coeffs:
                W[w_off + 4*b][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
