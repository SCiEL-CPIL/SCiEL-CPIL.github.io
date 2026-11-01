"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Strassen-based decomposition for 2x2 * 2x14 multiplication.
    
    We split the 14 columns of B into 7 blocks of 2 columns each.
    For each block, we apply Strassen's algorithm (rank 7).
    Total rank = 7 blocks * 7 rank/block = 49.
    
    Dimensions:
    n = 2, m = 2, p = 14.
    U shape: (n*m, rank) = (4, 49)
    V shape: (m*p, rank) = (28, 49)
    W shape: (p*n, rank) = (28, 49)
    
    Trace convention: W[k*n + i, r]
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for s in range(num_blocks):
        # Column indices for this block: 2s and 2s+1
        k0 = 2 * s
        k1 = 2 * s + 1
        
        for t in range(rank_per_block):
            r = s * rank_per_block + t
            
            # U entries (coefficients of A)
            # A indices: 00->0, 01->1, 10->2, 11->3
            if t == 0: # P1: a00
                U[0][r] = 1.0
            elif t == 1: # P2: a10 + a11
                U[2][r] = 1.0
                U[3][r] = 1.0
            elif t == 2: # P3: a11
                U[3][r] = 1.0
            elif t == 3: # P4: a00
                U[0][r] = 1.0
            elif t == 4: # P5: a00 + a01
                U[0][r] = 1.0
                U[1][r] = 1.0
            elif t == 5: # P6: a01
                U[1][r] = 1.0
            elif t == 6: # P7: a10 + a01
                U[2][r] = 1.0
                U[1][r] = 1.0
            
            # V entries (coefficients of B)
            # B indices: j*p + k
            # j=0: 0*p+k = k
            # j=1: 1*p+k = p+k
            if t == 0: # P1: b0k0 + b1k1
                V[0 * p + k0][r] = 1.0
                V[1 * p + k1][r] = 1.0
            elif t == 1: # P2: b0k0
                V[0 * p + k0][r] = 1.0
            elif t == 2: # P3: b0k1 - b1k0
                V[0 * p + k1][r] = 1.0
                V[1 * p + k0][r] = -1.0
            elif t == 3: # P4: b0k0 - b1k0
                V[0 * p + k0][r] = 1.0
                V[1 * p + k0][r] = -1.0
            elif t == 4: # P5: b1k1
                V[1 * p + k1][r] = 1.0
            elif t == 5: # P6: b0k0 - b1k1
                V[0 * p + k0][r] = 1.0
                V[1 * p + k1][r] = -1.0
            elif t == 6: # P7: b1k0
                V[1 * p + k0][r] = 1.0
            
            # W entries (coefficients of C reconstruction)
            # C indices: k*n + i
            # k=k0, i=0 -> k0*n + 0
            # k=k0, i=1 -> k0*n + 1
            # k=k1, i=0 -> k1*n + 0
            # k=k1, i=1 -> k1*n + 1
            
            # c0k0 = P1 + P4 - P5 + P7
            # c1k0 = P2 + P4
            # c0k1 = P3 + P5
            # c1k1 = P1 - P2 + P3 + P6
            
            if t == 0: # P1
                W[k0 * n + 0][r] = 1.0
                W[k1 * n + 1][r] = 1.0
            elif t == 1: # P2
                W[k0 * n + 1][r] = 1.0
                W[k1 * n + 1][r] = -1.0
            elif t == 2: # P3
                W[k1 * n + 0][r] = 1.0
                W[k1 * n + 1][r] = 1.0
            elif t == 3: # P4
                W[k0 * n + 0][r] = 1.0
                W[k0 * n + 1][r] = 1.0
            elif t == 4: # P5
                W[k0 * n + 0][r] = -1.0
                W[k1 * n + 0][r] = 1.0
            elif t == 5: # P6
                W[k1 * n + 1][r] = 1.0
            elif t == 6: # P7
                W[k0 * n + 0][r] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
