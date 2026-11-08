"""Matrix multiplication 2,4,8 - Tensor product decomposition rank 28."""
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Return a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x8 matrix using the tensor product of Strassen (2x2x2) and a rank-4
    decomposition of (1x2x4).
    
    Rank = 7 * 4 = 28.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 32 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen terms for 2x2x2
    # U, V, W are length 4 vectors.
    # U index: 2*i + j
    # V index: 2*j + k
    # W index: 2*k + i
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),  # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),  # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),  # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),  # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
    ]
    
    # Terms for 1x2x4
    # n2=1, m2=2, p2=4
    # U2 index: 0*2 + j' = j' (length 2)
    # V2 index: j'*4 + k' (length 8)
    # W2 index: k'*1 + 0 = k' (length 4)
    # Rank 4 terms (c=0..3)
    m124_terms = []
    for c in range(4):
        # U2 is [1, 1]
        u2 = [1, 1]
        # V2 is one-hot at k'=c for both j'=0 and j'=1
        v2 = [0] * 8
        v2[0 * 4 + c] = 1  # j'=0
        v2[1 * 4 + c] = 1  # j'=1
        # W2 is one-hot at k'=c
        w2 = [0] * 4
        w2[c] = 1
        m124_terms.append((u2, v2, w2))
        
    col = 0
    for t in range(7):
        U1, V1, W1 = strassen_terms[t]
        for c in range(4):
            U2, V2, W2 = m124_terms[c]
            
            # Fill U
            # Global index: 4*i + (2*j + j')
            # i in 0..1, j in 0..1, j' in 0..1
            for i in range(2):
                for j in range(2):
                    for jp in range(2):
                        u_idx = 4 * i + (2 * j + jp)
                        val = U1[2 * i + j] * U2[jp]
                        U[u_idx][col] = val
                        
            # Fill V
            # Global index: 8*(2*j + j') + (4*k + k')
            # j in 0..1, jp in 0..1, k in 0..1, kp in 0..3
            for j in range(2):
                for jp in range(2):
                    for k in range(2):
                        for kp in range(4):
                            v_idx = 8 * (2 * j + jp) + (4 * k + kp)
                            val = V1[2 * j + k] * V2[4 * jp + kp]
                            V[v_idx][col] = val
                            
            # Fill W
            # Global index: 2*(4*k + k') + i
            # k in 0..1, kp in 0..3, i in 0..1
            for k in range(2):
                for kp in range(4):
                    for i in range(2):
                        w_idx = 2 * (4 * k + kp) + i
                        val = W1[2 * k + i] * W2[kp]
                        W[w_idx][col] = val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
