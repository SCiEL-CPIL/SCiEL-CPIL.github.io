"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for multiplying a 2x3 matrix by a 3x8 matrix.
    Uses a rank-32 decomposition based on the rank-8 decomposition for 2x3x2,
    replicated 4 times to cover the 8 columns (p=8).
    
    The rank-8 for 2x3x2 comes from:
    - Strassen (rank 7) for the 2x2 x 2x2 sub-block
    - 1 term for the remaining 2x1 x 1x2 outer product
    
    Actually, let me use a cleaner approach:
    For 2x3 x 3x2:
    We can view this as computing 2x2 output.
    A (2x3) * B (3x2) = C (2x2)
    
    Let A = [[a1, a2, a3], [b1, b2, b3]]
    Let B = [[x1, x2], [y1, y2], [z1, z2]]
    
    C[0,0] = a1*x1 + a2*y1 + a3*z1
    C[0,1] = a1*x2 + a2*y2 + a3*z2
    C[1,0] = b1*x1 + b2*y1 + b3*z1
    C[1,1] = b1*x2 + b2*y2 + b3*z2
    
    Schoolbook rank for 2x3x2 is 12.
    
    Let me try a different approach. I'll use the known fact that
    R(2,3,8) can be at most 32 by using R(2,3,2)=8.
    
    Actually, let me just use a simple construction that achieves rank 32.
    
    For p=8, we can split B into 4 blocks of 3x2.
    For each 3x2 block, we compute 2x3 * 3x2 using rank 8.
    Total rank = 4 * 8 = 32.
    
    Rank-8 for 2x3x2:
    A (2x3) * B (3x2)
    A = [[a, b, c], [d, e, f]]
    B = [[x, y], [u, v], [w, z]]
    
    C[0,0] = ax + bu + cw
    C[0,1] = ay + bv + cw... wait, C[0,1] = ay + bv + cz
    C[1,0] = dx + eu + fw
    C[1,1] = dy + ev + fz
    
    8 multiplications:
    m1 = a*x
    m2 = b*u
    m3 = c*w
    m4 = d*x
    m5 = e*u
    m6 = f*w
    m7 = a*y
    m8 = b*v  ... but then C[0,1] = m7 + m8 + c*z, need c*z too
    
    Let me just use schoolbook for 2x3x2 which is rank 12, giving total 4*12=48.
    That's the same as before.
    
    Let me try to find a better decomposition for 2x3x2.
    
    Actually, I'll use a known rank-8 for 2x3x2:
    m1 = (a+b)*x
    m2 = (d+e)*x
    m3 = a*u
    m4 = d*u
    m5 = (a+b)*y
    m6 = (d+e)*y
    m7 = c*(w+z)
    m8 = f*(w+z)
    
    Hmm, this doesn't quite work cleanly.
    
    Let me just use the straightforward schoolbook for now and ensure correctness.
    The current program already does schoolbook with rank 48.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
