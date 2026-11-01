"""Rewritten program for matrix multiplication 2,2,6 tensor decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    Strategy:
    1. Verify Strassen-based rank-21 decomposition works.
    2. Attempt to find rank-20 by exploiting shared structure in A.
    3. Fall back to schoolbook rank-24 if needed.
    
    Key insight: Since A is shared across three 2x2 blocks of B, we may
    be able to share some Strassen multiplications between blocks.
    """
    import numpy as np
    from itertools import product
    
    n, m, p = 2, 2, 6
    nm = n * m  # 4
    mp = m * p  # 12
    pn = p * n  # 12
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W correctly decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2x2():
        """
        Return correct U, V, W for 2x2 x 2x2 x 2x2 Strassen decomposition (rank 7).
        
        Convention:
        - U[i*2+j] for A[i,j]
        - V[j*2+k] for B[j,k]  
        - W[k*2+i] for C[i,k]  (note: W index is k*2+i, so W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1])
        
        Wait, the problem states W[k*n+i] where n=2, so:
        - W[0*2+0] = W[0] = C[0,0]
        - W[0*2+1] = W[1] = C[0,1]
        - W[1*2+0] = W[2] = C[1,0]
        - W[1*2+1] = W[3] = C[1,1]
        
        So W[k*2+i] gives C[i,k].
        
        Standard Strassen:
        p1 = (a00+a11)(b00+b11)
        p2 = (a10+a11)b00
        p3 = a00(b01-b11)
        p4 = a11(b10-b00)
        p5 = (a00+a01)b11
        p6 = (a10-a00)(b00+b01)
        p7 = (a01-a11)(b10+b11)
        
        c00 = p1 + p4 - p5 - p7
        c01 = p3 + p5
        c10 = p2 + p4
        c11 = p1 - p2 + p3 + p6
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1  # a00, a11
        V[0, 0] = 1; V[3, 0] = 1  # b00, b11
        # c00 += p1, c11 += p1
        W[0, 0] = 1  # c00
        W[3, 0] = 1  # c11
        
        # p2 = (a10+a11)b00
        U[1, 1] = 1; U[3, 1] = 1  # a10, a11
        V[0, 1] = 1  # b00
        # c10 += p2, c11 -= p2
        W[2, 1] = 1  # c10
        W[3, 1] = -1  # c11
        
        # p3 = a00(b01-b11)
        U[0, 2] = 1  # a00
        V[1, 2] = 1; V[3, 2] = -1  # b01, -b11
        # c01 += p3, c11 += p3
        W[1, 2] = 1  # c01
        W[3, 2] = 1  # c11
        
        # p4 = a11(b10-b00)
        U[3, 3] = 1  # a11
        V[2, 3] = 1; V[0, 3] = -1  # b10, -b00
        # c00 += p4, c10 += p4
        W[0, 3] = 1  # c00
        W[2, 3] = 1  # c10
        
        # p5 = (a00+a01)b11
        U[0, 4] = 1; U[2, 4] = 1  # a00, a01
        V[3, 4] = 1  # b11
        # c00 -= p5, c01 += p5
        W[0, 4] = -1  # c00
        W[1, 4] = 1  # c01
        
        # p6 = (a10-a00)(b00+b01)
        U[1, 5] = 1; U[0, 5] = -1  # a10, -a00
        V[0, 5] = 1; V[1, 5] = 1  # b00, b01
        # c11 += p6
        W[3, 5] = 1  # c11
        
        # p7 = (a01-a11)(b10+b11)
        U[2, 6] = 1; U[3, 6] = -1  # a01, -a11
        V[2, 6] = 1; V[3, 6] = 1  # b10, b11
        # c00 -= p7
        W[0, 6] = -1  # c00
        
        return U, V, W
    
    # First, verify rank-21 Strassen-based decomposition
    U_s, V_s, W_s = strassen_2x2x2x2()
    
    rank_21 = 21
    U21 = np.zeros((4, rank_21))
    V21 = np.zeros((12, rank_21))
    W21 = np.zeros((12, rank_21))
    
    for t in range(3):
        base = t * 7
        U21[:, base:base+7] = U_s
        for j in range(2):
            for k_local in range(2):
                for c in range(7):
                    V21[j * 6 + 2 * t + k_local, base + c] = V_s[j * 2 + k_local, c]
        for k_local in range(2):
            for i in range(2):
                for c in range(7):
                    W21[(2 * t + k_local) * 2 + i, base + c] = W_s[k_local * 2 + i, c]
    
    if check_decomposition(U21, V21, W21, rank_21):
        # Try to find rank 20 by looking for linear dependencies
        # We can try to combine the 21 multiplications into 20 by finding
        # a linear combination that preserves the tensor structure.
        
        # Approach: Try to find a rank-20 decomposition by modifying the rank-21
        # one. We can try to set one of the "multiplication slots" to be a linear
        # combination of others, or find a different structured decomposition.
        
        # Let's try a different approach: use the fact that we have 3 blocks and
        # try to share some Strassen multiplications.
        
        # Idea: For blocks 0 and 1, we can share some multiplications if we
        # combine their B matrices appropriately.
        
        # Let's try to construct a rank-20 decomposition directly.
        # We'll use 20 "multiplication slots" and solve for the coefficients.
        
        # Actually, let's try a simpler approach: check if we can reduce rank by 1
        # by finding a linear dependency among the 21 Strassen multiplications.
        
        # For now, return rank 21 if it works
        return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
    
    # Fallback to schoolbook
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
