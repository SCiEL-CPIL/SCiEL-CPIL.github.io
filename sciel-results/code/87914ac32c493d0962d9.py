"""Rewritten program for matrix multiplication 2,2,6 tensor decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    We use an exhaustive/structured search over integer/half-integer entries
    to find a decomposition of rank < 24 (the schoolbook rank).
    
    The tensor T[i,j,k] = e_{i,k} in the output (where input A is 2x2, B is 2x6,
    output C is 2x6 with C[i,k] = sum_j A[i,j]*B[j,k]).
    
    We seek U (4 x r), V (12 x r), W (12 x r) such that for all (i,j,k):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Strategy: Try to construct a decomposition with rank 20, 21, 22, 23.
    We use a structured approach inspired by Strassen-style block decompositions
    and algebraic identities.
    
    For 2x2 x 2x6, we can think of it as two 2x2 x 2x2 multiplications (for
    columns 0-1 and 4-5 of B, and columns 2-3 of B) plus cross terms.
    
    Actually, let's use a more systematic approach:
    - The tensor has symmetry: A is 2x2, B is 2x6.
    - We can partition B into three 2x2 blocks: B = [B1 | B2 | B3] where each
      Bi is 2x2.
    - Then C = A * B = [A*B1 | A*B2 | A*B3].
    - Each A*Bi is a 2x2 x 2x2 multiplication.
    
    If we could do each 2x2 x 2x2 in rank 7 (Strassen), total would be 21.
    But we can share structure: A is the same for all three products.
    
    Let's try to find a rank-20 decomposition using a linear algebra approach.
    
    Approach: Set up the bilinear constraints and solve.
    For rank r, we need U (4xr), V (12xr), W (12xr) with entries in Z or Z+1/2.
    
    We'll try r = 20, 21, 22, 23 and use a structured initialization.
    """
    import numpy as np
    from itertools import product
    
    n, m, p = 2, 2, 6
    nm = n * m  # 4
    mp = m * p  # 12
    pn = p * n  # 12
    
    # The tensor T has entries T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k
    # i in 0..1, j in 0..1, k in 0..5
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W correctly decompose the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # For each (i, j, k), compute sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to build a decomposition using a structured approach.
    # Key insight: We can use the fact that A is 2x2 and try to exploit
    # the linearity in A's entries.
    
    # Let's try a direct construction with rank 20.
    # We'll use a pattern based on Strassen's algorithm for 2x2x2x2.
    
    # For a single 2x2 x 2x2 multiplication, Strassen uses 7 multiplications:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    
    # For 2x2 x 2x6, we have 3 such blocks. If we naively use Strassen for each:
    # rank = 7 * 3 = 21.
    
    # Can we do better? Let's try rank 20 by finding a shared structure.
    
    # Actually, let's first verify rank 21 works, then try to improve.
    
    def strassen_2x2x2x2():
        """Return U, V, W for 2x2 x 2x2 x 2x2 Strassen decomposition (rank 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # p1 = (a00+a11)(b00+b11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # p2 = (a10+a11)b00
        U[1, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = 1
        
        # p3 = a00(b01-b11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[0, 2] = 1; W[3, 2] = 1
        
        # p4 = a11(b10-b00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[1, 3] = 1; W[3, 3] = 1
        
        # p5 = (a00+a01)b11
        U[0, 4] = 1; U[2, 4] = 1
        V[3, 4] = 1
        W[0, 4] = 1; W[3, 4] = 1
        
        # p6 = (a10-a00)(b00+b01)
        U[1, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[1, 5] = 1; W[3, 5] = 1
        
        # p7 = (a01-a11)(b10+b11)
        U[2, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        # Reconstruct W based on C = AB formulas:
        # c00 = p1 + p4 - p5 - p7
        # c01 = p3 + p5
        # c10 = p2 + p4
        # c11 = p1 - p2 + p3 + p6
        # Indices: W[k*2+i]
        W = np.zeros((4, 7))
        # c00 (k=0, i=0) -> W[0]
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
        # c01 (k=0, i=1) -> W[1]
        W[1, 2] = 1; W[1, 4] = 1
        # c10 (k=1, i=0) -> W[2]
        W[2, 1] = 1; W[2, 3] = 1
        # c11 (k=1, i=1) -> W[3]
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
        
        return U, V, W
    
    # For 2x2 x 2x6, we have 3 blocks of 2x2 x 2x2.
    # B has columns 0-1, 2-3, 4-5.
    # For block t (t=0,1,2), B_t is columns 2t, 2t+1.
    # V index for block t, local col k_local: V[j*6 + (2t + k_local), c]
    
    U_strassen, V_strassen, W_strassen = strassen_2x2x2x2()
    
    rank_21 = 21
    U = np.zeros((4, rank_21))
    V = np.zeros((12, rank_21))
    W = np.zeros((12, rank_21))
    
    for t in range(3):
        base = t * 7
        # U is the same for all blocks (A is shared)
        U[:, base:base+7] = U_strassen
        # V: for block t, local V[j*2+k_local] maps to global V[j*6 + 2t + k_local]
        for j in range(2):
            for k_local in range(2):
                for c in range(7):
                    V[j * 6 + 2 * t + k_local, base + c] = V_strassen[j * 2 + k_local, c]
        # W: for block t, local W[k_local*2+i] maps to global W[(2t+k_local)*2+i]
        for k_local in range(2):
            for i in range(2):
                for c in range(7):
                    W[(2 * t + k_local) * 2 + i, base + c] = W_strassen[k_local * 2 + i, c]
    
    # Verify rank-21 decomposition
    if check_decomposition(U, V, W, rank_21):
        # Convert to lists
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If rank 21 doesn't work (shouldn't happen), fall back to schoolbook
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
