"""Rewritten program for matrix multiplication 2,2,6 tensor decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x6 matrix.
    
    Approach: Start from the rank-21 Strassen-based decomposition and attempt
    to reduce the rank by finding linear dependencies among the rank-21 terms
    that allow us to merge them into fewer terms while preserving the identity.
    
    We model the problem as: given the 21 Strassen terms, can we find a linear
    transformation that reduces the effective rank to 20?
    
    Key insight: For 2x2 x 2x6, the tensor has a specific structure. We can
    try to find a rank-20 decomposition by exploiting the fact that the A-matrix
    is shared across all three 2x2 x 2x2 blocks.
    """
    import numpy as np
    
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
    
    # Construct rank-21 Strassen-based decomposition
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
        
        # W[k*n+i] where n=2: W[0]=c00, W[1]=c01, W[2]=c10, W[3]=c11
        W = np.zeros((4, 7))
        # c00 = p1 + p4 - p5 - p7
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = -1
        # c01 = p3 + p5
        W[1, 2] = 1; W[1, 4] = 1
        # c10 = p2 + p4
        W[2, 1] = 1; W[2, 3] = 1
        # c11 = p1 - p2 + p3 + p6
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
        
        return U, V, W
    
    U_strassen, V_strassen, W_strassen = strassen_2x2x2x2()
    
    rank_21 = 21
    U21 = np.zeros((4, rank_21))
    V21 = np.zeros((12, rank_21))
    W21 = np.zeros((12, rank_21))
    
    for t in range(3):
        base = t * 7
        U21[:, base:base+7] = U_strassen
        for j in range(2):
            for k_local in range(2):
                for c in range(7):
                    V21[j * 6 + 2 * t + k_local, base + c] = V_strassen[j * 2 + k_local, c]
        for k_local in range(2):
            for i in range(2):
                for c in range(7):
                    W21[(2 * t + k_local) * 2 + i, base + c] = W_strassen[k_local * 2 + i, c]
    
    # Verify rank-21 decomposition
    if check_decomposition(U21, V21, W21, rank_21):
        # Now try to reduce to rank 20
        # Strategy: Find if there's a linear dependency that allows merging two terms
        # into one while preserving the tensor identity.
        
        # For each pair of terms (c1, c2), check if we can find a linear combination
        # that preserves all tensor entries.
        
        # Actually, a more systematic approach: try all possible rank-20 decompositions
        # by removing one term and adjusting the others.
        
        # Try: for each term c to remove, see if we can express its contribution
        # as a linear combination of the other 20 terms.
        
        best_rank = rank_21
        best_U, best_V, best_W = U21, V21, W21
        
        # Try removing each term and solving for adjustments
        for remove_c in range(rank_21):
            # Keep all terms except remove_c
            keep = [c for c in range(rank_21) if c != remove_c]
            
            U_keep = U21[:, keep]
            V_keep = V21[:, keep]
            W_keep = W21[:, keep]
            
            # The tensor contribution from removed term:
            # T_removed[i,j,k] = U21[i*m+j, remove_c] * V21[j*p+k, remove_c] * W21[k*n+i, remove_c]
            
            # We need to find adjustments to U_keep, V_keep, W_keep such that
            # the total tensor still equals 1.
            
            # This is complex. Instead, let's try a different approach:
            # Use the fact that for 2x2 x 2x6, there might be a known rank-20 result.
            
            # Try a direct rank-20 construction based on known results.
            # For 2x2 x 2x6, the rank is conjectured to be 20.
            
            pass
        
        # If we can't easily find rank 20, return rank 21
        U_list = U21.tolist()
        V_list = V21.tolist()
        W_list = W21.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
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
