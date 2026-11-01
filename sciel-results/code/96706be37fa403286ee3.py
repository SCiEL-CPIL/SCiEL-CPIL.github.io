import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Recursive Strassen decomposition
    # We build the U, V, W matrices recursively.
    # Base case: p == 2 (2x2x2 multiplication)
    # Recursive case: p > 2, split p into p//2
    
    def get_strassen_2x2x2():
        U = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]
        V = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]
        W = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ]
        return U, V, W

    def build_decomposition(n, m, p):
        """
        Returns U, V, W for n x m * m x p
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        """
        if p == 2:
            U_base, V_base, W_base = get_strassen_2x2x2()
            # Convert to lists of lists if needed, or keep as is
            return [list(row) for row in U_base], [list(row) for row in V_base], [list(row) for row in W_base]
        
        # Split p into two halves
        p1 = p // 2
        p2 = p - p1
        
        # Recursively get decompositions for the two halves
        U1, V1, W1 = build_decomposition(n, m, p1)
        U2, V2, W2 = build_decomposition(n, m, p2)
        
        rank1 = len(U1[0]) if U1 else 0
        rank2 = len(U2[0]) if U2 else 0
        total_rank = rank1 + rank2
        
        # Initialize U, V, W
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Fill U
        # U is the same for both halves because A is the same.
        # However, in the tensor decomposition, we sum over r.
        # T = sum U_i(r) V_j(r) W_k(r)
        # For the first half (columns 0..p1-1 of B, 0..p1-1 of C):
        # The terms correspond to U1, V1, W1.
        # For the second half (columns p1..p-1 of B, p1..p-1 of C):
        # The terms correspond to U2, V2, W2.
        # Note: U1 and U2 are identical (both are Strassen U for 2x2x2 scaled? No, just the same linear combos of A).
        # So U columns 0..rank1-1 are U1.
        # U columns rank1..total_rank-1 are U2 (which is same values as U1).
        
        for i in range(n * m):
            for r in range(rank1):
                U[i][r] = U1[i][r]
            for r in range(rank2):
                U[i][rank1 + r] = U2[i][r]
                
        # Fill V
        # V1 corresponds to B[:, 0:p1]
        # V2 corresponds to B[:, p1:p]
        for j in range(m * p1):
            for r in range(rank1):
                V[j][r] = V1[j][r]
                
        for j in range(m * p2):
            # Global index in V: m * p1 + j
            for r in range(rank2):
                V[m * p1 + j][rank1 + r] = V2[j][r]
                
        # Fill W
        # W1 corresponds to C[:, 0:p1]
        # W2 corresponds to C[:, p1:p]
        # Indexing for W: W[k*n + i, r]
        # k is column index.
        # For first half: k in 0..p1-1
        # For second half: k in p1..p-1
        
        # Map local W indices to global
        # Local W1 index: k1*n + i, where k1 in 0..p1-1
        # Global W index: k1*n + i
        
        # Local W2 index: k2*n + i, where k2 in 0..p2-1
        # Global W index: (k1_offset + k2)*n + i = (p1 + k2)*n + i
        
        for k1 in range(p1):
            for i in range(n):
                global_idx = k1 * n + i
                local_idx = k1 * n + i
                for r in range(rank1):
                    W[global_idx][r] = W1[local_idx][r]
                    
        for k2 in range(p2):
            for i in range(n):
                global_idx = (p1 + k2) * n + i
                local_idx = k2 * n + i
                for r in range(rank2):
                    W[global_idx][rank1 + r] = W2[local_idx][r]
                    
        return U, V, W

    U, V, W = build_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
