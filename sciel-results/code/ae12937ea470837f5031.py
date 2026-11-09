"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Try to find a low-rank decomposition
    # Start with schoolbook and try to improve
    
    # Schoolbook: rank = n*m*p = 40
    # Let's try to use structure to reduce rank
    
    # Approach: Split the 2x4 x 4x5 into two 2x2 x 2x5 multiplications
    # A = [A1 A2] where A1, A2 are 2x2
    # B = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2
    
    # For each 2x2 x 2x5 multiplication, find a low-rank decomposition
    
    def decomp_2x2x5():
        """Decompose 2x2 x 2x5 matrix multiplication.
        A is 2x2, B is 2x5, C is 2x5.
        C[i,k] = sum_j A[i,j]*B[j,k]
        Returns U, V, W as lists of lists with rank columns.
        U shape: (2*2, rank), V shape: (2*5, rank), W shape: (5*2, rank)
        W uses convention W[k*n+i, c] where n=2
        """
        # Schoolbook for 2x2x5: rank 20
        # Let's try to do better
        
        # Using known results, 2x2xp has rank 3p-1 for p>=2
        # So 2x2x5 should have rank 14
        
        # Construction for 2x2xp with rank 3p-1:
        # This is a known result. Let me construct it.
        
        # For 2x2xp:
        # A = [[a0, a1], [a2, a3]]
        # B = [[b0, b1, ..., b_{p-1}], [c0, c1, ..., c_{p-1}]]
        # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
        
        # Rank 3p-1 construction:
        # We'll use the following approach:
        # For k=0: use 4 terms (a0*b0, a1*c0, a2*b0, a3*c0)
        # For k=1..p-1: use 3 terms each by reusing
        
        # Actually, let me use a different known construction.
        # The rank of 2x2xp is 3p-1. Here's one construction:
        
        p_local = 5
        rank_local = 3 * p_local - 1  # 14
        
        U = [[0] * rank_local for _ in range(4)]   # 2*2 = 4 rows
        V = [[0] * rank_local for _ in range(10)]  # 2*5 = 10 rows
        W = [[0] * rank_local for _ in range(10)]  # 5*2 = 10 rows
        
        # Let me use a concrete construction
        # For 2x2xp, one approach:
        # 
        # M0 = a0 * b0
        # M1 = a1 * c0  
        # M2 = a2 * b0
        # M3 = a3 * c0
        # For k >= 1:
        # M_{3+k} = (a0 + a1 + a2) * (b_k - b_{k-1})  -- this doesn't quite work
        # 
        # Let me try a different approach. I'll use the fact that we can write:
        # C[0,k] = a0*b_k + a1*c_k
        # C[1,k] = a2*b_k + a3*c_k
        #
        # Define:
        # X_k = b_k + c_k
        # Y_k = b_k - c_k
        # Then:
        # C[0,k] = a0*b_k + a1*c_k = (a0+a1)/2 * X_k + (a0-a1)/2 * Y_k
        # C[1,k] = a2*b_k + a3*c_k = (a2+a3)/2 * X_k + (a2-a3)/2 * Y_k
        #
        # So for each k, we need 2 products: (a0+a1)*X_k, (a0-a1)*Y_k, etc.
        # But we can share the (a0+a1) and (a0-a1) factors...
        # 
        # Actually, let me just use 4 terms per k for simplicity: rank 20
        # Or try to be smarter.
        
        # Let me try: for each k, use 2 terms with precomputed combinations
        # P0 = a0 + a1, P1 = a0 - a1, P2 = a2 + a3, P3 = a2 - a3
        # For each k: 
        #   C[0,k] = (P0/2)*X_k + (P1/2)*Y_k where X_k=b_k+c_k, Y_k=b_k-c_k
        #   C[1,k] = (P2/2)*X_k + (P3/2)*Y_k
        # 
        # This uses 4 "products" per k if we count P0*X_k, P1*Y_k, P2*X_k, P3*Y_k
        # But P0, P1, P2, P3 are linear combinations of A entries, and X_k, Y_k are linear combos of B entries
        # So each term is rank 1: (linear combo of A) * (linear combo of B)
        # Total: 4 * 5 = 20 terms. Same as schoolbook!
        
        # Let me try yet another approach. Use the known 3p-1 result.
        # For 2x2xp, rank is 3p-1. Here's a construction from the literature:
        
        # I'll just use schoolbook for now (rank 20) and combine two to get rank 40
        # Then try to improve later.
        
        # Actually, let me just implement schoolbook for the full 2x4x5 and return rank 40
        # This is the baseline. I'll try to improve in subsequent iterations.
        
        return None  # Placeholder
    
    # Just do schoolbook for now
    rank = n * m * p  # 40
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
