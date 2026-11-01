"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to stay safely under 150s
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition of rank n*m*p
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    # Strategy: Use block decomposition with Strassen for 2x2x2 sub-problems
    # For 2x3x6, we can split the inner dimension (m=3) into parts and use Strassen where possible.
    # However, a simpler approach:
    # The tensor for 2x3x6 can be seen as two copies of 2x3x3 (since p=6=2*3)
    # If we can find a low-rank decomposition for 2x3x3, we can use it twice.
    # But even better: use the known rank for small matrix multiplications.
    # Rank(2x2x2) = 7 (Strassen)
    # Rank(2x3x3) is known to be 11 or 12? Let's assume we can construct a rank 11 or 12.
    # Actually, let's try to construct a decomposition using Strassen on 2x2 blocks.
    
    # Alternative strategy: Direct construction using known results.
    # For 2x3x6, the rank is at most 2 * Rank(2x3x3).
    # If Rank(2x3x3) = 11, then Rank(2x3x6) <= 22.
    # If Rank(2x3x3) = 12, then Rank(2x3x6) <= 24.
    # Schoolbook is 36.
    
    # Let's try to find a rank-11 or rank-12 decomposition for 2x3x3 first.
    # Then we can use it for the two 2x3x3 blocks in the 2x3x6 problem.
    
    # But wait, the dimensions are U(n*m, rank), V(m*p, rank), W(p*n, rank).
    # For 2x3x3: U(6, r), V(9, r), W(6, r).
    # For 2x3x6: U(6, R), V(18, R), W(12, R).
    
    # We can view the 2x3x6 multiplication as:
    # C = A * B, where A is 2x3, B is 3x6.
    # Split B into two 3x3 matrices B1 and B2.
    # C1 = A * B1, C2 = A * B2.
    # So we need two independent decompositions of 2x3x3.
    # If we have a rank-r decomposition for 2x3x3, we can use it twice to get rank 2r for 2x3x6.
    
    # Let's try to find a low-rank decomposition for 2x3x3.
    # Known: Rank(2x2x2) = 7.
    # Can we extend Strassen to 2x3x3?
    
    # Let's try a random search for a low-rank decomposition of 2x3x3.
    # But that might be too slow.
    
    # Instead, let's use a structured approach.
    # We know that 2x2x2 has rank 7.
    # For 2x3x3, we can think of it as 2x2x3 plus 2x1x3.
    # This is getting complicated.
    
    # Let's try a different approach: use the fact that the rank of matrix multiplication
    # is multiplicative in a certain sense for block diagonal structures.
    
    # Actually, the simplest improvement is to use Strassen's algorithm directly on the 2x3x6 problem
    # by padding to 4x4 or using the 2x2x2 Strassen decomposition on sub-blocks.
    
    # Let's try to construct a decomposition of rank 33 (as hinted in the knowledge).
    # The knowledge says: "Using block decomposition with Strassen's algorithm for 2x2x2 sub-problems
    # reduces the rank from 36 to 33 by exploiting the structure of the 2x3x2 multiplication (rank 11 vs 12)."
    
    # So, we can split the 3x6 matrix B into three 3x2 matrices B1, B2, B3.
    # Then C = A * B = [A*B1, A*B2, A*B3].
    # Each A*Bj is a 2x3 * 3x2 multiplication.
    # If we can find a rank-11 decomposition for 2x3x2, then we can use it three times to get rank 33 for 2x3x6.
    
    # Let's try to find a rank-11 decomposition for 2x3x2.
    # U(6, 11), V(6, 11), W(4, 11).
    
    # We can try to construct this using Strassen on 2x2x2 sub-problems.
    # 2x3x2 can be seen as 2x2x2 plus 2x1x2.
    # The 2x2x2 part has rank 7.
    # The 2x1x2 part has rank 2 (since it's just two scalar multiplications).
    # But they might share some rank-1 tensors.
    
    # Let's try to find a rank-11 decomposition for 2x3x2 using random search.
    # We'll use a simple random search with gradient descent or just random initialization.
    
    # Actually, let's just try to return the schoolbook decomposition for now,
    # but with a rank of 33 if we can find a valid decomposition.
    
    # For now, let's just return the schoolbook decomposition.
    # The improvement to rank 33 would require finding a valid rank-11 decomposition for 2x3x2.
    
    # Let's try a quick random search for a rank-11 decomposition of 2x3x2.
    # If we find one, we can use it three times to get rank 33 for 2x3x6.
    
    def find_2x3x2_decomposition(rank=11, max_iter=1000):
        """Try to find a rank-r decomposition for 2x3x2 matrix multiplication."""
        n2, m2, p2 = 2, 3, 2
        # Target tensor: T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,2)
        # We need sum_r U[i,r] V[j,r] W[k,r] = 1 for all i,j,k
        
        for attempt in range(max_iter):
            # Initialize with random values
            U = np.random.randn(n2 * m2, rank)
            V = np.random.randn(m2 * p2, rank)
            W = np.random.randn(p2 * n2, rank)
            
            # Check if the decomposition is valid
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Note: U is indexed by (i*m2+j), V by (j*p2+k), W by (k*n2+i)
            
            valid = True
            for i in range(n2):
                for j in range(m2):
                    for k in range(p2):
                        val = sum(U[i*m2+j, r] * V[j*p2+k, r] * W[k*n2+i, r] for r in range(rank))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return U, V, W
        
        return None
    
    # Try to find a rank-11 decomposition for 2x3x2
    # But this is unlikely to succeed with random search in a reasonable time.
    # Let's just return the schoolbook decomposition.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
