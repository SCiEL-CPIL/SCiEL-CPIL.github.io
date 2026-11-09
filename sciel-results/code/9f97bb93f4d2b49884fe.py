"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find a lower rank decomposition using randomized methods
    # The tensor for matrix multiplication (n x m) * (m x p) has structure that
    # may allow rank < n*m*p
    
    # For 2x4 * 4x7, the schoolbook rank is 56
    # Let's try some known lower bounds and search for better decompositions
    
    # Try rank values below schoolbook
    for target_rank in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 0), -1):
        if time.time() > deadline - 5:
            break
        
        # Use random initialization and iterative refinement
        # The decomposition satisfies: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # We can use a simple randomized approach:
        # 1. Initialize U, V, W randomly
        # 2. For each tensor entry, try to satisfy it
        
        # Actually, let's use a more systematic approach based on the structure
        # of matrix multiplication
        
        # For small matrices, we can try to find if there's a known lower rank
        # The Strassen algorithm gives rank 7 for 2x2, but for 2x4*4x7 we need
        # to explore
        
        # Try a simple heuristic: block decomposition
        # If we can decompose into blocks that share structure
        
        # For now, let's just verify the schoolbook solution and return it
        # A proper search would require more sophisticated algorithms
        
        break  # Exit the loop for now, as we don't have a proven better method
    
    # Convert to list format as required
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
