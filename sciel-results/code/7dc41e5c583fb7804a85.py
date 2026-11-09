"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s out of 150s limit
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p
    U_schoolbook = [[0] * rank_schoolbook for _ in range(n * m)]
    V_schoolbook = [[0] * rank_schoolbook for _ in range(m * p)]
    W_schoolbook = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j][col] = 1
                V_schoolbook[j * p + k][col] = 1
                W_schoolbook[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    best_result = {"U": U_schoolbook, "V": V_schoolbook, "W": W_schoolbook}
    best_rank = rank_schoolbook
    
    # Strategy: Try small ranks with random integer/half-integer entries
    # and verify exactness
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        # The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # should equal delta_{i,j,k} for matrix mult... 
        # Actually, the decomposition is: (AB)[i,k] = sum_j A[i,j] B[j,k]
        # In tensor form: C[i,k] = sum_j A[i,j] B[j,k]
        # The rank-r decomposition means: C[i,k] = sum_c U[i,c] * V[k,c] ... 
        # Wait, let me re-read the problem.
        
        # "Find a rank-minimal exact tensor decomposition for multiplying a 3 by 5 matrix 
        #  by a 5 by 6 matrix. The output factor follows the official trace convention 
        #  W[k*n+i,r]."
        
        # So we have matrices A (3x5) and B (5x6), product C = A*B (3x6)
        # The tensor for matrix multiplication has indices (i, j, k) where:
        #   i: row of A (0..n-1)
        #   j: column of A / row of B (0..m-1) 
        #   k: column of B (0..p-1)
        # The tensor T[i,j,k] = 1 if this term contributes to C[i,k]
        
        # Actually, the standard bilinear form: C[i,k] = sum_j A[i,j] * B[j,k]
        # So the "tensor" is T[i,j,k] = delta (it's just the identity in j)
        
        # For a rank-r decomposition:
        # C[i,k] = sum_{c=1}^{r} U[i,c] * V[k,c] ... no that's not right either.
        
        # Let me think again. The matrix multiplication tensor T is defined by:
        # T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
        # No wait, that's not a tensor in the usual sense.
        
        # The bilinear map: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
        # This can be written as: C = sum_j (A[:,j] outer B[j,:])
        # So it's a sum of m rank-1 tensors.
        
        # For a rank-r decomposition of this bilinear map:
        # C[i,k] = sum_{c=1}^{r} sum_j U[i,j,c] * V[j,k,c]
        # where U has shape (n, m, r) and V has shape (m, p, r)
        
        # But the problem says U has shape (n*m, rank), V has shape (m*p, rank), 
        # W has shape (p*n, rank)
        
        # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), 
        # (p*n, rank) with integer/half-integer entries that exactly decompose 
        # the matrix multiplication tensor"
        
        # So the decomposition is: for each c in 1..rank,
        #   contribution to C[i,k] = sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, that doesn't make sense dimensionally.
        
        # Actually, I think the convention is:
        # C[i,k] = sum_{c=1}^{rank} U[i*m + ?, c] * V[? * p + k, c] * W[k*n + i, c]
        
        # Hmm, let me look at the schoolbook example:
        # U[i*m+j][c] = 1, V[j*p+k][c] = 1, W[k*n+i][c] = 1
        # So for each (i,j,k), there's one column c where all three are 1.
        # Then C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But which j? It seems like for each c, we need to know which j it corresponds to.
        
        # I think the correct interpretation is:
        # C[i,k] = sum_{j=0}^{m-1} sum_{c=1}^{rank} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # No, that would give rank m*rank...
        
        # Let me try: C[i,k] = sum_{c=1}^{rank} (sum_{j=0}^{m-1} U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
        # That doesn't work either.
        
        # Actually, I think it's simpler. The tensor T[i,j,k] for matrix multiplication is:
        # T[i,j,k] = 1 (for all i,j,k)
        # And a rank-r decomposition is: T[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, but that would mean each c contributes a rank-1 tensor, and the sum should equal T.
        
        # Let's check with schoolbook: For each (i,j,k), there's exactly one c where 
        # U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1. All other c have at least one zero.
        # So T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1. Correct!
        
        # So the condition is: For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
        #   sum_{c=1}^{rank} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Let's verify this is the right condition.
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try various ranks from low to high
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Also rank >= max(n, m, p) * something... 
    # For 3x5x6, the trivial bound is 90. 
    # Strassen-like algorithms might help, but for these small sizes, 
    # let's try random search.
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: The tensor has n*m*p = 90 "terms" but they can be combined.
    # Actually, a better lower bound: rank >= max(n*p, m*p, n*m) = 30
    # But we can do better: rank >= (n*m*p) / (something)
    
    # Let's try ranks from 30 up to 90, but focus on promising ranges
    # For efficiency, try a few ranks with many random attempts each
    
    random.seed(42)
    
    # Try different rank values
    ranks_to_try = list(range(30, 91))
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        
        if rank >= best_rank:
            continue
        
        # Try multiple random initializations
        max_attempts = 50 if rank < 50 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random U, V, W with integer or half-integer entries
            # Use small values to increase chance of exactness
            U_rand = np.zeros((n * m, rank), dtype=float)
            V_rand = np.zeros((m * p, rank), dtype=float)
            W_rand = np.zeros((p * n, rank), dtype=float)
            
            # Fill with random small integers or half-integers
            for i in range(n * m):
                for c in range(rank):
                    if random.random() < 0.5:  # 50% chance of nonzero
                        val = random.choice([-1, -0.5, 0.5, 1])
                        U_rand[i, c] = val
            
            for j in range(m * p):
                for c in range(rank):
                    if random.random() < 0.5:
                        val = random.choice([-1, -0.5, 0.5, 1])
                        V_rand[j, c] = val
            
            for k in range(p * n):
                for c in range(rank):
                    if random.random() < 0.5:
                        val = random.choice([-1, -0.5, 0.5, 1])
                        W_rand[k, c] = val
            
            # Check if this decomposes correctly
            if check_decomposition(U_rand, V_rand, W_rand, rank):
                # Convert to lists with exact integer/half-integer representation
                U_list = [[int(U_rand[i, c]) if abs(U_rand[i, c] - round(U_rand[i, c])) < 1e-9 
                          else U_rand[i, c] for c in range(rank)] for i in range(n * m)]
                V_list = [[int(V_rand[j, c]) if abs(V_rand[j, c] - round(V_rand[j, c])) < 1e-9 
                          else V_rand[j, c] for c in range(rank)] for j in range(m * p)]
                W_list = [[int(W_rand[k, c]) if abs(W_rand[k, c] - round(W_rand[k, c])) < 1e-9 
                          else W_rand[k, c] for c in range(rank)] for k in range(p * n)]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                print(f"Found rank {rank} decomposition!")
                break  # Move to next (lower) rank
        
        if rank < best_rank:
            continue
    
    return best_result
# EVOLVE-BLOCK-END
