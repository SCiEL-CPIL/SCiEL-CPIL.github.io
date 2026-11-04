"""Matrix multiplication 2x3 x 3x9 - rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 27
    dim_w = p * n  # 18
    
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    # Build target tensor indices
    # For matrix mult: T[i*m+j, j*p+k, k*n+i] = 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i*m+j, j*p+k, k*n+i))
    
    target_set = set(target_indices)
    
    # Structured decomposition: rank = m*p = 27
    # For each (j,k) pair, create a rank-1 term
    rank_struct = m * p  # 27
    U = np.zeros((dim_u, rank_struct))
    V = np.zeros((dim_v, rank_struct))
    W = np.zeros((dim_w, rank_struct))
    
    col = 0
    for j in range(m):
        for k in range(p):
            for i in range(n):
                U[i * m + j, col] = 1
            V[j * p + k, col] = 1
            for i in range(n):
                W[k * n + i, col] = 1
            col += 1
    
    def verify_full(U, V, W, tol=1e-8):
        """Full verification of the decomposition."""
        for (a, b, c) in target_indices:
            val = np.sum(U[a] * V[b] * W[c])
            if abs(val - 1.0) > tol:
                return False
        for a in range(dim_u):
            for b in range(dim_v):
                for c in range(dim_w):
                    if (a, b, c) not in target_set:
                        val = np.sum(U[a] * V[b] * W[c])
                        if abs(val) > tol:
                            return False
        return True
    
    if verify_full(U, V, W, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook decomposition (rank 54)
    schoolbook_rank = n * m * p  # 54
    U_sb = np.zeros((dim_u, schoolbook_rank))
    V_sb = np.zeros((dim_v, schoolbook_rank))
    W_sb = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    # Try to find a lower rank decomposition using structured approach
    # Key insight: The matrix multiplication tensor has structure that can be exploited
    # by grouping terms based on the middle index j
    
    # Try rank 27 with a different structure: group by k and use linear combinations
    # For each k, we have m terms (one for each j). We can try to combine them.
    
    # Alternative structured approach: Use the fact that for fixed j, the terms over k
    # have a regular structure in W
    
    # Let's try to find rank < 27 by exploiting linearity
    
    # Approach: For each j, the sub-problem involves multiplying a 2x1 vector by a 1x9 vector
    # This has rank at most 2 (since min(2,9)=2). But we need to share structure across j.
    
    # Actually, let's think about this differently.
    # The tensor is T[i*m+j, j*p+k, k*n+i]
    # For fixed i: T[i*m+j, j*p+k, k*n+i] - this couples j and k through the middle index
    
    # Let's try a different structured decomposition
    # Group by i: for each i, we have a 3x9 matrix multiplication (since n=2, each i gives a row)
    # Wait, for fixed i, we're looking at row i of the result.
    # Row i of A (2x3) times B (3x9) gives a 1x9 row.
    # The (i, k) entry is sum_j A[i,j] * B[j,k]
    
    # So for fixed i and k: sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # But W[k*n+i, c] is the same for all j in this sum (for fixed i,k).
    
    # This suggests we can factor out W[k*n+i, c] and get:
    # W[k*n+i, c] * sum_j U[i*m+j, c] * V[j*p+k, c] = 1
    
    # This is a strong constraint. Let's try to exploit it.
    
    # For rank 27, we already have a valid solution. Let's see if we can do better.
    
    # Try rank 18: one term per (k, i) pair
    # For each (k, i), we need: sum_j U[i*m+j, c_{k,i}] * V[j*p+k, c_{k,i}] = 1/W[k*n+i, c_{k,i}]
    # And for other (k', i') with k'!=k or i'!=i, the cross terms must vanish.
    
    # This is getting complex. Let's try a computational search for rank < 27.
    
    # Try ranks from 26 down to 18
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank
    
    for target_rank in range(26, 17, -1):
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of this rank
        # Use a structured initialization based on the rank-27 solution
        # and try to merge terms
        
        # For now, just use the rank-27 solution if target_rank >= 27
        if target_rank >= 27:
            if verify_full(U, V, W, tol=1e-8):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = 27
                break
        
        # Try random search for lower rank
        rng = np.random.RandomState(target_rank)
        found = False
        
        for trial in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U_try = rng.randn(dim_u, target_rank) * 0.1
            V_try = rng.randn(dim_v, target_rank) * 0.1
            W_try = rng.randn(dim_w, target_rank) * 0.1
            
            # Check if this happens to work (very unlikely)
            if verify_full(U_try, V_try, W_try, tol=1e-6):
                found = True
                best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                best_rank = target_rank
                break
        
        if found:
            break
    
    # Return the best result found
    if best_rank < schoolbook_rank:
        return best_result
    
    # If no improvement found, return the rank-27 structured solution
    if verify_full(U, V, W, tol=1e-8):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
