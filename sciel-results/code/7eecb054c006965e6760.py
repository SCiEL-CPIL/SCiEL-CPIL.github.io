"""Matrix multiplication 3x4x8 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, n*p), dtype=np.int64)
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
                entries.append((i, j, k))
    
    deadline = time.time() + 140
    
    # Schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school), dtype=np.int64)
    V_school = np.zeros((m*p, rank_school), dtype=np.int64)
    W_school = np.zeros((n*p, rank_school), dtype=np.int64)
    col = 0
    for (i, j, k) in entries:
        U_school[i*m+j, col] = 1
        V_school[j*p+k, col] = 1
        W_school[k*n+i, col] = 1
        col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower ranks using a smarter approach
    # For each target rank, try to find a valid decomposition
    for target_rank in range(95, 85, -1):
        if time.time() > deadline:
            break
        
        # Use a greedy approach with backtracking
        # Assign each entry to a column, ensuring consistency
        for restart in range(50):
            if time.time() > deadline:
                break
            
            U = np.zeros((n*m, target_rank), dtype=np.int64)
            V = np.zeros((m*p, target_rank), dtype=np.int64)
            W = np.zeros((n*p, target_rank), dtype=np.int64)
            
            # Random permutation of entries
            perm = np.random.permutation(len(entries))
            valid = True
            
            for idx in perm:
                i, j, k = entries[idx]
                a, b, c = i*m+j, j*p+k, k*n+i
                
                # Find a column where we can place this entry
                # The column must have U[a, col] * V[b, col] * W[c, col] = 1
                # Currently, we want to add 1 to this position
                # So we need a column where the current product is 0
                # and after adding, it becomes 1
                
                # Try each column
                placed = False
                for col in range(target_rank):
                    # Check if adding to this column keeps things consistent
                    # The new value at (a,b,c) will be:
                    # old_sum + U[a,col]*V[b,col]*W[c,col]
                    # We want this to equal 1
                    # Currently, the sum over all columns is 0 (since we haven't placed this entry yet)
                    # So we need U[a,col]*V[b,col]*W[c,col] = 1
                    # This means U[a,col] = V[b,col] = W[c,col] = 1 or all -1, etc.
                    
                    # For simplicity, try to set U[a,col] = V[b,col] = W[c,col] = 1
                    # But this might conflict with other entries
                    
                    # Check if setting these to 1 would cause any conflict
                    # An entry (i',j',k') is placed in column col if U[i'*m+j',col] != 0
                    # The contribution to position (a',b',c') is sum_col U[a',col]*V[b',col]*W[c',col]
                    
                    # This is getting too complex. Let's just try it and check at the end.
                    U[a, col] += 1
                    V[b, col] += 1
                    W[c, col] += 1
                    placed = True
                    break
                
                if not placed:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if the decomposition is valid
            T_recon = np.einsum('ar,br,cr->abc', U.astype(float), V.astype(float), W.astype(float))
            if np.array_equal(T_recon.astype(int), T):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < rank_school:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
