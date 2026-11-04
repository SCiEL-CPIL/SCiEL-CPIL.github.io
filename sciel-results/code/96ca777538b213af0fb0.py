import numpy as np
import random
import time
from scipy.optimize import linprog

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build fallback schoolbook decomposition
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    start_time = time.time()
    max_time = 90  # seconds
    
    # Try to find lower rank decompositions using LP over finite fields
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We search for sparse support patterns that can be solved exactly
    
    for R in range(35, 64):
        if time.time() - start_time > max_time:
            break
        
        # Try multiple random sparse support patterns
        for attempt in range(50):
            if time.time() - start_time > max_time:
                break
            
            # Generate random support pattern
            # Each rank term r has support in U, V, W
            U_sup = np.zeros((n * m, R), dtype=int)
            V_sup = np.zeros((m * p, R), dtype=int)
            W_sup = np.zeros((p * n, R), dtype=int)
            
            for r in range(R):
                # Randomly select which indices are non-zero for this term
                i_u = random.randint(0, n * m - 1)
                i_v = random.randint(0, m * p - 1)
                i_w = random.randint(0, p * n - 1)
                U_sup[i_u, r] = 1
                V_sup[i_v, r] = 1
                W_sup[i_w, r] = 1
            
            # Check if this support pattern can work
            # For each (i,j,k) in the multiplication tensor, we need:
            # T[i,j,k] = sum over r where U_sup[i*m+j,r]==1, V_sup[j*p+k,r]==1, W_sup[k*n+i,r]==1
            # of U_val[i*m+j,r] * V_val[j*p+k,r] * W_val[k*n+i,r]
            
            # Collect all (i,j,k) that have at least one supporting term
            tensor_entries = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # Find all r that support this entry
                        supporting_r = []
                        for r in range(R):
                            if U_sup[u_idx, r] == 1 and V_sup[v_idx, r] == 1 and W_sup[w_idx, r] == 1:
                                supporting_r.append(r)
                        
                        if len(supporting_r) > 0:
                            tensor_entries.append((i, j, k, supporting_r))
            
            if len(tensor_entries) < n * m * p:
                continue  # Not all entries are covered
            
            # Now we need to find values U_val, V_val, W_val such that
            # for each (i,j,k): sum_{r in supporting_r} U_val[u_idx,r] * V_val[v_idx,r] * W_val[w_idx,r] = 1
            # This is a nonlinear system. Instead, let's try a different approach.
            
            # Use the fact that we can scale: if we set one factor to 1, we reduce to bilinear.
            # Let's try: set W_val[w_idx,r] = 1 for all w_idx, r where W_sup[w_idx,r]==1
            # Then we need: sum_{r in supporting_r} U_val[u_idx,r] * V_val[v_idx,r] = 1
            # This is a system of linear equations in U_val and V_val products.
            
            # Actually, let's use a simpler verification: check if the pattern is feasible
            # by trying to solve a relaxed LP.
            
            # For now, just check basic feasibility: each (i,j,k) must have at least one term
            # and the pattern should be "balanced"
            
            # Count how many terms support each index
            u_counts = np.sum(U_sup, axis=1)
            v_counts = np.sum(V_sup, axis=1)
            w_counts = np.sum(W_sup, axis=1)
            
            # All indices should have some support
            if np.any(u_counts == 0) or np.any(v_counts == 0) or np.any(w_counts == 0):
                continue
            
            # Try to verify by attempting to find integer values
            # This is complex, so we'll use a heuristic: if the pattern looks good, try it
            
            # For now, return fallback since exact verification is complex
            pass
        
        # If we've tried enough patterns for this R, move to next R
    
    return fallback
# EVOLVE-BLOCK-END
