"""Matrix multiplication 3x5x7 rank-minimal exact tensor decomposition search."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 7]
START_TIME = time.time()
TIME_LIMIT = 135  # seconds, with safety margin before 150s kill

def _elapsed():
    return time.time() - START_TIME

def _check_time():
    return _elapsed() > TIME_LIMIT

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    sb_rank = n * m * p
    U_sb = np.zeros((n * m, sb_rank))
    V_sb = np.zeros((m * p, sb_rank))
    W_sb = np.zeros((p * n, sb_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Convert to list format for return
    def _mat_to_list(M):
        return M.tolist()
    
    best_result = {
        "U": _mat_to_list(U_sb),
        "V": _mat_to_list(V_sb),
        "W": _mat_to_list(W_sb)
    }
    best_rank = sb_rank
    
    # Try to find lower rank using random initialization and iterative refinement
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Use a structured approach: try ranks from small to larger
    # Key insight: for rectangular matrix multiplication, the rank can be lower than n*m*p
    
    # Strategy: Use random integer matrices and check if they work
    # For small ranks, use structured patterns
    
    # First, let's try a few specific structured approaches
    
    # Approach 1: Block diagonal structure
    # If we can find a block structure where each block handles a subset of (i,j,k)
    
    # Approach 2: Use the fact that this is a tensor rank problem
    # Random search with integer entries in {-2,-1,0,1,2}
    
    np.random.seed(42)
    
    # Try various ranks
    for rank in range(15, sb_rank + 1, 3):
        if _check_time():
            break
        
        # Multiple restarts for each rank
        n_restarts = max(1, min(10, int((TIME_LIMIT - _elapsed()) / (sb_rank - rank + 15) / 2)))
        
        for restart in range(n_restarts):
            if _check_time():
                break
            
            # Random initialization with small integers
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            W = np.random.randint(-1, 2, (p * n, rank))
            
            # Check if this decomposition works
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # We need this to equal 1 for all i,j,k
            
            # Compute the tensor from the decomposition
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if val != 1:
                            error += 1
                            break
                    if error > 0:
                        break
                if error > 0:
                    break
            
            if error == 0:
                # Found a valid decomposition!
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if best_rank < sb_rank:
            break
    
    # If we didn't find anything better, try a smarter structured approach
    if best_rank == sb_rank:
        # Try using Kronecker-like structures
        # For 3x5x7, think of it as combining smaller multiplications
        
        # Approach: Pad to 4x8x8 and use known results
        # Actually, let's try a more systematic approach
        
        # Try rank = 50 (arbitrary lower bound to test)
        for rank in [50, 45, 40, 35, 30]:
            if _check_time():
                break
            
            # Use structured initialization based on block patterns
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Fill with a pattern that might work
            # Each column corresponds to a "basis" element
            for c in range(rank):
                # Random sparse pattern
                u_idx = np.random.randint(0, n * m, size=max(1, rank // 10))
                v_idx = np.random.randint(0, m * p, size=max(1, rank // 10))
                w_idx = np.random.randint(0, p * n, size=max(1, rank // 10))
                
                U[u_idx, c] = np.random.choice([-1, 1], size=len(u_idx))
                V[v_idx, c] = np.random.choice([-1, 1], size=len(v_idx))
                W[w_idx, c] = np.random.choice([-1, 1], size=len(w_idx))
            
            # Quick check
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
    
    return best_result
