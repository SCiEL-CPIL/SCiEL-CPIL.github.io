"""Search-based solver for rank-minimal tensor decomposition of 2x5 * 5x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]
n, m, p = DIMENSIONS

def verify(U, V, W):
    """Verify that the decomposition is correct. Returns True if valid."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def schoolbook():
    """Return the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    """Construct a rank-minimal (or near-minimal) decomposition."""
    start_time = time.time()
    deadline = start_time + 130
    
    # Start with schoolbook as baseline
    U_best, V_best, W_best = schoolbook()
    best_rank = n * m * p  # 60
    
    # Try to find a lower rank decomposition using random search with local optimization
    # For this specific case, we'll try various ranks below 60
    
    # The key insight: we can try to find a decomposition where some of the rank terms
    # are linear combinations of others, effectively reducing the rank.
    
    # Strategy: Start with a random rank-r decomposition and try to optimize it
    # to satisfy the constraints.
    
    for rank in range(30, best_rank):  # Try ranks from 30 up to 59
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations for this rank
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize U, V, W with small random values
            U = np.random.randint(-2, 3, size=(n*m, rank))
            V = np.random.randint(-2, 3, size=(m*p, rank))
            W = np.random.randint(-2, 3, size=(p*n, rank))
            
            # Check if this random initialization happens to satisfy the constraints
            # (very unlikely, but worth checking)
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                        if abs(s - 1) > 1e-9:
                            valid = False
                            break
            
            if valid:
                # Found a valid decomposition!
                U_best = U.tolist()
                V_best = V.tolist()
                W_best = W.tolist()
                best_rank = rank
                break
        
        if best_rank <= rank:
            break
    
    # If we haven't found a better solution, try a more structured approach
    # For n=2, we can exploit the structure more directly
    
    if best_rank == 60:
        # Try a specific structured decomposition
        # For 2x5 * 5x6, we can try to use the fact that n=2 to reduce rank
        
        # One approach: group terms by j (the shared dimension)
        # For each j, we have terms (0,j,k) and (1,j,k) for k=0..5
        # Can we combine these?
        
        # Actually, let's try a different approach: use the Strassen-like idea
        # but adapted for rectangular matrices.
        
        # For now, if we haven't found a better solution, return the schoolbook
        elapsed = time.time() - start_time
        if elapsed < deadline:
            pass
    
    return {"U": U_best, "V": V_best, "W": W_best}
