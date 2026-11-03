"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def _verify_decomposition(n, m, p, U, V, W, rank):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Build tensor from decomposition
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_vec = U[:, c].reshape(n, m)
        v_vec = V[:, c].reshape(m, p)
        w_vec = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    # Target tensor is all ones
    target = np.ones((n, m, p))
    if not np.allclose(T, target, atol=1e-10):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _try_perturbed_search(n, m, p, target_rank, seed=0, max_attempts=200):
    """
    Try to find a decomposition of target_rank by starting from schoolbook
    and randomly merging columns (setting some entries to zero or adjusting).
    """
    rng = random.Random(seed)
    
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    
    for attempt in range(max_attempts):
        # Start with schoolbook and randomly set some entries to 0 or adjust
        U = [row[:] for row in U_sb]
        V = [row[:] for row in V_sb]
        W = [row[:] for row in W_sb]
        
        # We need to reduce from rank_sb to target_rank
        # Strategy: randomly pick columns to "merge" by setting entries to 0
        # or by combining patterns
        
        # First, try a simple approach: randomly zero out some entries
        # and see if we can still verify (this won't work directly)
        
        # Better: use a structured approach where we try to share columns
        # across different (i,j,k) tuples
        
        # For each column in the reduced rank, assign it to cover multiple (i,j,k)
        # terms by using values other than 1
        
        # Let's try: for target_rank columns, each column covers a subset of (i,j,k)
        # with U[i*m+j,c], V[j*p+k,c], W[k*n+i,c] chosen so that the sum works out
        
        # This is complex. Let's try a different approach:
        # Use the fact that we can have U[i,j,c]=a, V[j,k,c]=b, W[k,i,c]=c
        # such that sum over c of a*b*c = 1 for each (i,j,k)
        
        # Try random assignments with small values
        vals = [0, 0.5, 1, 1.5, 2, 3, -0.5, -1, -1.5, -2, -3]
        
        U_new = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(n*m)]
        V_new = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(m*p)]
        W_new = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(p*n)]
        
        if _verify_decomposition(n, m, p, U_new, V_new, W_new, target_rank):
            return U_new, V_new, W_new
    
    return None

def _try_structured_low_rank(n, m, p, target_rank, seed=0):
    """
    Try structured approaches that exploit the rectangular shape.
    For 3x3x7, we can think of this as 3 blocks of 3x3x(7/3) but 7 isn't divisible.
    
    Alternative: use the fact that the target tensor T[i,j,k]=1 for all i,j,k.
    We need sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k.
    
    One approach: for each j, the sub-tensor T[:,j,:] is 3x7 all-ones.
    We can try to find a low-rank decomposition where columns are shared across j.
    """
    rng = random.Random(seed)
    
    # Try: assign each column to a "type" that determines which (i,j,k) it affects
    # For simplicity, try random but with constraints
    
    vals = [0, 0.5, 1, -0.5, -1]
    
    for attempt in range(100):
        U = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(n*m)]
        V = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(m*p)]
        W = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(p*n)]
        
        if _verify_decomposition(n, m, p, U, V, W, target_rank):
            return U, V, W
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    # Start with schoolbook as fallback
    U_fb, V_fb, W_fb, rank_schoolbook = _schoolbook(n, m, p)
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*p, m*p, n*m) = max(21, 21, 9) = 21
    # But tensor rank is higher. For 3x3x3, best known is 7 (Strassen).
    # For 3x3x7, we expect something like 7 + 3*4 = 19 or similar, but let's search.
    
    # Try ranks from a reasonable lower bound up to schoolbook
    min_rank_to_try = 22  # slightly above theoretical lower bound
    max_rank_to_try = min(rank_schoolbook - 1, 55)
    
    seed_counter = 0
    found_improvement = False
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - 10:  # safety margin
            break
        
        # Try multiple random seeds for this rank
        max_seeds = 50
        for seed in range(seed_counter, seed_counter + max_seeds):
            if time.time() > deadline - 10:
                break
            
            result = _try_perturbed_search(n, m, p, rank, seed=seed, max_attempts=50)
            if result is not None:
                U, V, W = result
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                
                # Verify once more
                if _verify_decomposition(n, m, p, U_list, V_list, W_list, rank):
                    best_rank = rank
                    best_U, best_V, best_W = U_list, V_list, W_list
                    found_improvement = True
                    break
        
        if found_improvement:
            # Found something, continue trying lower ranks
            continue
        else:
            # No improvement at this rank, move to higher rank
            seed_counter += max_seeds
    
    # Also try structured approach with a few seeds
    if time.time() < deadline - 15:
        for seed in range(10):
            if time.time() > deadline - 10:
                break
            for rank in range(max(best_rank - 5, min_rank_to_try), best_rank + 1):
                result = _try_structured_low_rank(n, m, p, rank, seed=seed)
                if result is not None:
                    U, V, W = result
                    U_list = [[float(x) for x in row] for row in U]
                    V_list = [[float(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W]
                    
                    if _verify_decomposition(n, m, p, U_list, V_list, W_list, rank):
                        best_rank = rank
                        best_U, best_V, best_W = U_list, V_list, W_list
                        break
    
    # Convert to proper format - ensure values are exact integers or half-integers
    def clean_values(val):
        # Snap to nearest 0.5 if close
        rounded = round(val * 2) / 2
        if abs(val - rounded) < 1e-10:
            return rounded
        return val
    
    U_final = [[clean_values(x) for x in row] for row in best_U]
    V_final = [[clean_values(x) for x in row] for row in best_V]
    W_final = [[clean_values(x) for x in row] for row in best_W]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
