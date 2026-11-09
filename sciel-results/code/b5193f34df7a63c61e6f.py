import math
import time
import numpy as np
from scipy.optimize import differential_evolution

def _score(U):
    """
    Compute the score for a finite set U of nonnegative integers containing 0.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    
    # Efficiently compute sumset and difference set
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
            
    n_plus = len(plus)
    n_minus = len(minus)
    
    if n_plus == 0:
        return 0.0
        
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
        
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _objective(x):
    """
    Objective function for differential_evolution.
    x is a vector of gaps [g1, g2, ..., gk] where U = {0, g1, g1+g2, ...}
    We minimize the negative score.
    """
    # Reconstruct U from gaps
    U = [0]
    current = 0
    for g in x:
        g_int = int(round(g))
        if g_int <= 0:
            # If gap is non-positive, skip or handle collision
            # To maintain strictly increasing, we enforce min gap 1
            g_int = 1
        current += g_int
        U.append(current)
    
    # Ensure 0 is in U (it is by construction)
    # Ensure all elements are non-negative (they are)
    
    s = _score(U)
    return -s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s buffer for safety
    
    # Initial best from known good sets
    best_U = [0, 1, 2, 4]
    best_score = _score(best_U)
    
    # Strategy: Use differential evolution to search for better sets.
    # We parameterize the set by its gaps.
    # Since the score depends on the structure of U+U and U-U,
    # and these are determined by the relative positions,
    # we can search over the gaps.
    
    # The max element is sum of gaps.
    # To keep the search space manageable, we limit the max element.
    # The current best has max=4.
    # Let's try to find sets with max up to 50.
    # This means sum of gaps <= 50.
    
    # Differential evolution works on a fixed-dimensional box.
    # We need to decide on the size of U (number of gaps).
    # Let's try sizes k=2 to k=8 (so U has 3 to 9 elements).
    
    # For each size k, we define bounds for the gaps.
    # Min gap is 1.
    # Max gap is limited by the total max_val constraint.
    
    max_val_limit = 60
    
    for k in range(2, 9): # k gaps -> k+1 elements
        if time.time() > deadline - 10:
            break
            
        # Bounds for k gaps
        # Each gap >= 1
        # Sum of gaps <= max_val_limit
        # This is a simplex constraint, which DE doesn't handle directly.
        # Instead, we can just set an upper bound on each gap to max_val_limit
        # and rely on the objective function to penalize large max values
        # implicitly (since log(2mx+1) in denominator grows, but ratio might drop).
        # Actually, if mx is too large, the score drops.
        # So we can set bounds [1, max_val_limit] for each gap.
        
        bounds = [(1, max_val_limit)] * k
        
        # Run differential evolution
        # maxiter and popsize need to be tuned for speed.
        # We have ~140s total. Let's allocate ~10s per k.
        
        try:
            res = differential_evolution(
                _objective, 
                bounds, 
                maxiter=100, 
                popsize=15, 
                seed=42, 
                tol=1e-4,
                init='latinhypercube',
                polish=False # Skip polish to save time, DE is good enough
            )
            
            if res.success or res.fun < -best_score:
                # Reconstruct U
                U = [0]
                current = 0
                for g in res.x:
                    g_int = int(round(g))
                    if g_int < 1:
                        g_int = 1
                    current += g_int
                    U.append(current)
                
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_U = U
                    
        except Exception as e:
            pass # Skip if DE fails
        
        # Check time
        if time.time() > deadline - 5:
            break
            
    # If we have time, try a few more specific structured searches
    # or larger k if time permits
    
    return best_U

if __name__ == "__main__":
    # Quick self-test
    u = construct()
    print(f"Best U: {u}")
    print(f"Score: {_score(u)}")
