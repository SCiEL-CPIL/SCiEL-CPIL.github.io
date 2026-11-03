"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def construct():
    # Generate all integer vectors with squared norm = 4
    # These are the candidates for the kissing configuration
    # Norm 2 vectors. Constraint: pairwise dot product <= 2.
    
    candidates = []
    
    # Type 1: One entry is ±2, rest 0
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
            
    # Type 2: Four entries are ±1, rest 0
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for signs in product((1, -1), repeat=4):
                        v = np.zeros(D, dtype=np.int64)
                        v[i], v[j], v[k], v[l] = signs
                        candidates.append(v)
                        
    # Convert to numpy array for faster operations if needed, 
    # but list of arrays is fine for greedy loop.
    # Number of candidates: 22 + 5280 = 5302
    
    best_selected = []
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    # Strategy: Randomized Greedy with Restart
    # We shuffle the candidates and greedily pick compatible vectors.
    # We repeat this multiple times to find a larger set.
    
    rng = np.random.default_rng(42)
    
    num_restarts = 100 # Adjust based on performance
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        # Shuffle candidates
        indices = np.arange(len(candidates))
        rng.shuffle(indices)
        
        selected = []
        # For efficiency, we can precompute dot products or just check on the fly.
        # On-the-fly is O(k * N) per restart.
        # k ~ 500, N ~ 5300. 2.6M ops. Very fast in Python/NumPy.
        
        for idx in indices:
            v = candidates[idx]
            ok = True
            for w in selected:
                if np.dot(v, w) > 2:
                    ok = False
                    break
            if ok:
                selected.append(v)
                
        if len(selected) > len(best_selected):
            best_selected = selected
            # If we hit the theoretical max or a very high number, we can stop early?
            # K_11 = 512. If we hit 512, we can break.
            if len(best_selected) >= 512:
                break
                
    if not best_selected:
        # Fallback: return any valid set, e.g., the Type 1 vectors
        best_selected = candidates[:22]
        
    return np.array(best_selected, dtype=np.int64)
# EVOLVE-BLOCK-END
