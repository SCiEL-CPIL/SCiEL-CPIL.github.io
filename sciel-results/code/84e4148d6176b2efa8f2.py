import itertools
import numpy as np
import time

def construct(q, n, radius):
    # Fallback for non-standard parameters to ensure validity
    # The task specifies 3, 7, 1. 
    # If parameters differ, we use a simple greedy or the seed logic if applicable.
    # For this specific task, we optimize for 3, 7, 1.
    
    if q != 3 or n != 7 or radius != 1:
        # Simple greedy for other cases, or seed logic
        # Seed logic: Direct sum of tetracode
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Fallback: return all words (valid but high score)
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    words = list(itertools.product(range(3), repeat=7))
    word_to_idx = {w: i for i, w in enumerate(words)}
    N = len(words)
    
    # Pre-compute neighbors (distance <= 1)
    neighbors = []
    for w in words:
        neigh = [word_to_idx[w]]
        for i in range(7):
            for delta in [1, 2]:
                new_w = list(w)
                new_w[i] = (new_w[i] + delta) % 3
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors.append(neigh)
    neighbors = np.array(neighbors)
    
    best_sol = None
    best_len = float('inf')
    
    start_time = time.time()
    time_limit = 110.0
    
    # Randomized Greedy Set Cover
    while time.time() - start_time < time_limit:
        uncovered = np.ones(N, dtype=bool)
        sol = []
        
        while uncovered.any():
            # Calculate coverage gain for each codeword
            # gains[i] = number of uncovered points covered by codeword i
            gains = np.sum(uncovered[neighbors], axis=1)
            max_gain = gains.max()
            
            if max_gain == 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking to explore different solutions
            c = candidates[np.random.randint(len(candidates))]
            
            sol.append(c)
            # Mark covered points
            uncovered[neighbors[c]] = False
            
        if len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol[:]
            
    if best_sol is None:
        # Should not happen
        return [list(w) for w in words[:243]]
        
    return [list(words[i]) for i in best_sol]
