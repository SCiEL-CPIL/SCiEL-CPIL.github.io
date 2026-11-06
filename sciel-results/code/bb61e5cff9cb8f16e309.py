import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    def run_greedy(noise_scale=0.0):
        chosen = []
        counts = defaultdict(int)
        candidates = list(range(1, N + 1))
        
        while candidates:
            best_x = None
            best_score = float('inf')
            
            # Optional: Early exit if time is tight
            if time.time() > deadline:
                break
                
            for x in candidates:
                # Check validity and compute tight count in one pass
                valid = True
                tight = 0
                if counts.get(2 * x, 0) >= 2:
                    continue
                if counts.get(2 * x, 0) == 1:
                    tight += 1
                for y in chosen:
                    s = x + y
                    c = counts.get(s, 0)
                    if c >= 2:
                        valid = False
                        break
                    if c == 1:
                        tight += 1
                if not valid:
                    continue
                score = tight + (random.random() * noise_scale if noise_scale > 0 else 0)
                if score < best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
                
            # Add to set
            counts[2 * best_x] = counts.get(2 * best_x, 0) + 1
            for y in chosen:
                s = best_x + y
                counts[s] = counts.get(s, 0) + 1
            chosen.append(best_x)
            candidates.remove(best_x)
        
        return chosen

    def improve_set(chosen):
        """Local search: try to swap elements to improve the set."""
        chosen = list(chosen)
        counts = defaultdict(int)
        for x in chosen:
            counts[2 * x] += 1
            for y in chosen:
                if y > x:
                    counts[x + y] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i, x in enumerate(chosen):
                # Try removing x and adding a different element
                # First, remove x
                for y in chosen:
                    if y == x:
                        counts[2 * x] -= 1
                        if counts[2 * x] == 0:
                            del counts[2 * x]
                    else:
                        s = x + y
                        counts[s] -= 1
                        if counts[s] == 0:
                            del counts[s]
                
                chosen_without_x = [z for z in chosen if z != x]
                
                # Try to find a better replacement
                best_y = None
                best_tight = float('inf')
                for y in range(1, N + 1):
                    if y in chosen_without_x:
                        continue
                    # Check if y is valid
                    valid = True
                    tight = 0
                    if counts.get(2 * y, 0) >= 2:
                        continue
                    if counts.get(2 * y, 0) == 1:
                        tight += 1
                    for z in chosen_without_x:
                        s = y + z
                        c = counts.get(s, 0)
                        if c >= 2:
                            valid = False
                            break
                        if c == 1:
                            tight += 1
                    if not valid:
                        continue
                    if tight < best_tight:
                        best_tight = tight
                        best_y = y
                
                if best_y is not None:
                    # Add best_y
                    counts[2 * best_y] = counts.get(2 * best_y, 0) + 1
                    for z in chosen_without_x:
                        s = best_y + z
                        counts[s] = counts.get(s, 0) + 1
                    chosen[i] = best_y
                    improved = True
                else:
                    # Put x back
                    for y in chosen_without_x:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    chosen[i] = x
        
        return chosen

    # Phase 1: Deterministic Min-Cost Greedy (Noise = 0)
    # This gives a strong baseline
    initial_set = run_greedy(noise_scale=0.0)
    best_set = initial_set
    best_size = len(initial_set)
    
    # Phase 2: Randomized Noisy Min-Cost Greedy Restarts
    # We use noise to explore the space of "good" greedy choices
    while time.time() < deadline:
        # Increase noise over time? Or keep it fixed?
        # Fixed small noise is good for diversity.
        current_set = run_greedy(noise_scale=0.5)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return best_set
