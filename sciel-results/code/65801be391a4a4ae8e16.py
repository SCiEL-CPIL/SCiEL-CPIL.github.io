import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    
    N = 512
    max_a = N - 1
    
    # Pre-compute weights
    weights = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        weights[a] = 1.0 / (N - a)
        
    # Precompute conflict masks
    # conflict_mask[a] has bit j set if gcd(a, j) > 1
    conflict_mask = [0] * (max_a + 1)
    for a in range(1, max_a + 1):
        mask = 0
        for b in range(1, max_a + 1):
            if gcd(a, b) > 1:
                mask |= (1 << b)
        conflict_mask[a] = mask

    # Beam Search
    # We can keep the list-based beam search for now, it's not the bottleneck if local search is fast.
    # But let's optimize the coprime check in beam search too? 
    # The beam width is 150. The list length grows. 
    # Using bitmask for beam search states would be efficient.
    # State: (score, mask)
    
    beam_width = 200 # Slightly wider beam
    beams = [(0.0, 0)] # (score, mask)
    
    # Iterate a from max_a down to 1
    for a in range(max_a, 0, -1):
        w = weights[a]
        new_beams = []
        
        for score, mask in beams:
            # Option 1: Skip a
            new_beams.append((score, mask))
            
            # Option 2: Include a
            # Check coprime: (mask & conflict_mask[a]) == 0
            if (mask & conflict_mask[a]) == 0:
                new_beams.append((score + w, mask | (1 << a)))
                
        # Prune
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]
        
        # Time check
        if time.time() - start_time > deadline - safety_margin - 30:
            break
            
    # Best beam
    best_beam = max(beams, key=lambda x: x[0])
    best_score = best_beam[0]
    best_mask = best_beam[1]
    
    # Convert mask to list for return, but keep mask for local search
    # We need to return a list of distinct integers.
    
    # Local Search with Bitmasks
    improved = True
    while improved:
        improved = False
        
        # Get current elements from mask
        current_elements = [a for a in range(1, max_a + 1) if (best_mask >> a) & 1]
        
        # 1-Opt: Replace one element with a better one
        # We iterate through current elements. If we find a swap, we do it and restart.
        for a in current_elements:
            others_mask = best_mask ^ (1 << a)
            # Find best b > a (higher weight) not in mask, coprime to others
            for b in range(max_a, a, -1):
                if (best_mask >> b) & 1:
                    continue
                if (others_mask & conflict_mask[b]) == 0:
                    # Swap a with b
                    best_mask = (best_mask ^ (1 << a)) | (1 << b)
                    best_score += weights[b] - weights[a]
                    improved = True
                    break
            if improved:
                break
        if improved:
            if time.time() - start_time > deadline - safety_margin:
                break
            continue

        # Drop-and-Refill
        for a in current_elements:
            temp_mask = best_mask ^ (1 << a)
            temp_score = best_score - weights[a]
            
            # Greedily fill temp_mask with best available coprime elements
            # Iterate b from max_a down to 1
            for b in range(max_a, 0, -1):
                if (temp_mask >> b) & 1:
                    continue
                if (temp_mask & conflict_mask[b]) == 0:
                    temp_mask |= (1 << b)
                    temp_score += weights[b]
                    
            if temp_score > best_score + 1e-12:
                best_mask = temp_mask
                best_score = temp_score
                improved = True
                break
            if time.time() - start_time > deadline - safety_margin:
                break
        if improved:
            continue
        if time.time() - start_time > deadline - safety_margin:
            break

    # Convert best_mask back to list
    best_chosen = [a for a in range(1, max_a + 1) if (best_mask >> a) & 1]
    
    return best_chosen
