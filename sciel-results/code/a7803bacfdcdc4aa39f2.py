import random
import time
import math

def construct():
    n = 12
    num_sets = 40
    max_mask = 1 << n
    
    # Precompute list of all masks
    all_masks = list(range(max_mask))
    
    # Precompute popcounts and maybe subset relations? 
    # For speed, we check comparability on the fly using bit operations.
    # (a & b) == a or (a & b) == b
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    def count_comparable_pairs(chosen):
        count = 0
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                if is_comparable(chosen[i], chosen[j]):
                    count += 1
        return count

    def get_gain(mask, chosen):
        gain = 0
        for c in chosen:
            if is_comparable(mask, c):
                gain += 1
        return gain

    # 1. Initial Solution: 
    # We want to start with a high-quality solution.
    # A good heuristic is to pick sets that are "dense" in the poset.
    # Let's try a multi-start greedy approach to initialize.
    
    def greedy_init():
        # Start with a random chain
        order = list(range(n))
        random.shuffle(order)
        current = 0
        chain = [0]
        for i in order:
            current |= (1 << i)
            chain.append(current)
        
        chosen = set(chain)
        
        # Greedily add remaining sets
        # To speed up, we can pre-sort masks by "potential" or just random scan.
        # We'll scan all masks a few times or use a random permutation.
        remaining_masks = [m for m in all_masks if m not in chosen]
        random.shuffle(remaining_masks)
        
        # We need 40 - len(chain) more sets.
        # Greedy: pick the one with max gain.
        while len(chosen) < num_sets:
            best_mask = -1
            best_gain = -1
            # Optimization: Instead of scanning ALL remaining masks every time (4096 * 40 * 40 ops),
            # we can just scan a random subset or the whole list. 4096 is small enough.
            for m in remaining_masks:
                if m in chosen:
                    continue
                gain = get_gain(m, list(chosen))
                if gain > best_gain:
                    best_gain = gain
                    best_mask = m
            if best_mask != -1:
                chosen.add(best_mask)
                # Remove from remaining to speed up next iterations? 
                # List removal is O(N). Set lookup is O(1). 
                # We'll just keep checking `if m in chosen`.
        return list(chosen)

    best_chosen = greedy_init()
    best_score = count_comparable_pairs(best_chosen)
    
    # 2. Simulated Annealing
    # Time limit: 50 seconds
    start_time = time.time()
    limit = 50.0
    
    # Cooling schedule
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    
    while time.time() - start_time < limit and T > T_min:
        # Pick a random index to remove
        idx = random.randint(0, len(best_chosen) - 1)
        removed_mask = best_chosen[idx]
        
        # Remove it
        temp_chosen = [best_chosen[i] for i in range(len(best_chosen)) if i != idx]
        temp_set = set(temp_chosen)
        
        # Find best replacement
        # To make this faster and more "clique-like" (greedy with backtracking feel),
        # we pick the mask with the highest gain. If we want randomness, we can pick
        # from top K or use a random mask.
        # Let's pick the single best gain mask to ensure strong moves, 
        # but the SA acceptance criteria allows us to escape if this specific 
        # "best swap" leads to a local minimum? No, if we always pick best swap,
        # it's deterministic hill climbing.
        
        # To make it SA, we should pick a *random* replacement or a *random* 
        # replacement from a candidate pool.
        # Let's pick a random mask from all_masks not in temp_set.
        # This allows exploring the space.
        
        # Optimization: Pre-compute a list of all masks not in temp_set?
        # 4096 - 39 = 4057 options. Picking one is fast.
        
        # Strategy: Pick a random candidate, but bias towards high gain?
        # Or just pick random?
        # Let's pick a random mask.
        
        new_mask = random.choice(all_masks)
        while new_mask in temp_set:
            new_mask = random.choice(all_masks)
            
        # Calculate score change
        removed_gain = get_gain(removed_mask, temp_chosen)
        new_gain = get_gain(new_mask, temp_chosen)
        
        delta_score = new_gain - removed_gain
        
        if delta_score >= 0:
            # Accept
            best_chosen[idx] = new_mask
            best_score += delta_score
        else:
            # Reject or accept with probability exp(delta / T)
            # delta_score is negative, so exp is < 1
            probability = math.exp(delta_score / T)
            if random.random() < probability:
                best_chosen[idx] = new_mask
                best_score += delta_score
        
        T *= cooling_rate

    # Final local search (Hill Climbing) to polish
    # We can run a few steps of strict hill climbing to ensure we are at a local max
    improved = True
    while improved:
        improved = False
        for idx in range(len(best_chosen)):
            removed_mask = best_chosen[idx]
            temp_chosen = [best_chosen[i] for i in range(len(best_chosen)) if i != idx]
            temp_set = set(temp_chosen)
            
            best_new_mask = -1
            best_new_gain = -1
            
            # Scan all masks for best replacement
            for m in all_masks:
                if m in temp_set:
                    continue
                gain = get_gain(m, temp_chosen)
                if gain > best_new_gain:
                    best_new_gain = gain
                    best_new_mask = m
            
            removed_gain = get_gain(removed_mask, temp_chosen)
            
            if best_new_gain > removed_gain:
                best_chosen[idx] = best_new_mask
                best_score += (best_new_gain - removed_gain)
                improved = True
                # Restart loop as set changed
                break

    # Convert to list of lists
    result = []
    for mask in best_chosen:
        subset = [i for i in range(n) if (mask >> i) & 1]
        result.append(subset)
        
    return result
