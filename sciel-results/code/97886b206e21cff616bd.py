import random
import time

def _count_comparable(masks):
    """Count pairs where one is strict subset of other."""
    n = len(masks)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = masks[i], masks[j]
            if a & b == a and a != b:
                count += 1
            elif a & b == b and a != b:
                count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    n = 12
    m = 40
    all_masks = list(range(1 << n))
    
    # Initial solution: greedy chain-like
    # Start with a chain, then add sets that maximize comparable pairs
    random.seed(42)
    
    # Try multiple random restarts with simulated annealing
    best_score = -1
    best_masks = None
    
    for restart in range(50):
        # Initial: random 40 masks, or structured
        if restart == 0:
            # Structured: take a chain of 13, then fill with sets close to chain
            # Chain: {}, {0}, {0,1}, {0,1,2}, ..., {0..11}
            chain = [(1 << (i+1)) - 1 for i in range(n+1)]  # 13 sets
            # Add more sets that are subsets of larger chain members
            masks = list(chain)
            remaining = [m for m in all_masks if m not in set(chain)]
            random.shuffle(remaining)
            for m in remaining:
                if len(masks) >= m:
                    break
                masks.append(m)
            masks = masks[:m]
        else:
            # Random initial
            masks = random.sample(all_masks, m)
        
        score = _count_comparable(masks)
        
        # Simulated annealing
        T = 100.0
        cooling = 0.9995
        min_T = 0.01
        steps = 0
        
        while T > min_T and time.time() < deadline:
            steps += 1
            # Pick a random index to modify
            idx = random.randint(0, m-1)
            old_mask = masks[idx]
            
            # Try replacing with a random mask, or a mask near old_mask
            if random.random() < 0.5:
                new_mask = random.choice(all_masks)
            else:
                # Flip 1-3 bits
                new_mask = old_mask
                for _ in range(random.randint(1, 3)):
                    bit = random.randint(0, n-1)
                    new_mask ^= (1 << bit)
            
            if new_mask == old_mask:
                continue
            if new_mask in masks:
                continue
            
            # Compute delta
            # Remove old_mask's contributions, add new_mask's
            delta = 0
            for j in range(m):
                if j == idx:
                    continue
                other = masks[j]
                # Old contribution
                if old_mask & other == old_mask and old_mask != other:
                    delta -= 1
                elif old_mask & other == other and old_mask != other:
                    delta -= 1
                # New contribution
                if new_mask & other == new_mask and new_mask != other:
                    delta += 1
                elif new_mask & other == other and new_mask != other:
                    delta += 1
            
            if delta >= 0 or random.random() < min(1.0, delta / T):
                masks[idx] = new_mask
                score += delta
                if score > best_score:
                    best_score = score
                    best_masks = list(masks)
            
            T *= cooling
    
        if time.time() > deadline:
            break
    
    # Convert best_masks to list of lists
    if best_masks is None:
        # Fallback
        best_masks = list(range(m))
    
    result = []
    for mask in best_masks:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
