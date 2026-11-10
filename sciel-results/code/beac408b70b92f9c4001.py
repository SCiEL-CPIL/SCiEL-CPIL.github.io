import time
import random
import numpy as np

# EVOLVE-BLOCK-START

N = 256
DEADLINE_FRACTION = 0.85

# Precompute all 3-APs
AP3s = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 2 + 1):
        b = a + d
        c = a + 2 * d
        if c <= N:
            AP3s.append((a, b, c))

# Precompute all 4-APs
AP4s = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        if e <= N:
            AP4s.append((a, b, c, e))

# For each x, store the indices of the other 3 elements in all 4-APs containing x
# This allows fast checking if adding x creates a 4-AP
# forbidden_triples[x] is a list of tuples of 3 indices (0-based)
forbidden_triples = [[] for _ in range(N + 1)]
for ap4 in AP4s:
    a, b, c, e = ap4
    # a is in AP4, others are b, c, e
    forbidden_triples[a].append((b - 1, c - 1, e - 1))
    forbidden_triples[b].append((a - 1, c - 1, e - 1))
    forbidden_triples[c].append((a - 1, b - 1, e - 1))
    forbidden_triples[e].append((a - 1, b - 1, c - 1))

def count_3ap(mask):
    """Count 3-APs in the set represented by mask."""
    count = 0
    for a, b, c in AP3s:
        if (mask >> (a - 1) & 1) and (mask >> (b - 1) & 1) and (mask >> (c - 1) & 1):
            count += 1
    return count

def can_add(mask, x):
    """Check if x can be added to set mask without creating a 4-AP."""
    # x is 1-based
    idx = x - 1
    if (mask >> idx) & 1:
        return False # Already in set
    
    # Check all 4-APs containing x
    for t in forbidden_triples[x]:
        i, j, k = t
        if (mask >> i) & 1 and (mask >> j) & 1 and (mask >> k) & 1:
            return False
    return True

def greedy_construct(order):
    """Greedy construction with given order."""
    mask = 0
    for x in order:
        if can_add(mask, x):
            mask |= (1 << (x - 1))
    return mask

def local_search(mask, deadline):
    """Local search to maximize 3-AP count."""
    current_score = count_3ap(mask)
    best_score = current_score
    best_mask = mask
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        
        # Try adding elements
        best_gain = 0
        best_add_x = None
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                continue
            if not can_add(mask, x):
                continue
            
            new_mask = mask | (1 << (x - 1))
            new_score = count_3ap(new_mask)
            gain = new_score - current_score
            if gain > best_gain:
                best_gain = gain
                best_add_x = x
        
        if best_add_x is not None:
            mask |= (1 << (best_add_x - 1))
            current_score += best_gain
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
        
        # Try removing elements
        best_loss = 0
        best_remove_x = None
        for x in range(1, N + 1):
            if not (mask >> (x - 1)) & 1:
                continue
            
            new_mask = mask & ~(1 << (x - 1))
            new_score = count_3ap(new_mask)
            loss = current_score - new_score
            # We want to remove if it allows better additions later, 
            # but simple hill climbing only removes if it doesn't decrease score?
            # Actually, removing always decreases or keeps score (since 3-APs are positive contributions).
            # So simple hill climbing will never remove.
            # We need swaps or random restarts.
            
            # Let's try swaps: remove x, add y
            # This is expensive. 
            # Alternative: Random restart if stuck.
        
        if not improved:
            # Stuck in local optimum. Try a random valid move or restart.
            # Let's try a random swap.
            # Pick random x in mask
            in_set = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            not_in_set = [x for x in range(1, N + 1) if not (mask >> (x - 1)) & 1]
            
            if in_set and not_in_set:
                x = rng.choice(in_set)
                # Try to find a y to add after removing x
                temp_mask = mask & ~(1 << (x - 1))
                best_swap_score = -1
                best_swap_y = None
                for y in not_in_set:
                    if can_add(temp_mask, y):
                        new_mask = temp_mask | (1 << (y - 1))
                        new_score = count_3ap(new_mask)
                        if new_score > best_swap_score:
                            best_swap_score = new_score
                            best_swap_y = y
                
                if best_swap_y is not None and best_swap_score > current_score:
                    mask = temp_mask | (1 << (best_swap_y - 1))
                    current_score = best_swap_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                    # Continue loop to try more improvements
                    continue
        
        # If still stuck, do a random restart with a biased order
        # Or just break and let outer loop restart
        # To save time, we'll just break and restart from scratch or with perturbation
        break
        
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 90 * DEADLINE_FRACTION
    
    best_mask = 0
    best_score = -1
    
    # Strategy 1: Greedy with random orders
    for i in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask = greedy_construct(order)
        score, _ = local_search(mask, deadline)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    # Convert best_mask to list
    if best_mask == 0:
        # Fallback
        return [1]
        
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result

# EVOLVE-BLOCK-END
