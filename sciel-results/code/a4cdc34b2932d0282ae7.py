"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _precompute_conflicts(N=180):
    """Precompute pair masks for each element y.
    pair_masks[y] is a list of integers, each representing a pair {a, b} such that {y, a, b} is a forbidden triple.
    """
    pair_masks = [[] for _ in range(N + 1)]
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a < b and a != c:
                    # Triple is (a, b, c)
                    # For a: pair is (b, c)
                    # For b: pair is (a, c)
                    # For c: pair is (a, b)
                    
                    mask_bc = (1 << b) | (1 << c)
                    mask_ac = (1 << a) | (1 << c)
                    mask_ab = (1 << a) | (1 << b)
                    
                    pair_masks[a].append(mask_bc)
                    pair_masks[b].append(mask_ac)
                    pair_masks[c].append(mask_ab)
    
    return pair_masks

def _can_add(y, S_mask, pair_masks):
    """Check if y can be added to S_mask without creating a forbidden triple."""
    for pm in pair_masks[y]:
        if (pm & S_mask) == pm:
            return False
    return True

def _greedy_fill(S_mask, pair_masks, order=None, deadline=None):
    """Greedily add elements to S_mask if possible.
    If order is provided, use that order. Otherwise, use 1..180.
    """
    if order is None:
        order = range(1, 181)
        
    for y in order:
        if deadline and time.time() > deadline:
            break
        if S_mask & (1 << y):
            continue
        if _can_add(y, S_mask, pair_masks):
            S_mask |= (1 << y)
            
    return S_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 180
    pair_masks = _precompute_conflicts(N)
    
    best_size = 0
    best_mask = 0
    
    # 1. Try multiple random greedy orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1))
    ]
    
    # Add some random permutations
    for _ in range(10):
        rand_order = list(range(1, N + 1))
        random.shuffle(rand_order)
        orders.append(rand_order)
        
    for order in orders:
        if time.time() > deadline - 5:
            break
        trial_mask = 0
        trial_mask = _greedy_fill(trial_mask, pair_masks, order=order, deadline=deadline)
        size = bin(trial_mask).count("1")
        if size > best_size:
            best_size = size
            best_mask = trial_mask
            
    # 2. Local Search on the best greedy result
    current_mask = best_mask
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try single removals
        for x in range(1, N + 1):
            if not (current_mask & (1 << x)):
                continue
            
            trial_mask = current_mask & ~(1 << x)
            # Refill using a random order to diversify
            refill_order = list(range(1, N + 1))
            random.shuffle(refill_order)
            trial_mask = _greedy_fill(trial_mask, pair_masks, order=refill_order, deadline=deadline)
            
            if bin(trial_mask).count("1") > best_size:
                best_size = bin(trial_mask).count("1")
                best_mask = trial_mask
                current_mask = trial_mask
                improved = True
                break # Restart search with new best
        
        if time.time() > deadline - 2:
            break
            
        # Try pair removals (if time permits)
        if not improved:
            # Sample random pairs to escape local minima
            for _ in range(50): # Limit attempts per iteration
                if time.time() > deadline - 2:
                    break
                # Get list of set bits
                bits = [i for i in range(1, N+1) if current_mask & (1 << i)]
                if len(bits) < 2:
                    break
                i, j = random.sample(bits, 2)
                trial_mask = current_mask & ~(1 << i) & ~(1 << j)
                refill_order = list(range(1, N + 1))
                random.shuffle(refill_order)
                trial_mask = _greedy_fill(trial_mask, pair_masks, order=refill_order, deadline=deadline)
                
                if bin(trial_mask).count("1") > best_size:
                    best_size = bin(trial_mask).count("1")
                    best_mask = trial_mask
                    current_mask = trial_mask
                    improved = True
                    break
                     
    # Convert best_mask to list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    
    # Ensure non-empty
    if not result:
        return [180]
        
    return sorted(result)
# EVOLVE-BLOCK-END
