import time
import random
import math

N = 256
DEADLINE = 85.0

# Precompute 4-APs
FAP = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            FAP.append((a, b, c, e))

# Precompute 3-APs
TAP = []
for a in range(1, N + 1):
    for d in range(1, (N - a) // 2 + 1):
        b = a + d
        c = b + d
        if c <= N:
            TAP.append((a, b, c))

# Bitmask helpers
def to_mask(selected):
    m = 0
    for x in selected:
        m |= (1 << (x - 1))
    return m

def from_mask(m):
    return [x + 1 for x in range(N) if (m >> x) & 1]

# Precompute 4-AP masks and element-specific conflicts
# For each element x, store list of masks of the OTHER 3 elements in 4-APs containing x
FAP_CONFLICTS = {x: [] for x in range(1, N + 1)}
for (a, b, c, e) in FAP:
    # For a, others are b, c, e
    FAP_CONFLICTS[a].append((1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
    FAP_CONFLICTS[b].append((1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1)))
    FAP_CONFLICTS[c].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1)))
    FAP_CONFLICTS[e].append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)))

# Precompute 3-AP masks and element-specific contributions
# For each element x, store list of masks of the OTHER 2 elements in 3-APs containing x
TAP_CONTRIBS = {x: [] for x in range(1, N + 1)}
for (a, b, c) in TAP:
    TAP_CONTRIBS[a].append((1 << (b - 1)) | (1 << (c - 1)))
    TAP_CONTRIBS[b].append((1 << (a - 1)) | (1 << (c - 1)))
    TAP_CONTRIBS[c].append((1 << (a - 1)) | (1 << (b - 1)))

def is_valid_add(mask, x):
    """Check if adding x to mask creates a 4-AP."""
    bit = 1 << (x - 1)
    # Check all 4-APs containing x
    for conflict_mask in FAP_CONFLICTS[x]:
        if (mask & conflict_mask) == conflict_mask:
            return False
    return True

def calc_delta_add(mask, x):
    """Count new 3-APs if x is added to mask."""
    count = 0
    for contrib_mask in TAP_CONTRIBS[x]:
        if (mask & contrib_mask) == contrib_mask:
            count += 1
    return count

def calc_delta_remove(mask, x):
    """Count 3-APs lost if x is removed from mask."""
    count = 0
    for contrib_mask in TAP_CONTRIBS[x]:
        if (mask & contrib_mask) == contrib_mask:
            count += 1
    return count

def count_3aps(mask):
    """Count total 3-APs in mask."""
    count = 0
    # Iterate over all 3-APs
    # This is slower, used for initial scoring
    # But for SA, we use deltas.
    # For final verification, we can use this.
    # Actually, let's just sum deltas from scratch? No, just iterate TAP.
    # TAP size is ~ 256*128/2 ~ 16k.
    # 16k bitwise checks is fast.
    for (a, b, c) in TAP:
        m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
        if (mask & m) == m:
            count += 1
    return count

def greedy_construct():
    """Greedy construction maximizing 3-APs while maintaining validity."""
    mask = 0
    score = 0
    # We will try to add elements one by one.
    # To make it better, we can pick the best element at each step.
    # But that is O(N^2 * cost).
    # Let's do a simple pass first, then a local improvement pass.
    
    # Pass 1: Add in order 1..N if valid
    for x in range(1, N + 1):
        if is_valid_add(mask, x):
            mask |= (1 << (x - 1))
            score += calc_delta_add(mask ^ (1 << (x - 1)), x) # Note: delta calc uses mask WITHOUT x?
            # Wait, calc_delta_add(mask, x) expects mask to NOT contain x.
            # My implementation above: `mask` is current set. `x` is new.
            # So `calc_delta_add(mask, x)` is correct.
            # But I need to update score correctly.
            # Let's re-calc score at the end to be safe, or update carefully.
            
    # Let's just return the mask and score it properly.
    # The greedy above is weak. Let's do a "best-first" greedy.
    
    mask = 0
    available = list(range(1, N + 1))
    current_score = 0
    
    while available:
        best_x = -1
        best_delta = -1
        # Find x in available that gives max delta and is valid
        for x in available:
            if is_valid_add(mask, x):
                d = calc_delta_add(mask, x)
                if d > best_delta:
                    best_delta = d
                    best_x = x
        if best_x == -1:
            break
        mask |= (1 << (best_x - 1))
        current_score += best_delta
        available.remove(best_x)
        
    return mask, current_score

def local_search(mask, score, time_limit=10.0):
    """Local search: try swaps and adds/removes to improve score."""
    start = time.time()
    best_mask = mask
    best_score = score
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try adding
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                continue
            if is_valid_add(mask, x):
                d = calc_delta_add(mask, x)
                if d > 0:
                    new_mask = mask | (1 << (x - 1))
                    new_score = score + d
                    if new_score > best_score:
                        best_score = new_score
                        best_mask = new_mask
                    mask = new_mask
                    score = new_score
                    improved = True
                    break # Restart loop to find more adds? Or continue?
                    # Restarting is safer for local search logic (greedy hill climb)
        if improved:
            continue
            
        # Try removing
        for x in range(1, N + 1):
            if not ((mask >> (x - 1)) & 1):
                continue
            d = -calc_delta_remove(mask, x)
            if d > 0:
                new_mask = mask & ~(1 << (x - 1))
                new_score = score + d
                if new_score > best_score:
                    best_score = new_score
                    best_mask = new_mask
                mask = new_mask
                score = new_score
                improved = True
                break
        if improved:
            continue

        # Try swapping
        # This is expensive. O(N^2).
        # Only do if time permits.
        if time.time() - start < time_limit - 1.0:
            for x in range(1, N + 1):
                if (mask >> (x - 1)) & 1:
                    continue
                # x is candidate to add
                if not is_valid_add(mask, x):
                    continue
                delta_add = calc_delta_add(mask, x)
                
                for y in range(1, N + 1):
                    if not ((mask >> (y - 1)) & 1):
                        continue
                    # y is candidate to remove
                    # Check if mask - y + x is valid
                    # We know mask is valid.
                    # Removing y keeps it valid.
                    # Adding x to (mask - y) must be checked.
                    mask_minus_y = mask & ~(1 << (y - 1))
                    if is_valid_add(mask_minus_y, x):
                        delta_remove = -calc_delta_remove(mask, y)
                        # Note: delta_remove is calculated on original mask.
                        # But the 3-APs lost by removing y are those in mask containing y.
                        # The 3-APs gained by adding x to mask_minus_y are those in mask_minus_y containing x.
                        # These are disjoint sets of 3-APs?
                        # A 3-AP cannot be both "lost by removing y" and "gained by adding x" unless it contains both x and y?
                        # If a 3-AP contains x and y, and we remove y and add x... wait.
                        # Original mask has y, not x.
                        # New mask has x, not y.
                        # 3-APs in original: must have y, not x.
                        # 3-APs in new: must have x, not y.
                        # So yes, they are disjoint.
                        net_delta = delta_add + delta_remove
                        if net_delta > 0:
                            new_mask = mask_minus_y | (1 << (x - 1))
                            new_score = score + net_delta
                            if new_score > best_score:
                                best_score = new_score
                                best_mask = new_mask
                            mask = new_mask
                            score = new_score
                            improved = True
                            break
                if improved:
                    break
    return best_mask, best_score

def simulated_annealing(mask, score, time_limit=30.0):
    start = time.time()
    best_mask = mask
    best_score = score
    T = 5.0
    cooling = 0.999
    
    while time.time() - start < time_limit and T > 0.01:
        # Pick move
        move_type = random.randint(0, 2) # 0: add, 1: remove, 2: swap
        
        if move_type == 0: # Add
            # Pick random x not in mask
            unselected = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]
            if not unselected:
                T *= cooling
                continue
            x = random.choice(unselected)
            if not is_valid_add(mask, x):
                T *= cooling
                continue
            delta = calc_delta_add(mask, x)
            if delta > 0 or random.random() < math.exp(delta / T):
                mask |= (1 << (x - 1))
                score += delta
                if score > best_score:
                    best_score = score
                    best_mask = mask
        elif move_type == 1: # Remove
            selected = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            if not selected:
                T *= cooling
                continue
            x = random.choice(selected)
            delta = -calc_delta_remove(mask, x)
            if delta > 0 or random.random() < math.exp(delta / T):
                mask &= ~(1 << (x - 1))
                score += delta
                if score > best_score:
                    best_score = score
                    best_mask = mask
        else: # Swap
            unselected = [x for x in range(1, N + 1) if not ((mask >> (x - 1)) & 1)]
            selected = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
            if not unselected or not selected:
                T *= cooling
                continue
            x = random.choice(unselected)
            y = random.choice(selected)
            
            mask_minus_y = mask & ~(1 << (y - 1))
            if not is_valid_add(mask_minus_y, x):
                T *= cooling
                continue
            
            delta_add = calc_delta_add(mask_minus_y, x)
            delta_remove = -calc_delta_remove(mask, y)
            delta = delta_add + delta_remove
            
            if delta > 0 or random.random() < math.exp(delta / T):
                mask = mask_minus_y | (1 << (x - 1))
                score += delta
                if score > best_score:
                    best_score = score
                    best_mask = mask
        
        T *= cooling
        
    return best_mask, best_score

def construct():
    start_time = time.time()
    
    # 1. Greedy Construction
    mask, score = greedy_construct()
    
    # 2. Local Search
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5.0:
        mask, score = local_search(mask, score, time_limit=min(10.0, remaining - 1.0))
    
    # 3. Simulated Annealing
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5.0:
        mask, score = simulated_annealing(mask, score, time_limit=min(remaining - 1.0, 60.0))
        
    # 4. Final Local Search Polish
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 2.0:
        mask, score = local_search(mask, score, time_limit=min(5.0, remaining - 1.0))
        
    # Convert back to list
    result = from_mask(mask)
    
    # Fallback
    if not result:
        result = [1]
        
    return result

def validate(result):
    sel = set(result)
    for (a, b, c, e) in FAP:
        if a in sel and b in sel and c in sel and e in sel:
            return False
    return True

if __name__ == "__main__":
    result = construct()
    score = count_3aps(to_mask(result))
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {score}")
    print(f"Valid (no 4-AP): {valid}")
