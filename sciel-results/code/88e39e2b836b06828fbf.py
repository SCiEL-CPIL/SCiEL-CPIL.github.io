"""4-AP-free subset of [1,256] maximizing 3-APs via simulated annealing with bitmask."""
import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # safety margin before 90s kill

# Precompute 4-AP conflict masks
# For each element x, conflict_mask[x] is a bitmask of elements that cannot coexist with x
# because they'd form a 4-AP with some other element already in the set.
# Actually, we need: given current mask, can we add x? x conflicts if there exists a 4-AP
# containing x where the other 3 elements are all in the current mask.

# Precompute for each x, the list of (other1, other2, other3) triples that form a 4-AP with x
# Then has_4ap(mask, x) checks if any of those triples is fully contained in mask.

def precompute_conflicts():
    """For each x in 1..N, precompute the 4-APs containing x.
    Returns a list where conflicts[x-1] is a list of 3-element masks (bitmasks of the other 3 elements).
    has_4ap(mask, x) = any((mask & m) == m for m in conflicts[x-1])
    """
    conflicts = [[] for _ in range(N)]
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > N:
                break
            # 4-AP: a, b, c, e
            # For x=a: other 3 are b, c, e
            m = (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
            conflicts[a-1].append(m)
            # For x=b: other 3 are a, c, e
            m = (1 << (a-1)) | (1 << (c-1)) | (1 << (e-1))
            conflicts[b-1].append(m)
            # For x=c: other 3 are a, b, e
            m = (1 << (a-1)) | (1 << (b-1)) | (1 << (e-1))
            conflicts[c-1].append(m)
            # For x=e: other 3 are a, b, c
            m = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            conflicts[e-1].append(m)
    return conflicts

def precompute_3ap_neighbors():
    """For each x, precompute which pairs (y,z) with x in the middle or end form a 3-AP.
    Actually, let's precompute for each x:
      - as_first[x]: list of (d, b, c) such that (x, x+d, x+2d) is a 3-AP
      - as_mid[x]: list of (d, a, c) such that (x-d, x, x+d) is a 3-AP
      - as_last[x]: list of (d, a, b) such that (x-2d, x-d, x) is a 3-AP
    """
    as_first = [[] for _ in range(N)]
    as_mid = [[] for _ in range(N)]
    as_last = [[] for _ in range(N)]
    
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 2 + 1):
            c = a + 2 * d
            if c > N:
                break
            b = a + d
            as_first[a-1].append((b-1, c-1))  # (b-1, c-1) are 0-indexed
            as_mid[b-1].append((a-1, c-1))
            as_last[c-1].append((a-1, b-1))
    
    return as_first, as_mid, as_last

def count_3aps_fast(mask, as_first, as_mid, as_last):
    """Count 3-APs using precomputed neighbor lists."""
    count = 0
    for x in range(N):
        if not (mask >> x) & 1:
            continue
        for (y, z) in as_first[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                count += 1
        for (y, z) in as_mid[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                count += 1
        for (y, z) in as_last[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                count += 1
    # Each 3-AP is counted 3 times (once per element)
    return count // 3

def delta_3aps_add(mask, x_idx, as_first, as_mid, as_last):
    """Count new 3-APs when adding element at 0-indexed x_idx."""
    gain = 0
    for (y, z) in as_first[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            gain += 1
    for (y, z) in as_mid[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            gain += 1
    for (y, z) in as_last[x_idx]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            gain += 1
    return gain

def can_add(mask, x_idx, conflicts):
    """Check if element x_idx can be added without creating a 4-AP."""
    for m in conflicts[x_idx]:
        if (mask & m) == m:
            return False
    return True

def degree_greedy_init(conflicts, as_first, as_mid, as_last):
    """Initialize using degree-based greedy: add elements that create most 3-APs first."""
    mask = 0
    remaining = set(range(N))
    
    while remaining:
        best_x = -1
        best_gain = -1
        for x in remaining:
            if not can_add(mask, x, conflicts):
                continue
            gain = delta_3aps_add(mask, x, as_first, as_mid, as_last)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x == -1:
            break
        mask |= (1 << best_x)
        remaining.discard(best_x)
    
    return mask

def simulated_annealing(mask, conflicts, as_first, as_mid, as_last, start_time, deadline):
    """Simulated annealing to improve the mask."""
    current_mask = mask
    current_score = count_3aps_fast(mask, as_first, as_mid, as_last)
    best_mask = mask
    best_score = current_score
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    rng = random.Random(123)
    
    # Precompute list of all elements for random selection
    all_elements = list(range(N))
    
    while time.time() < deadline and T > T_min:
        # Pick a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Add a random element
            x = rng.choice(all_elements)
            if (current_mask >> x) & 1:
                continue
            if not can_add(current_mask, x, conflicts):
                continue
            gain = delta_3aps_add(current_mask, x, as_first, as_mid, as_last)
            new_mask = current_mask | (1 << x)
            delta = gain
        elif move_type < 0.7:
            # Remove a random element
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if not bits:
                continue
            x = rng.choice(bits)
            new_mask = current_mask & ~(1 << x)
            # Loss = number of 3-APs that x participates in
            loss = 0
            for (y, z) in as_first[x]:
                if (current_mask >> y) & 1 and (current_mask >> z) & 1:
                    loss += 1
            for (y, z) in as_mid[x]:
                if (current_mask >> y) & 1 and (current_mask >> z) & 1:
                    loss += 1
            for (y, z) in as_last[x]:
                if (current_mask >> y) & 1 and (current_mask >> z) & 1:
                    loss += 1
            delta = -loss
        else:
            # Swap: remove one, add another
            bits = [i for i in range(N) if (current_mask >> i) & 1]
            if not bits:
                continue
            x = rng.choice(bits)  # to remove
            # Try to find a y to add
            added = False
            for _ in range(20):
                y = rng.choice(all_elements)
                if (current_mask >> y) & 1:
                    continue
                mask_without_x = current_mask & ~(1 << x)
                if can_add(mask_without_x, y, conflicts):
                    # Compute delta
                    # Remove x's contributions
                    loss = 0
                    for (a2, b2) in as_first[x]:
                        if (current_mask >> a2) & 1 and (current_mask >> b2) & 1:
                            loss += 1
                    for (a2, b2) in as_mid[x]:
                        if (current_mask >> a2) & 1 and (current_mask >> b2) & 1:
                            loss += 1
                    for (a2, b2) in as_last[x]:
                        if (current_mask >> a2) & 1 and (current_mask >> b2) & 1:
                            loss += 1
                    # Add y's contributions (in mask without x)
                    gain = 0
                    for (a2, b2) in as_first[y]:
                        if (mask_without_x >> a2) & 1 and (mask_without_x >> b2) & 1:
                            gain += 1
                    for (a2, b2) in as_mid[y]:
                        if (mask_without_x >> a2) & 1 and (mask_without_x >> b2) & 1:
                            gain += 1
                    for (a2, b2) in as_last[y]:
                        if (mask_without_x >> a2) & 1 and (mask_without_x >> b2) & 1:
                            gain += 1
                    new_mask = mask_without_x | (1 << y)
                    delta = gain - loss
                    added = True
                    break
            if not added:
                continue
        
        new_score = current_score + delta
        if new_score > current_score:
            current_mask = new_mask
            current_score = new_score
            if new_score > best_score:
                best_mask = new_mask
                best_score = new_score
        else:
            # Accept with probability exp(delta/T)
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_mask = new_mask
                    current_score = new_score
        
        T *= cooling
    
    return best_mask, best_score

def local_search(mask, conflicts, as_first, as_mid, as_last, max_iter=2000):
    """Hill climbing local search."""
    current_mask = mask
    current_score = count_3aps_fast(mask, as_first, as_mid, as_last)
    best_mask = mask
    best_score = current_score
    
    for _ in range(max_iter):
        improved = False
        
        # Try adding each element
        for x in range(N):
            if (current_mask >> x) & 1:
                continue
            if not can_add(current_mask, x, conflicts):
                continue
            gain = delta_3aps_add(current_mask, x, as_first, as_mid, as_last)
            if gain > 0:
                new_mask = current_mask | (1 << x)
                new_score = current_score + gain
                if new_score > best_score:
                    best_mask = new_mask
                    best_score = new_score
                    current_mask = new_mask
                    current_score = new_score
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swaps
        for x in range(N):
            if not (current_mask >> x) & 1:
                continue
            mask_without = current_mask & ~(1 << x)
            for y in range(N):
                if (current_mask >> y) & 1:
                    continue
                if not can_add(mask_without, y, conflicts):
                    continue
                # Compute delta
                loss = 0
                for (a2, b2) in as_first[x]:
                    if (current_mask >> a2) & 1 and (current_mask >> b2) & 1:
                        loss += 1
                for (a2, b2) in as_mid[x]:
                    if (current_mask >> a2) & 1 and (current_mask >> b2) & 1:
                        loss += 1
                for (a2, b2) in as_last[x]:
                    if (current_mask >> a2) & 1 and (current_mask >> b2) & 1:
                        loss += 1
                gain = 0
                for (a2, b2) in as_first[y]:
                    if (mask_without >> a2) & 1 and (mask_without >> b2) & 1:
                        gain += 1
                for (a2, b2) in as_mid[y]:
                    if (mask_without >> a2) & 1 and (mask_without >> b2) & 1:
                        gain += 1
                for (a2, b2) in as_last[y]:
                    if (mask_without >> a2) & 1 and (mask_without >> b2) & 1:
                        gain += 1
                delta = gain - loss
                if delta > 0:
                    new_mask = mask_without | (1 << y)
                    new_score = current_score + delta
                    if new_score > best_score:
                        best_mask = new_mask
                        best_score = new_score
                        current_mask = new_mask
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute
    conflicts = precompute_conflicts()
    as_first, as_mid, as_last = precompute_3ap_neighbors()
    
    # Initialize with degree greedy
    mask = degree_greedy_init(conflicts, as_first, as_mid, as_last)
    score = count_3aps_fast(mask, as_first, as_mid, as_last)
    
    best_mask = mask
    best_score = score
    
    # Local search
    if time.time() < deadline - 5:
        mask, score = local_search(mask, conflicts, as_first, as_mid, as_last, max_iter=3000)
        if score > best_score:
            best_mask, best_score = mask, score
    
    # Simulated annealing
    if time.time() < deadline - 10:
        mask, score = simulated_annealing(best_mask, conflicts, as_first, as_mid, as_last, start_time, deadline - 5)
        if score > best_score:
            best_mask, best_score = mask, score
    
    # Final local search on best
    if time.time() < deadline - 2:
        mask, score = local_search(best_mask, conflicts, as_first, as_mid, as_last, max_iter=2000)
        if score > best_score:
            best_mask, best_score = mask, score
    
    # Convert to list
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return sorted(result)
