import time
import random
import sys

N = 256
TOTAL_BITS = N + 1  # bits 1..256

def _precompute():
    """Precompute 4-AP constraints and 3-AP structure."""
    # All 4-APs: (a, a+d, a+2d, a+3d)
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                all_4aps.append((a, b, c, e))
    
    # For each element x, list of 4-APs containing x (as index into all_4aps)
    # and for each such 4-AP, the other 3 elements
    all_4aps_by_element = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(all_4aps):
        all_4aps_by_element[a].append((b, c, e))
        all_4aps_by_element[b].append((a, c, e))
        all_4aps_by_element[c].append((a, b, e))
        all_4aps_by_element[e].append((a, b, c))
    
    # Precompute for 3-AP counting: for each middle b, list of valid (a, c) pairs
    # where a < b < c and a + c = 2b
    three_ap_pairs = [[] for _ in range(N + 1)]
    for b in range(1, N + 1):
        for a in range(1, b):
            c = 2 * b - a
            if c > b and c <= N:
                three_ap_pairs[b].append((a, c))
    
    # For fast delta computation: for each x, precompute which 3-APs it participates in
    # x as first: (x, b, c) where c = 2b - x, b > x, c > b
    # x as middle: (a, x, c) where a < x < c, a + c = 2x
    # x as last: (a, b, x) where a + x = 2b, a < b < x
    
    # For delta_add(mask, x): count new 3-APs
    # x as middle: for each (a, c) in three_ap_pairs[x], if a and c in mask
    # x as first: for each b > x, c = 2b - x, if b and c in mask and c <= N and c > b
    # x as last: for each a < x, b = (a+x)//2, if b in mask and (a+x) even and a < b < x
    
    # Precompute x_as_first[x] = list of (b, c)
    x_as_first = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c > b and c <= N:
                x_as_first[x].append((b, c))
    
    # x_as_last[x] = list of (a, b) where a + x = 2b, a < b < x
    x_as_last = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, x):
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if a < b < x:
                    x_as_last[x].append((a, b))
    
    return all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last

def _creates_4ap(mask, x, all_4aps_by_element):
    """Check if adding x to mask creates a 4-AP."""
    for triple in all_4aps_by_element[x]:
        if (mask & (1 << triple[0])) and (mask & (1 << triple[1])) and (mask & (1 << triple[2])):
            return True
    return False

def _delta_add(mask, x, three_ap_pairs, x_as_first, x_as_last):
    """Count new 3-APs created by adding x to the set represented by mask."""
    count = 0
    # x as middle
    for a, c in three_ap_pairs[x]:
        if (mask & (1 << a)) and (mask & (1 << c)):
            count += 1
    # x as first
    for b, c in x_as_first[x]:
        if (mask & (1 << b)) and (mask & (1 << c)):
            count += 1
    # x as last
    for a, b in x_as_last[x]:
        if (mask & (1 << a)) and (mask & (1 << b)):
            count += 1
    return count

def _count_3aps(mask, three_ap_pairs):
    """Count total 3-APs in set represented by mask."""
    count = 0
    for b in range(1, N + 1):
        if not (mask & (1 << b)):
            continue
        for a, c in three_ap_pairs[b]:
            if (mask & (1 << a)) and (mask & (1 << c)):
                count += 1
    return count

def _greedy_construct(all_4aps_by_element, order=None):
    """Greedy construction: add elements in given order if no 4-AP created."""
    if order is None:
        order = list(range(1, N + 1))
    mask = 0
    for x in order:
        if not _creates_4ap(mask, x, all_4aps_by_element):
            mask |= (1 << x)
    return mask

def _local_search(mask, all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline):
    """Hill climbing local search."""
    current_mask = mask
    current_score = _count_3aps(current_mask, three_ap_pairs)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        set_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
        unset_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
        
        # Try all additions
        for y in unset_elements[:]:
            if _creates_4ap(current_mask, y, all_4aps_by_element):
                continue
            delta = _delta_add(current_mask, y, three_ap_pairs, x_as_first, x_as_last)
            if delta > 0:
                current_mask |= (1 << y)
                current_score += delta
                improved = True
                unset_elements.remove(y)
                set_elements.append(y)
        
        # Try all swaps
        if time.time() < deadline:
            for x in set_elements[:]:
                if not improved and time.time() > deadline:
                    break
                temp_mask = current_mask & ~(1 << x)
                for y in unset_elements:
                    if _creates_4ap(temp_mask, y, all_4aps_by_element):
                        continue
                    delta = _delta_add(temp_mask, y, three_ap_pairs, x_as_first, x_as_last) - _delta_add(current_mask, x, three_ap_pairs, x_as_first, x_as_last)
                    if delta > 0:
                        current_mask = temp_mask | (1 << y)
                        current_score += delta
                        improved = True
                        set_elements.remove(x)
                        set_elements.append(y)
                        unset_elements.remove(y)
                        unset_elements.append(x)
                        break
    
    return current_mask, current_score

def _simulated_annealing(mask, all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline, seed=42):
    """Simulated annealing."""
    random.seed(seed)
    current_mask = mask
    current_score = _count_3aps(current_mask, three_ap_pairs)
    best_mask = current_mask
    best_score = current_score
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap
            set_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
            unset_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
            if not set_elements or not unset_elements:
                T *= cooling
                continue
            x = random.choice(set_elements)
            y = random.choice(unset_elements)
            temp_mask = current_mask & ~(1 << x)
            if _creates_4ap(temp_mask, y, all_4aps_by_element):
                T *= cooling
                continue
            new_mask = temp_mask | (1 << y)
            delta = (_delta_add(temp_mask, y, three_ap_pairs, x_as_first, x_as_last) - 
                     _delta_add(current_mask, x, three_ap_pairs, x_as_first, x_as_last))
            
        elif move_type < 0.7:
            # Add
            unset_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
            if not unset_elements:
                T *= cooling
                continue
            y = random.choice(unset_elements)
            if _creates_4ap(current_mask, y, all_4aps_by_element):
                T *= cooling
                continue
            new_mask = current_mask | (1 << y)
            delta = _delta_add(current_mask, y, three_ap_pairs, x_as_first, x_as_last)
        else:
            # Remove
            set_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
            if not set_elements:
                T *= cooling
                continue
            x = random.choice(set_elements)
            new_mask = current_mask & ~(1 << x)
            delta = -_delta_add(current_mask, x, three_ap_pairs, x_as_first, x_as_last)
        
        new_score = current_score + delta
        
        if new_score > current_score or (delta < 0 and random.random() < pow(2.718281828, delta / T)):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute
    all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last = _precompute()
    
    # Phase 1: Multiple greedy constructions with different orders
    best_mask = 0
    best_score = 0
    
    # Standard greedy
    mask1 = _greedy_construct(all_4aps_by_element, list(range(1, N + 1)))
    score1 = _count_3aps(mask1, three_ap_pairs)
    if score1 > best_score:
        best_mask = mask1
        best_score = score1
    
    # Reverse greedy
    mask2 = _greedy_construct(all_4aps_by_element, list(range(N, 0, -1)))
    score2 = _count_3aps(mask2, three_ap_pairs)
    if score2 > best_score:
        best_mask = mask2
        best_score = score2
    
    # Randomized greedy (multiple runs)
    random.seed(123)
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask_t = _greedy_construct(all_4aps_by_element, order)
        score_t = _count_3aps(mask_t, three_ap_pairs)
        if score_t > best_score:
            best_mask = mask_t
            best_score = score_t
    
    # Phase 2: Local search on best
    if time.time() < deadline - 25:
        best_mask, best_score = _local_search(best_mask, all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline - 25)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5:
        sa_mask, sa_score = _simulated_annealing(best_mask, all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline - 5, seed=42)
        if sa_score > best_score:
            best_mask = sa_mask
            best_score = sa_score
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 2:
        best_mask, best_score = _local_search(best_mask, all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline - 2)
    
    # Convert mask to sorted list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    
    return result
