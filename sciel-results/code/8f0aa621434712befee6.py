import time
import random
import sys

N = 256

def _precompute():
    """Precompute all 4-APs and 3-AP structures."""
    # All 4-APs: (a, a+d, a+2d, a+3d)
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                all_4aps.append((a, b, c, e))
    
    # For each element x, list of 4-APs containing x
    # Each entry is (x, other1, other2, other3)
    aps_by_element = [[] for _ in range(N + 1)]
    for (a, b, c, e) in all_4aps:
        aps_by_element[a].append((b, c, e))
        aps_by_element[b].append((a, c, e))
        aps_by_element[c].append((a, b, e))
        aps_by_element[e].append((a, b, c))
    
    # All 3-APs: (a, b, c) with b-a = c-b > 0, i.e., a < b < c and a+c=2b
    all_3aps = []
    for b in range(2, N):
        for a in range(1, b):
            c = 2 * b - a
            if c > b and c <= N:
                all_3aps.append((a, b, c))
    
    return all_4aps, aps_by_element, all_3aps

def _creates_4ap(mask, x, aps_by_element):
    """Check if adding x to mask creates a 4-AP."""
    for (o1, o2, o3) in aps_by_element[x]:
        if (mask & (1 << o1)) and (mask & (1 << o2)) and (mask & (1 << o3)):
            return True
    return False

def _delta_add(mask, x):
    """Count new 3-APs created by adding x to the set represented by mask."""
    count = 0
    # x as middle: (a, x, c) where a < x < c, c = 2x - a
    for a in range(1, x):
        if mask & (1 << a):
            c = 2 * x - a
            if c > x and c <= N and (mask & (1 << c)):
                count += 1
    # x as first: (x, b, c) where b = x+d, c = x+2d
    for d in range(1, N - x + 1):
        b = x + d
        c = x + 2 * d
        if b <= N and c <= N:
            if (mask & (1 << b)) and (mask & (1 << c)):
                count += 1
    # x as last: (a, b, x) where b = (a+x)/2
    for a in range(1, x):
        s = a + x
        if s % 2 == 0:
            b = s // 2
            if a < b < x and (mask & (1 << b)):
                count += 1
    return count

def _delta_remove(mask, x):
    """Count 3-APs destroyed by removing x from the set represented by mask."""
    return _delta_add(mask & ~(1 << x), x)

def _count_3aps(mask):
    """Count 3-APs in set represented by bitmask."""
    count = 0
    for b in range(2, N):
        if not (mask & (1 << b)):
            continue
        for a in range(1, b):
            if mask & (1 << a):
                c = 2 * b - a
                if c > b and c <= N and (mask & (1 << c)):
                    count += 1
    return count

def _greedy_construct(aps_by_element):
    """Greedy: add elements in order, skip if creates 4-AP."""
    mask = 0
    for x in range(1, N + 1):
        if not _creates_4ap(mask, x, aps_by_element):
            mask |= (1 << x)
    return mask

def _greedy_reverse(aps_by_element):
    """Greedy from the other end."""
    mask = 0
    for x in range(N, 0, -1):
        if not _creates_4ap(mask, x, aps_by_element):
            mask |= (1 << x)
    return mask

def _greedy_random(aps_by_element, seed):
    """Greedy with random ordering."""
    random.seed(seed)
    order = list(range(1, N + 1))
    random.shuffle(order)
    mask = 0
    for x in order:
        if not _creates_4ap(mask, x, aps_by_element):
            mask |= (1 << x)
    return mask

def _local_search(mask, aps_by_element, deadline):
    """Hill climbing with add/remove/swap moves."""
    best_mask = mask
    best_score = _count_3aps(mask)
    
    while time.time() < deadline - 2.0:
        improved = False
        set_elements = [i for i in range(1, N + 1) if best_mask & (1 << i)]
        unset_elements = [i for i in range(1, N + 1) if not (best_mask & (1 << i))]
        
        # Try all single additions
        for y in unset_elements[:]:
            if _creates_4ap(best_mask, y, aps_by_element):
                continue
            new_score = best_score + _delta_add(best_mask, y)
            if new_score > best_score:
                best_mask = best_mask | (1 << y)
                best_score = new_score
                improved = True
                unset_elements.remove(y)
                set_elements.append(y)
        
        # Try all single removals
        for x in set_elements[:]:
            new_score = best_score - _delta_remove(best_mask, x)
            if new_score > best_score:
                best_mask = best_mask & ~(1 << x)
                best_score = new_score
                improved = True
                set_elements.remove(x)
                unset_elements.append(x)
        
        # Try swaps
        for x in set_elements[:]:
            for y in unset_elements[:]:
                temp_mask = best_mask & ~(1 << x)
                if _creates_4ap(temp_mask, y, aps_by_element):
                    continue
                new_score = best_score - _delta_remove(best_mask, x) + _delta_add(temp_mask, y)
                if new_score > best_score:
                    best_mask = temp_mask | (1 << y)
                    best_score = new_score
                    improved = True
                    set_elements.remove(x)
                    unset_elements.remove(y)
                    set_elements.append(y)
                    unset_elements.append(x)
                    break
        
        if not improved:
            break
    
    return best_mask, best_score

def _simulated_annealing(mask, aps_by_element, deadline):
    """Simulated annealing for further improvement."""
    random.seed(123)
    current_mask = mask
    current_score = _count_3aps(mask)
    best_mask = current_mask
    best_score = current_score
    
    T = 5.0
    T_min = 0.01
    cooling = 0.9998
    
    while time.time() < deadline - 5.0 and T > T_min:
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
            if _creates_4ap(temp_mask, y, aps_by_element):
                T *= cooling
                continue
            new_mask = temp_mask | (1 << y)
            delta = _delta_add(temp_mask, y) - _delta_remove(current_mask, x)
            new_score = current_score + delta
            
        elif move_type < 0.7:
            # Add
            unset_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
            if not unset_elements:
                T *= cooling
                continue
            y = random.choice(unset_elements)
            if _creates_4ap(current_mask, y, aps_by_element):
                T *= cooling
                continue
            new_mask = current_mask | (1 << y)
            delta = _delta_add(current_mask, y)
            new_score = current_score + delta
            
        else:
            # Remove
            set_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
            if not set_elements:
                T *= cooling
                continue
            x = random.choice(set_elements)
            new_mask = current_mask & ~(1 << x)
            delta = -_delta_remove(current_mask, x)
            new_score = current_score + delta
        
        # Accept/reject
        if new_score > current_score or random.random() < pow(2.718281828, delta / T):
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        
        T *= cooling
    
    return best_mask, best_score

def _two_opt_local(mask, aps_by_element, deadline):
    """More aggressive local search with pair moves."""
    best_mask = mask
    best_score = _count_3aps(mask)
    
    while time.time() < deadline - 3.0:
        improved = False
        set_elements = [i for i in range(1, N + 1) if best_mask & (1 << i)]
        unset_elements = [i for i in range(1, N + 1) if not (best_mask & (1 << i))]
        
        # Try double additions
        for i in range(len(unset_elements)):
            if time.time() > deadline - 3.0:
                break
            y1 = unset_elements[i]
            if _creates_4ap(best_mask, y1, aps_by_element):
                continue
            temp_mask = best_mask | (1 << y1)
            for j in range(i + 1, len(unset_elements)):
                y2 = unset_elements[j]
                if _creates_4ap(temp_mask, y2, aps_by_element):
                    continue
                new_score = best_score + _delta_add(best_mask, y1) + _delta_add(temp_mask, y2)
                if new_score > best_score:
                    best_mask = temp_mask | (1 << y2)
                    best_score = new_score
                    improved = True
                    set_elements.extend([y1, y2])
                    unset_elements.remove(y2)
                    unset_elements.remove(y1)
                    break
        
        # Try double removals
        for i in range(len(set_elements)):
            if time.time() > deadline - 3.0:
                break
            x1 = set_elements[i]
            temp_mask = best_mask & ~(1 << x1)
            for j in range(i + 1, len(set_elements)):
                x2 = set_elements[j]
                new_score = best_score - _delta_remove(best_mask, x1) - _delta_remove(temp_mask, x2)
                if new_score > best_score:
                    best_mask = temp_mask & ~(1 << x2)
                    best_score = new_score
                    improved = True
                    set_elements.remove(x2)
                    set_elements.remove(x1)
                    unset_elements.extend([x1, x2])
                    break
        
        # Try add+remove
        for x in set_elements[:]:
            if time.time() > deadline - 3.0:
                break
            for y in unset_elements[:]:
                temp_mask = best_mask & ~(1 << x)
                if _creates_4ap(temp_mask, y, aps_by_element):
                    continue
                new_score = best_score - _delta_remove(best_mask, x) + _delta_add(temp_mask, y)
                if new_score > best_score:
                    best_mask = temp_mask | (1 << y)
                    best_score = new_score
                    improved = True
                    set_elements.remove(x)
                    unset_elements.remove(y)
                    set_elements.append(y)
                    unset_elements.append(x)
                    break
        
        if not improved:
            break
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute
    all_4aps, aps_by_element, all_3aps = _precompute()
    
    # Generate multiple initial solutions
    candidates = []
    
    # Greedy forward
    m1 = _greedy_construct(aps_by_element)
    candidates.append(m1)
    
    # Greedy reverse
    m2 = _greedy_reverse(aps_by_element)
    candidates.append(m2)
    
    # Several random-order greedy
    for seed in [1, 7, 42, 123, 999]:
        m = _greedy_random(aps_by_element, seed)
        candidates.append(m)
    
    # Local search on each candidate
    best_mask = None
    best_score = -1
    
    for m in candidates:
        if time.time() > deadline - 10.0:
            break
        lm, ls = _local_search(m, aps_by_element, deadline)
        if ls > best_score:
            best_score = ls
            best_mask = lm
    
    # Simulated annealing on best
    if time.time() < deadline - 15.0:
        sa_mask, sa_score = _simulated_annealing(best_mask, aps_by_element, deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_mask = sa_mask
    
    # More local search
    if time.time() < deadline - 5.0:
        lm, ls = _local_search(best_mask, aps_by_element, deadline)
        if ls > best_score:
            best_score = ls
            best_mask = lm
    
    # Two-opt
    if time.time() < deadline - 3.0:
        lm, ls = _two_opt_local(best_mask, aps_by_element, deadline)
        if ls > best_score:
            best_score = ls
            best_mask = lm
    
    # Final local search
    if time.time() < deadline - 1.0:
        lm, ls = _local_search(best_mask, aps_by_element, deadline)
        if ls > best_score:
            best_score = ls
            best_mask = lm
    
    # Convert mask to sorted list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
