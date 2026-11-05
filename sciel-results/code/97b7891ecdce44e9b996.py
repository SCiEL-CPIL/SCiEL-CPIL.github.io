"""Optimized construction for max 3-AP, no 4-AP subset of [1,256]."""
import time
import random
import sys
from collections import Counter

N = 256
DEADLINE_MARGIN = 15  # seconds before the 90s limit

# Precompute all 4-term APs
FAP = []  # list of (a, b, c, d) tuples (1-indexed)
for a in range(1, N + 1):
    for d in range(1, (N - a) // 3 + 1):
        b = a + d
        c = b + d
        e = c + d
        if e <= N:
            FAP.append((a, b, c, e))

# For each element, precompute which 4-APs it participates in
elem_to_4aps = {x: [] for x in range(1, N + 1)}
for i, (a, b, c, e) in enumerate(FAP):
    for elem in (a, b, c, e):
        elem_to_4aps[elem].append(i)

# Bitmask helpers
def to_mask(selected):
    m = 0
    for x in selected:
        m |= (1 << (x - 1))
    return m

def from_mask(m):
    return [x + 1 for x in range(N) if (m >> x) & 1]

def has_4ap_mask(mask):
    """Check if bitmask has any nonconstant 4-term AP using precomputed masks."""
    for (a, b, c, e) in FAP:
        if (mask >> (a-1)) & 1 and (mask >> (b-1)) & 1 and (mask >> (c-1)) & 1 and (mask >> (e-1)) & 1:
            return True
    return False

# Precompute 4-AP masks for faster checking
FAP_MASKS = []
for (a, b, c, e) in FAP:
    m = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1)) | (1 << (e-1))
    FAP_MASKS.append(m)

def has_4ap_mask_fast(mask):
    for fm in FAP_MASKS:
        if (mask & fm) == fm:
            return True
    return False

# Precompute for each element, the list of 4-AP masks that contain it
elem_4ap_masks = {x: [] for x in range(1, N + 1)}
for i, fm in enumerate(FAP_MASKS):
    for elem in range(1, N + 1):
        if (fm >> (elem - 1)) & 1:
            elem_4ap_masks[elem].append(fm)

# Precompute 3-AP contribution: for each element x, which 3-APs does it complete?
# A 3-AP is (a, b, c) with a+c=2b. Element x can be a, b, or c.
# Precompute: for each x, list of (other1, other2) pairs such that {x, other1, other2} forms a 3-AP

# For counting delta when adding x: count 3-APs where x is the new element
# x can be left (a): need b, c in set where b-x = c-b => b = x+d, c = x+2d
# x can be middle (b): need a, c in set where b-a = c-b => a = x-d, c = x+d
# x can be right (c): need a, b in set where b-a = c-b => b = x-d, a = x-2d

def count_3aps(selected):
    """Count distinct 3-term APs with positive difference in selected set."""
    sel = set(selected)
    count = 0
    for a in range(1, N + 1):
        if a not in sel:
            continue
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = b + d
            if b in sel and c in sel:
                count += 1
    return count

def count_3aps_delta_add(selected, x):
    """Count new 3-APs created by adding x to selected."""
    sel = set(selected)
    count = 0
    # x as left endpoint: (x, x+d, x+2d)
    for d in range(1, (N - x) // 2 + 1):
        b = x + d
        c = x + 2 * d
        if b in sel and c in sel:
            count += 1
    # x as middle: (x-d, x, x+d)
    for d in range(1, x):
        a = x - d
        c = x + d
        if c <= N and a in sel and c in sel:
            count += 1
    # x as right endpoint: (x-2d, x-d, x)
    for d in range(1, (x - 1) // 2 + 1):
        b = x - d
        a = x - 2 * d
        if a >= 1 and a in sel and b in sel:
            count += 1
    return count

def count_3aps_delta_remove(selected, x):
    """Count 3-APs destroyed by removing x from selected."""
    sel = set(selected)
    count = 0
    # x as left endpoint: (x, x+d, x+2d)
    for d in range(1, (N - x) // 2 + 1):
        b = x + d
        c = x + 2 * d
        if b in sel and c in sel:
            count += 1
    # x as middle: (x-d, x, x+d)
    for d in range(1, x):
        a = x - d
        c = x + d
        if c <= N and a in sel and c in sel:
            count += 1
    # x as right endpoint: (x-2d, x-d, x)
    for d in range(1, (x - 1) // 2 + 1):
        b = x - d
        a = x - 2 * d
        if a >= 1 and a in sel and b in sel:
            count += 1
    return count

def has_4ap(selected):
    """Check if selected set has any nonconstant 4-term AP."""
    sel = set(selected)
    for (a, b, c, e) in FAP:
        if a in sel and b in sel and c in sel and e in sel:
            return True
    return False

def greedy_construct():
    """Greedy: add elements that maximize 3-AP contribution while avoiding 4-APs."""
    selected = set()
    for x in range(1, N + 1):
        candidate = selected | {x}
        if not has_4ap(candidate):
            selected.add(x)
    return sorted(selected)

def greedy_reverse_construct():
    """Start with all elements, remove those in 4-APs to maximize remaining 3-APs."""
    selected = set(range(1, N + 1))
    changed = True
    while changed:
        changed = False
        violating = []
        for (a, b, c, e) in FAP:
            if a in selected and b in selected and c in selected and e in selected:
                violating.append((a, b, c, e))
        if not violating:
            break
        elem_4ap_count = Counter()
        for (a, b, c, e) in violating:
            for elem in (a, b, c, e):
                elem_4ap_count[elem] += 1
        worst_elem = max(elem_4ap_count, key=lambda x: elem_4ap_count[x])
        selected.remove(worst_elem)
        changed = True
    return sorted(selected)

def local_search(initial, time_limit=5.0):
    """Local search with delta evaluation: try adding/removing/swapping elements."""
    start = time.time()
    selected = set(initial)
    current_score = count_3aps(selected)
    
    best_selected = set(selected)
    best_score = current_score
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try adding elements (use delta)
        for x in range(1, N + 1):
            if x in selected:
                continue
            # Check 4-AP constraint
            candidate = selected | {x}
            if has_4ap(candidate):
                continue
            delta = count_3aps_delta_add(selected, x)
            if delta > 0:
                selected = candidate
                current_score += delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
                break
        
        if not improved:
            # Try removing elements
            for x in list(selected):
                delta = -count_3aps_delta_remove(selected, x)
                if delta > 0:
                    selected.discard(x)
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = set(selected)
                    break
        
        if not improved:
            # Try swaps
            found_swap = False
            for x in range(1, N + 1):
                if x in selected:
                    continue
                for y in list(selected):
                    candidate = (selected - {y}) | {x}
                    if has_4ap(candidate):
                        continue
                    delta_remove = count_3aps_delta_remove(selected, y)
                    delta_add = count_3aps_delta_add(selected - {y}, x)
                    net = delta_add - delta_remove
                    if net > 0:
                        selected = candidate
                        current_score += net
                        improved = True
                        found_swap = True
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = set(selected)
                        break
                if found_swap:
                    break
            if not improved:
                break
    
    return sorted(best_selected), best_score

def simulated_annealing(initial, time_limit=10.0):
    """Simulated annealing with delta evaluation to escape local optima."""
    start = time.time()
    selected = set(initial)
    current_score = count_3aps(selected)
    
    best_selected = set(selected)
    best_score = current_score
    
    T = 5.0
    cooling = 0.995
    
    while time.time() - start < time_limit and T > 0.01:
        # Pick a random move: add, remove, or swap
        move_type = random.choice(['add', 'remove', 'swap'])
        
        if move_type == 'add':
            unselected = [x for x in range(1, N + 1) if x not in selected]
            if not unselected:
                T *= cooling
                continue
            x = random.choice(unselected)
            candidate = selected | {x}
            if has_4ap(candidate):
                T *= cooling
                continue
            delta = count_3aps_delta_add(selected, x)
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                selected = candidate
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
        elif move_type == 'remove':
            if not selected:
                T *= cooling
                continue
            x = random.choice(list(selected))
            delta = -count_3aps_delta_remove(selected, x)
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                selected.discard(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
        else:  # swap
            unselected = [x for x in range(1, N + 1) if x not in selected]
            if not unselected or not selected:
                T *= cooling
                continue
            x = random.choice(unselected)
            y = random.choice(list(selected))
            candidate = (selected - {y}) | {x}
            if has_4ap(candidate):
                T *= cooling
                continue
            delta_remove = count_3aps_delta_remove(selected, y)
            delta_add = count_3aps_delta_add(selected - {y}, x)
            delta = delta_add - delta_remove
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                selected = candidate
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
        
        T *= cooling
    
    return sorted(best_selected), best_score

def construct():
    start_time = time.time()
    
    best_result = []
    best_score = -1
    
    # Strategy 1: Greedy forward
    if time.time() - start_time < DEADLINE_MARGIN:
        r1 = greedy_construct()
        s1 = count_3aps(r1)
        if s1 > best_score:
            best_score = s1
            best_result = r1
        if time.time() - start_time < DEADLINE_MARGIN:
            remaining = DEADLINE_MARGIN - (time.time() - start_time)
            r1_ls, s1_ls = local_search(r1, time_limit=min(5.0, remaining))
            if s1_ls > best_score:
                best_score = s1_ls
                best_result = r1_ls
    
    # Strategy 2: Greedy reverse
    if time.time() - start_time < DEADLINE_MARGIN:
        r2 = greedy_reverse_construct()
        s2 = count_3aps(r2)
        if s2 > best_score:
            best_score = s2
            best_result = r2
        if time.time() - start_time < DEADLINE_MARGIN:
            remaining = DEADLINE_MARGIN - (time.time() - start_time)
            r2_ls, s2_ls = local_search(r2, time_limit=min(5.0, remaining))
            if s2_ls > best_score:
                best_score = s2_ls
                best_result = r2_ls
    
    # Strategy 3: Randomized greedy with multiple restarts
    if time.time() - start_time < DEADLINE_MARGIN:
        for trial in range(30):
            if time.time() - start_time > DEADLINE_MARGIN:
                break
            random.seed(trial)
            order = list(range(1, N + 1))
            random.shuffle(order)
            selected = set()
            for x in order:
                candidate = selected | {x}
                if not has_4ap(candidate):
                    selected.add(x)
            r3 = sorted(selected)
            s3 = count_3aps(r3)
            if s3 > best_score:
                best_score = s3
                best_result = r3
    
    # Strategy 4: Simulated annealing from best with multiple restarts
    if best_result and time.time() - start_time < DEADLINE_MARGIN:
        remaining = DEADLINE_MARGIN - (time.time() - start_time)
        if remaining > 2.0:
            # Run SA with different random seeds
            for sa_trial in range(3):
                if time.time() - start_time > DEADLINE_MARGIN:
                    break
                remaining = DEADLINE_MARGIN - (time.time() - start_time)
                if remaining < 2.0:
                    break
                random.seed(1000 + sa_trial)
                sa_result, sa_score = simulated_annealing(best_result, time_limit=min(10.0, remaining))
                if sa_score > best_score:
                    best_score = sa_score
                    best_result = sa_result
    
    # Final local search on best
    if best_result and time.time() - start_time < DEADLINE_MARGIN:
        remaining = DEADLINE_MARGIN - (time.time() - start_time)
        if remaining > 1.0:
            final_ls, final_s = local_search(best_result, time_limit=min(10.0, remaining))
            if final_s > best_score:
                best_score = final_s
                best_result = final_ls
    
    if not best_result:
        best_result = [1]
    
    return best_result

def validate(result):
    sel = set(result)
    for (a, b, c, e) in FAP:
        if a in sel and b in sel and c in sel and e in sel:
            return False
    return True

if __name__ == "__main__":
    result = construct()
    score = count_3aps(result)
    valid = validate(result)
    print(f"Result size: {len(result)}")
    print(f"3-AP count: {score}")
    print(f"Valid (no 4-AP): {valid}")
