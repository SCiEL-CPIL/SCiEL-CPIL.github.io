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

# Precompute 3-APs as (a, b, c) where a < b < c and b-a == c-b
# i.e., a + c = 2b
# For counting, we can iterate over pairs (a, c) with a < c and (c-a) even
# and check if b = (a+c)//2 is in the set

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
    """Local search: try adding/removing elements to improve 3-AP count."""
    start = time.time()
    selected = set(initial)
    current_score = count_3aps(selected)
    
    best_selected = set(selected)
    best_score = current_score
    
    while time.time() - start < time_limit:
        improved = False
        # Try adding elements
        for x in range(1, N + 1):
            if x in selected:
                continue
            candidate = selected | {x}
            if has_4ap(candidate):
                continue
            new_score = count_3aps(candidate)
            if new_score > current_score:
                selected = candidate
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
                break
        if not improved:
            # Try removing elements
            for x in list(selected):
                candidate = selected - {x}
                new_score = count_3aps(candidate)
                if new_score > current_score:
                    selected = candidate
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = set(selected)
                    break
        if not improved:
            # Try swaps
            for x in range(1, N + 1):
                if x in selected:
                    continue
                for y in list(selected):
                    candidate = (selected - {y}) | {x}
                    if has_4ap(candidate):
                        continue
                    new_score = count_3aps(candidate)
                    if new_score > current_score:
                        selected = candidate
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = set(selected)
                        break
                if improved:
                    break
            if not improved:
                break
    
    return sorted(best_selected), best_score

def simulated_annealing(initial, time_limit=10.0):
    """Simulated annealing to escape local optima."""
    start = time.time()
    selected = set(initial)
    current_score = count_3aps(selected)
    
    best_selected = set(selected)
    best_score = current_score
    
    T = 5.0
    cooling = 0.995
    
    while time.time() - start < time_limit and T > 0.01:
        improved = False
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
            new_score = count_3aps(candidate)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                selected = candidate
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
        elif move_type == 'remove':
            if not selected:
                T *= cooling
                continue
            x = random.choice(list(selected))
            candidate = selected - {x}
            new_score = count_3aps(candidate)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                selected = candidate
                current_score = new_score
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
            new_score = count_3aps(candidate)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                selected = candidate
                current_score = new_score
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
            r1_ls, s1_ls = local_search(r1, time_limit=min(5.0, DEADLINE_MARGIN - (time.time() - start_time)))
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
            r2_ls, s2_ls = local_search(r2, time_limit=min(5.0, DEADLINE_MARGIN - (time.time() - start_time)))
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
    
    # Strategy 4: Simulated annealing from best
    if best_result and time.time() - start_time < DEADLINE_MARGIN:
        remaining = DEADLINE_MARGIN - (time.time() - start_time)
        if remaining > 2.0:
            sa_result, sa_score = simulated_annealing(best_result, time_limit=min(15.0, remaining))
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
