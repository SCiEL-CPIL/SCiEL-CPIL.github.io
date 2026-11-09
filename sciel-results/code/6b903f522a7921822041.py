"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints."""
    n = len(selected)
    # Check if x divides sum of any 3 distinct others in selected
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        return False
    # Check if any existing a divides sum of 3 distinct others including x
    if n >= 2:
        for i in range(n):
            a = selected[i]
            # Pick 2 from selected excluding a, plus x
            for j in range(n):
                if j == i:
                    continue
                for k in range(j+1, n):
                    if k == i:
                        continue
                    s = selected[j] + selected[k] + x
                    if s % a == 0:
                        return False
    return True

def _count_violations_for_new(x, selected):
    """Count how many violations would occur if x is added."""
    violations = 0
    n = len(selected)
    if n >= 3:
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        violations += 1
    if n >= 2:
        for i in range(n):
            a = selected[i]
            for j in range(n):
                if j == i:
                    continue
                for k in range(j+1, n):
                    if k == i:
                        continue
                    s = selected[j] + selected[k] + x
                    if s % a == 0:
                        violations += 1
    return violations

def _is_valid(selected):
    """Check if the set is valid."""
    n = len(selected)
    if n < 4:
        return True
    for i in range(n):
        a = selected[i]
        # Check triplets from selected excluding a
        count = 0
        for j in range(n):
            if j == i:
                continue
            for k in range(j+1, n):
                if k == i:
                    continue
                for l in range(k+1, n):
                    if l == i:
                        continue
                    if (selected[j] + selected[k] + selected[l]) % a == 0:
                        return False
    return True

def _greedy_largest_first():
    """Greedy that picks the largest valid candidate first."""
    selected = []
    remaining = set(range(1, 193))
    
    while remaining and len(selected) < 48:
        best_x = None
        for x in sorted(remaining, reverse=True):
            if _can_add(x, selected):
                best_x = x
                break
        
        if best_x is not None:
            selected.append(best_x)
            remaining.remove(best_x)
        else:
            break
    
    return selected

def _greedy_smallest_first():
    """Greedy that picks the smallest valid candidate first."""
    selected = []
    remaining = set(range(1, 193))
    
    while remaining and len(selected) < 48:
        best_x = None
        for x in sorted(remaining):
            if _can_add(x, selected):
                best_x = x
                break
        
        if best_x is not None:
            selected.append(best_x)
            remaining.remove(best_x)
        else:
            break
    
    return selected

def _greedy_random(ordering_seed=0):
    """Greedy with random ordering."""
    random.seed(ordering_seed)
    selected = []
    remaining = list(range(1, 193))
    random.shuffle(remaining)
    
    while remaining and len(selected) < 48:
        best_x = None
        for x in remaining:
            if _can_add(x, selected):
                best_x = x
                break
        
        if best_x is not None:
            selected.append(best_x)
            remaining.remove(best_x)
        else:
            break
    
    return selected

def _min_conflicts_search(time_limit=15.0, seed=0):
    """Min-conflicts style search: start with a random subset, iteratively fix violations."""
    random.seed(seed)
    start_time = time.time()
    
    # Start with a random set of size ~48
    all_vals = list(range(1, 193))
    random.shuffle(all_vals)
    current = all_vals[:48]
    current_set = set(current)
    
    best_found = []
    best_size = 0
    
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Check if current is valid
        if _is_valid(current):
            if len(current) > best_size:
                best_size = len(current)
                best_found = list(current)
            # Try to add more elements
            added = True
            while added and len(current) < 48:
                added = False
                candidates = [x for x in range(1, 193) if x not in current_set]
                random.shuffle(candidates)
                for x in candidates:
                    if _can_add(x, current):
                        current.append(x)
                        current_set.add(x)
                        added = True
                        break
            if len(current) > best_size:
                best_size = len(current)
                best_found = list(current)
            break
        
        # Find a violated element
        n = len(current)
        violations = []
        for i in range(n):
            a = current[i]
            for j in range(n):
                if j == i:
                    continue
                for k in range(j+1, n):
                    if k == i:
                        continue
                    for l in range(k+1, n):
                        if l == i:
                            continue
                        if (current[j] + current[k] + current[l]) % a == 0:
                            violations.append((i, j, k, l))
                            break
                    if len(violations) > 0 and violations[-1][0] == i:
                        break
                if len(violations) > 0 and violations[-1][0] == i:
                    break
            if len(violations) > 0 and violations[-1][0] == i:
                break
        
        if not violations:
            # Should be valid
            if len(current) > best_size:
                best_size = len(current)
                best_found = list(current)
            break
        
        # Pick a random violated element and try to replace it
        vi, vj, vk, vl = random.choice(violations)
        # Try replacing current[vi] with a random unused value
        unused = [x for x in range(1, 193) if x not in current_set]
        if not unused:
            break
        random.shuffle(unused)
        
        replaced = False
        for new_val in unused[:20]:  # Try up to 20 candidates
            test_set = current[:vi] + [new_val] + current[vi+1:]
            if _is_valid(test_set):
                current = test_set
                current_set.discard(current[vi])  # This is wrong, let me fix
                # Actually need to update properly
                current = test_set
                current_set = set(current)
                replaced = True
                break
        
        if not replaced:
            # Try removing the violated element and adding a different one
            if len(current) > 4:
                removed = current.pop(vi)
                current_set.discard(removed)
                # Try to add something
                candidates = [x for x in range(1, 193) if x not in current_set]
                random.shuffle(candidates)
                for x in candidates[:30]:
                    if _can_add(x, current):
                        current.append(x)
                        current_set.add(x)
                        break
                else:
                    # Put it back
                    current.insert(vi, removed)
                    current_set.add(removed)
    
    return best_found if best_found else current[:best_size] if best_size > 0 else []

def _local_search(selected, time_limit=10.0):
    """Try to improve the set by adding, swapping, or removing+adding elements."""
    if not _is_valid(selected):
        return selected
    
    start = time.time()
    best = list(selected)
    best_set = set(best)
    best_size = len(best)
    
    for iteration in range(5000):
        if time.time() - start > time_limit:
            break
        
        candidates = [x for x in range(1, 193) if x not in best_set]
        if not candidates:
            break
        
        random.shuffle(candidates)
        improved = False
        
        # Try adding an element
        for x in candidates:
            if _can_add(x, best):
                best.append(x)
                best_set.add(x)
                best_size += 1
                improved = True
                break
        
        if not improved and len(best) >= 2:
            # Try swapping: remove one, add one
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    break
                old = best[i]
                best.remove(old)
                best_set.discard(old)
                for x in candidates:
                    if _can_add(x, best):
                        best.append(x)
                        best_set.add(x)
                        improved = True
                        break
                if not improved:
                    # Put back
                    best.insert(i, old)
                    best_set.add(old)
        
        if not improved:
            break
    
    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin before 90s kill
    
    result = []
    
    # Largest first
    if time.time() < deadline - 10:
        s = _greedy_largest_first()
        if len(s) > len(result):
            result = s
    
    # Smallest first
    if time.time() < deadline - 10:
        s = _greedy_smallest_first()
        if len(s) > len(result):
            result = s
    
    # Multiple random orderings
    for seed in range(30):
        if time.time() > deadline - 20:
            break
        s = _greedy_random(seed)
        if len(s) > len(result):
            result = s
    
    # Min-conflicts search
    if time.time() < deadline - 15:
        remaining = deadline - 15 - time.time()
        for seed in range(5):
            if time.time() > deadline - 15:
                break
            s = _min_conflicts_search(time_limit=remaining/5.0, seed=seed)
            if len(s) > len(result):
                result = s
    
    # Local search to improve
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        result = _local_search(result, time_limit=remaining_time)
    
    return result
# EVOLVE-BLOCK-END
