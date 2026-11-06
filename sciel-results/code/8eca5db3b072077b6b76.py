import time
import random
import sys
from math import exp

# EVOLVE-BLOCK-START

def _is_valid_set(a):
    """Check if a is a valid Sidon set covering 1..30. Return (ok, diffs, diam)."""
    a = sorted(a)
    n = len(a)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                return False, diffs, a[-1]
            diffs.add(d)
    for d in range(1, 31):
        if d not in diffs:
            return False, diffs, a[-1]
    return True, diffs, a[-1]

def _cost(a):
    """Cost function: penalize conflicts, missing diffs, and diameter."""
    a = sorted(a)
    n = len(a)
    diffs = set()
    conflict = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = a[j] - a[i]
            if d in diffs:
                conflict += 1
            diffs.add(d)
    missing = sum(1 for d in range(1, 31) if d not in diffs)
    diam = a[-1]
    return 1000 * conflict + 100 * missing + diam

def _is_valid(a):
    ok, _, _ = _is_valid_set(a)
    return ok

def _greedy_build():
    """Build a valid Sidon set greedily, covering 1..30."""
    a = [0]
    used = set()
    for _ in range(50):
        covered = set()
        for x in a:
            for y in a:
                if x != y:
                    covered.add(abs(x - y))
        missing = set(range(1, 31)) - covered
        if not missing:
            break
        best_x = None
        best_score = -1
        for x in range(1, 200):
            if x in a:
                continue
            new_diffs = set()
            ok = True
            for v in a:
                d = abs(x - v)
                if d in used or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            covers = len(new_diffs & missing)
            if covers > best_score:
                best_score = covers
                best_x = x
        if best_x is None:
            break
        a.append(best_x)
        a.sort()
        for v in a[:-1]:
            used.add(abs(best_x - v))
    # Fill remaining missing
    for d in range(1, 31):
        if d not in used:
            found = False
            for x in range(1, 300):
                if x in a:
                    continue
                new_diffs = set()
                ok = True
                for v in a:
                    diff = abs(x - v)
                    if diff in used or diff in new_diffs:
                        ok = False
                        break
                    new_diffs.add(diff)
                if ok and d in new_diffs:
                    a.append(x)
                    a.sort()
                    used.update(new_diffs)
                    found = True
                    break
            if not found:
                x = max(a) + d + 1
                while True:
                    new_diffs = set()
                    ok = True
                    for v in a:
                        diff = x - v
                        if diff in used or diff in new_diffs:
                            ok = False
                            break
                        new_diffs.add(diff)
                    if ok:
                        a.append(x)
                        a.sort()
                        used.update(new_diffs)
                        break
                    x += 1
    return a

def _backtrack_optimal(max_diam, deadline):
    """
    Optimized backtracking with bitmask for used diffs.
    Returns a list if found, else None.
    """
    target_mask = (1 << 31) - 2  # bits 1..30 set
    
    current_set = [0]
    used_mask = 0  # bitmask of used differences
    
    def rec(current_set, used_mask):
        if time.time() > deadline:
            return None
        
        if (used_mask & target_mask) == target_mask:
            return list(current_set)
        
        missing_mask = target_mask & ~used_mask
        if missing_mask == 0:
            return list(current_set)
        
        last = current_set[-1]
        remaining_space = max_diam - last
        if remaining_space <= 0:
            return None
        
        # Pruning: max new useful diffs from one element = len(current_set)
        # Count missing bits
        missing_count = bin(missing_mask).count('1')
        max_new_per_elem = len(current_set)
        if missing_count > remaining_space * max_new_per_elem:
            return None
        
        # Also check: each missing diff d must be <= max_diam (since 0 is in set)
        # If any missing diff > max_diam, impossible
        for d in range(1, 31):
            if (missing_mask >> d) & 1 and d > max_diam:
                return None
        
        candidates = []
        
        for x in range(last + 1, max_diam + 1):
            new_mask = 0
            valid = True
            for v in current_set:
                d = x - v
                bit = 1 << d
                if (used_mask & bit) or (new_mask & bit):
                    valid = False
                    break
                new_mask |= bit
            
            if valid:
                covers = bin(new_mask & missing_mask).count('1')
                # Bonus for covering small missing diffs
                small_cover = 0
                for d in range(1, 16):
                    if (new_mask >> d) & 1 and (missing_mask >> d) & 1:
                        small_cover += 1
                score = covers * 10 + small_cover
                candidates.append((score, covers, x, new_mask))
        
        if not candidates:
            return None
            
        candidates.sort(key=lambda t: (-t[0], -t[1], t[2]))
        
        # Limit candidates to top 25 for speed
        if len(candidates) > 25:
            candidates = candidates[:25]
            
        for score, covers, x, new_mask in candidates:
            current_set.append(x)
            result = rec(current_set, used_mask | new_mask)
            current_set.pop()
            
            if result:
                return result
                
        return None

    return rec(current_set, used_mask)

def _solve_backtrack(deadline, max_diam_limit=55):
    """
    Iterative deepening backtracking solver.
    Tries to find a Sidon set covering 1..30 with diameter <= limit.
    """
    start_diam = 32
    for diam in range(start_diam, max_diam_limit + 1):
        if time.time() > deadline - 3:
            break
        solution = _backtrack_optimal(diam, deadline - 1)
        if solution:
            return solution
    return None

def _sa_search(initial, deadline, seed=42):
    """Simulated annealing search."""
    rng = random.Random(seed)
    current = list(initial)
    current_cost = _cost(current)
    best = list(current)
    best_cost = current_cost
    best_valid = _is_valid(current)
    
    T = 50.0
    cooling = 0.9995
    
    while time.time() < deadline:
        n = len(current)
        num_changes = 1 if rng.random() < 0.7 else 2
        
        trial = list(current)
        for _ in range(num_changes):
            idx = rng.randint(1, n - 1)
            center = trial[idx]
            spread = max(10, center // 2)
            new_val = rng.randint(1, center + spread)
            if new_val != trial[idx] and new_val not in trial:
                trial[idx] = new_val
                trial.sort()
        
        trial_cost = _cost(trial)
        delta = trial_cost - current_cost
        
        if delta < 0 or rng.random() < exp(-delta / max(T, 0.001)):
            current = trial
            current_cost = trial_cost
            current_valid = _is_valid(current)
            
            if current_valid and current_cost < best_cost:
                best = list(current)
                best_cost = current_cost
                best_valid = True
        
        T = max(0.1, T * cooling)
        
        if rng.random() < 0.01:
            current = list(best)
            current_cost = best_cost
    
    return best, best_cost, best_valid

def _final_polish(a, deadline):
    """Final polish: try to reduce max element by removing or adjusting elements."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    
    # Try removing elements (starting from largest) to see if set remains valid
    for i in range(len(a) - 1, 0, -1):
        if time.time() > deadline:
            break
        a_no_i = a[:i] + a[i+1:]
        if _is_valid(a_no_i):
            a = a_no_i
            best = list(a)
            best_diam = a[-1]
            # Restart removal check since indices shifted
            i = len(a) - 1
    
    # Try reducing each element from largest to smallest
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
            old = a[i]
            # Try reducing by 1 first, then by larger amounts
            for new_val in range(old - 1, 0, -1):
                if new_val in a:
                    continue
                trial = list(a)
                trial[i] = new_val
                trial.sort()
                if _is_valid(trial):
                    if trial[-1] < best_diam:
                        a = trial
                        best = list(a)
                        best_diam = trial[-1]
                        improved = True
                        break
            if time.time() > deadline:
                break
    
    # Try a more aggressive local search: swap elements to reduce diameter
    if time.time() < deadline - 1:
        for _ in range(200):
            if time.time() > deadline:
                break
            n = len(a)
            if n < 2:
                break
            i = random.randint(1, n - 1)
            j = random.randint(1, n - 1)
            if i == j:
                continue
            trial = list(a)
            # Try moving element i to a smaller value
            old_val = trial[i]
            for new_val in range(old_val - 1, 0, -1):
                if new_val in trial:
                    continue
                trial[i] = new_val
                trial.sort()
                if _is_valid(trial) and trial[-1] < best_diam:
                    a = trial
                    best = list(a)
                    best_diam = a[-1]
                    break
    
    return best, best_diam

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Backtracking search for minimal diameter
    bt_deadline = start_time + 115
    solution = _solve_backtrack(bt_deadline, max_diam_limit=55)
    
    if solution:
        best_set = solution
        best_diam = max(solution)
        ok, _, _ = _is_valid_set(best_set)
        if not ok:
            best_set = None
            best_diam = float('inf')
    
    # Phase 2: Fallback if backtracking failed
    if best_set is None and time.time() < deadline - 15:
        a0 = _greedy_build()
        if _is_valid(a0):
            best_set = a0
            best_diam = a0[-1]
        else:
            sa_result, sa_cost, sa_valid = _sa_search(a0, deadline - 15, seed=42)
            if sa_valid:
                best_set = sa_result
                best_diam = sa_result[-1]
    
    # Phase 3: Final polish to reduce diameter
    if best_set is not None and time.time() < deadline - 3:
        polished, polished_diam = _final_polish(best_set, deadline - 3)
        if polished_diam < best_diam:
            best_set = polished
            best_diam = polished_diam
    
    # Final Fallback
    if best_set is None:
        best_set = _greedy_build()
        best_diam = max(best_set)
        if time.time() < deadline - 1:
            sa_result, sa_cost, sa_valid = _sa_search(best_set, deadline - 1, seed=999)
            if sa_valid and sa_result[-1] < best_diam:
                best_set = sa_result
                best_diam = sa_result[-1]
                
    return best_set

# EVOLVE-BLOCK-END
