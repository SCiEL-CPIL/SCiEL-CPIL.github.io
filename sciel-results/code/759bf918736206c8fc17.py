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

def _solve_backtrack(deadline, max_diam_limit=60):
    """
    Iterative deepening backtracking solver.
    Tries to find a Sidon set covering 1..30 with diameter <= limit.
    """
    target_diffs = set(range(1, 31))
    best_solution = None
    start_diam = 35
    for diam in range(start_diam, max_diam_limit + 1):
        if time.time() > deadline - 5:
            break
        solution = _backtrack_single_diam(diam, deadline)
        if solution:
            return solution
    return best_solution

def _backtrack_single_diam(max_diam, deadline):
    """
    Backtracking for a fixed max diameter.
    Returns a list if found, else None.
    """
    target_diffs = set(range(1, 31))
    current_set = [0]
    used_diffs = set()
    
    def rec(current_set, used_diffs):
        if time.time() > deadline:
            return None
        
        if target_diffs.issubset(used_diffs):
            return list(current_set)
        
        missing = target_diffs - used_diffs
        if not missing:
            return list(current_set)
        
        last = current_set[-1]
        remaining_space = max_diam - last
        if remaining_space <= 0:
            return None
        
        # Pruning: if missing diffs can't be covered with remaining space
        # Each new element can cover at most len(current_set) new diffs
        # But we need to be conservative
        if len(missing) > remaining_space * len(current_set):
            return None
        
        candidates = []
        
        for x in range(last + 1, max_diam + 1):
            new_diffs = set()
            valid = True
            for v in current_set:
                d = x - v
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                covers = len(new_diffs & missing)
                candidates.append((covers, x, new_diffs))
        
        if not candidates:
            return None
            
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        if len(candidates) > 50:
            candidates = candidates[:50]
            
        for covers, x, new_diffs in candidates:
            current_set.append(x)
            result = rec(current_set, used_diffs | new_diffs)
            current_set.pop()
            
            if result:
                return result
                
        return None

    return rec(current_set, used_diffs)

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
    
    # Try removing the largest element
    if len(a) > 9:
        a_no_last = a[:-1]
        if _is_valid(a_no_last):
            best = a_no_last
            best_diam = best[-1]
            a = best
    
    # Try reducing each element from largest to smallest
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
            old = a[i]
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
    
    return best, best_diam

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Backtracking search for minimal diameter
    bt_deadline = start_time + 110
    solution = _solve_backtrack(bt_deadline, max_diam_limit=60)
    
    if solution:
        best_set = solution
        best_diam = max(solution)
        ok, _, _ = _is_valid_set(best_set)
        if not ok:
            best_set = None
            best_diam = float('inf')
    
    # Phase 2: Fallback if backtracking failed
    if best_set is None and time.time() < deadline - 10:
        a0 = _greedy_build()
        if _is_valid(a0):
            best_set = a0
            best_diam = a0[-1]
        else:
            sa_result, sa_cost, sa_valid = _sa_search(a0, deadline - 10, seed=42)
            if sa_valid:
                best_set = sa_result
                best_diam = sa_result[-1]
    
    # Phase 3: Final polish to reduce diameter
    if best_set is not None and time.time() < deadline - 2:
        polished, polished_diam = _final_polish(best_set, deadline - 2)
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
