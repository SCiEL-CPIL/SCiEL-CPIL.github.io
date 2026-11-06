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

def _solve_backtrack(deadline, max_diam_limit=100):
    """
    Iterative deepening backtracking solver.
    Tries to find a Sidon set covering 1..30 with diameter <= limit.
    Uses heuristic ordering: candidates that cover the most 'rare' missing diffs are tried first.
    """
    # Pre-compute missing diffs set
    target_diffs = set(range(1, 31))
    
    # We will try diameters from a lower bound up to max_diam_limit
    # Lower bound for Sidon set covering 1..N is roughly N/2 + something, but let's start from 30
    # Actually, min diameter for 1..30 is known to be around 35-40.
    
    best_solution = None
    
    # Iterative deepening on diameter
    # Start with a reasonable lower bound. 
    # A set of size k has at most k(k-1)/2 diffs. We need 30 diffs. k=8 gives 28, k=9 gives 36.
    # So we need at least 9 elements.
    # Diameter D allows roughly D^2/4 diffs? No, D^2/2.
    # Let's just start from 35 and go up.
    
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
    
    # Heuristic: Sort missing diffs by "rarity" or just keep track
    # We will generate candidates and sort them by how many NEW missing diffs they cover.
    
    def rec(current_set, used_diffs):
        if time.time() > deadline:
            return None
        
        # Check if we covered all
        if target_diffs.issubset(used_diffs):
            # We found a solution. 
            # Since we are doing iterative deepening, the first one found for this diam is valid.
            # But we want the smallest possible max element? 
            # The problem asks to minimize the largest element.
            # If we iterate diam from low to high, the first solution found has the minimal diameter.
            # However, within a diameter bound, we might find a set with max element < diam.
            # We should return the set.
            return list(current_set)
        
        # Pruning: Can we cover the remaining missing diffs?
        missing = target_diffs - used_diffs
        if not missing:
            return list(current_set)
            
        # Max possible new diffs if we add more elements?
        # If current_set has size n, adding 1 element adds at most n new diffs.
        # We have (max_diam - current_set[-1]) space left.
        # Rough bound: number of new elements we can add is limited by space and Sidon constraint.
        # Simple prune: if len(missing) > len(current_set) * 10: return None (very loose)
        
        # Generate candidates
        # Candidates must be > current_set[-1] and <= max_diam
        last = current_set[-1]
        candidates = []
        
        # To speed up, we don't generate ALL candidates if the space is large.
        # But for diam < 50, the space is small.
        for x in range(last + 1, max_diam + 1):
            if x in current_set:
                continue
            
            new_diffs = set()
            valid = True
            for v in current_set:
                d = x - v
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                # Score: number of missing diffs covered
                covers = len(new_diffs & missing)
                # Prefer candidates that cover more missing diffs
                # Tie-breaker: smaller x? Or larger x to leave room? 
                # Smaller x is usually better for minimizing diameter, but we are fixed on diam.
                # Let's prioritize higher coverage.
                candidates.append((covers, x, new_diffs))
        
        if not candidates:
            return None
            
        # Sort by coverage descending, then x ascending
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        # Limit branching factor to avoid explosion? 
        # For small diameters, branching is manageable.
        # If diam is large, we might need to limit.
        # Let's try all if < 100 candidates, else top 20.
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

def _local_optimize(a, deadline):
    """Local optimization: try reducing each element."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    best_valid = _is_valid(a)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
            old = a[i]
            for new_val in range(1, old):
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
                        best_valid = True
                        improved = True
                        break
    
    return best, best_diam, best_valid

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, keep 10s buffer -> 140s deadline
    deadline = start_time + 140
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Strong Backtracking Search
    # This is the most reliable way to find the minimal diameter solution.
    # We search for the smallest diameter that allows a valid Sidon set.
    # We cap the search at 120s to leave time for fallbacks.
    bt_deadline = start_time + 120
    solution = _solve_backtrack(bt_deadline, max_diam_limit=60)
    
    if solution:
        best_set = solution
        best_diam = max(solution)
        # Verify
        ok, _, _ = _is_valid_set(best_set)
        if not ok:
            best_set = None
            best_diam = float('inf')
    
    # Phase 2: If backtracking failed or found a large diameter, try SA/Local Opt
    if time.time() < deadline - 5:
        if best_set is None:
            # Fallback to Greedy + SA
            a0 = _greedy_build()
            if _is_valid(a0):
                best_set = a0
                best_diam = a0[-1]
            else:
                # Try to fix greedy result with SA
                sa_result, sa_cost, sa_valid = _sa_search(a0, deadline - 5, seed=42)
                if sa_valid:
                    best_set = sa_result
                    best_diam = sa_result[-1]
        else:
            # Try to improve the backtracking solution with local optimization
            # (Though backtracking should already be optimal for the given diameter, 
            # maybe we can find a solution with smaller max element within the same diameter? 
            # No, diameter IS the max element since 0 is in the set. 
            # So if we found a solution with diam D, we are looking for diam < D.
            # _solve_backtrack already iterates diam from low to high.
            # So if it returned, it's the minimal diameter.
            # However, if it hit the time limit during a higher diameter search, 
            # it might have found a non-minimal one? No, it returns immediately upon finding one.
            # So if it returns, it's the first (smallest) diam solution found.
            
            # Still, run local optimize just in case the backtracking heuristic missed a 
            # solution with smaller max element but same "diam" logic? 
            # Actually, max element = diameter. So we are good.
            
            # But what if _solve_backtrack returned a solution with diam 40, 
            # but there is a solution with diam 39 that it missed due to heuristic pruning?
            # The iterative deepening tries 35, 36, 37, 38, 39...
            # If it failed at 39, it tries 40. If it finds one at 40, it returns.
            # It does NOT go back to 39.
            # So we should be confident it's minimal.
            
            pass
    
    # Final Fallback: If we still don't have a valid set
    if best_set is None:
        best_set = _greedy_build()
        best_diam = max(best_set)
        
        # If even greedy failed (shouldn't happen), create a trivial one?
        # Greedy usually works for 1..30.
        
        # Try one more SA restart if we have time
        if time.time() < deadline - 1:
            sa_result, sa_cost, sa_valid = _sa_search(best_set, deadline - 1, seed=999)
            if sa_valid and sa_result[-1] < best_diam:
                best_set = sa_result
                best_diam = sa_result[-1]
                
    return best_set

# EVOLVE-BLOCK-END
