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

def _backtrack_mrv(deadline, max_diam=80):
    """Fast backtracking with bitmask diff tracking and MRV-style pruning."""
    ALL_30 = (1 << 31) - 2  # bits 1..30 set
    best = None
    best_diam = max_diam
    node_count = [0]
    check_interval = 2000
    
    def rec(a, used_mask, depth):
        nonlocal best, best_diam
        if time.time() > deadline:
            return
        node_count[0] += 1
        if node_count[0] % check_interval == 0:
            if time.time() > deadline:
                return
        
        if (used_mask & ALL_30) == ALL_30:
            if a[-1] < best_diam:
                best_diam = a[-1]
                best = list(a)
            return
        if a[-1] >= best_diam:
            return
        if len(a) >= 32:
            return
        
        missing_mask = ALL_30 & ~used_mask
        if missing_mask == 0:
            return
        
        # Feasibility pruning: each missing diff d must be coverable by some x in (a[-1], best_diam)
        # such that x = v + d for some v in a, and x doesn't create conflicts
        # Quick check: for each missing d, is there any v in a with a[-1] < v+d <= best_diam-1?
        upper = best_diam - 1
        if upper < a[-1] + 1:
            return
        for d in range(1, 31):
            if (missing_mask >> d) & 1:
                found = False
                for v in a:
                    x = v + d
                    if a[-1] < x <= upper:
                        found = True
                        break
                if not found:
                    return
        
        start = a[-1] + 1
        end = min(best_diam - 1, max_diam)
        
        # Build candidate list with coverage scoring
        candidates = []
        for x in range(start, end + 1):
            # Check if x creates any conflict
            conflict = False
            new_mask = 0
            for v in a:
                d = x - v
                bit = 1 << d
                if (used_mask & bit) or (new_mask & bit):
                    conflict = True
                    break
                new_mask |= bit
            if conflict:
                continue
            covers = bin(new_mask & missing_mask).count('1')
            candidates.append((covers, x, new_mask))
        
        # Sort: most coverage first, then smallest x
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        for covers, x, new_mask in candidates:
            if time.time() > deadline:
                return
            a.append(x)
            rec(a, used_mask | new_mask, depth + 1)
            a.pop()
    
    rec([0], 0, 0)
    return best, best_diam

def _sa_search(initial, deadline, seed=42):
    """Simulated annealing search with diameter-focused perturbation."""
    rng = random.Random(seed)
    current = list(initial)
    current_cost = _cost(current)
    best = list(current)
    best_cost = current_cost
    best_valid = _is_valid(current)
    
    T = 30.0
    cooling = 0.9997
    
    while time.time() < deadline:
        n = len(current)
        
        # Perturbation: focus on reducing largest elements
        if rng.random() < 0.6:
            # Reduce a large element
            idx = rng.randint(max(1, n - 3), n - 1)
            center = current[idx]
            spread = max(5, center // 3)
            new_val = rng.randint(1, center - 1) if center > 2 else 1
        else:
            # Random perturbation
            idx = rng.randint(1, n - 1)
            center = current[idx]
            spread = max(10, center // 2)
            new_val = rng.randint(1, center + spread)
        
        if new_val != current[idx] and new_val not in current:
            trial = list(current)
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
        
        if rng.random() < 0.02:
            current = list(best)
            current_cost = best_cost
    
    return best, best_cost, best_valid

def _local_optimize(a, deadline):
    """Local optimization: try reducing each element, with incremental checks."""
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
            # Try reducing by small amounts first
            for delta in range(1, min(old, 50)):
                new_val = old - delta
                if new_val <= 0 or new_val in a:
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

def _final_polish(a, deadline):
    """Final high-precision polish: try every possible reduction of each element."""
    a = sorted(a)
    best = list(a)
    best_diam = a[-1]
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(a) - 1, 0, -1):
            if time.time() > deadline:
                break
            old = a[i]
            # Try every possible value from 1 to old-1
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
                        changed = True
                        break
    
    return best, best_diam, _is_valid(best)

def _random_restart_search(deadline):
    """Multiple random restarts with SA."""
    best_set = None
    best_diam = float('inf')
    best_valid = False
    
    rng = random.Random(777)
    
    a0 = _greedy_build()
    if _is_valid(a0):
        best_set = a0
        best_diam = a0[-1]
        best_valid = True
    
    seeds = [1, 2, 3, 42, 123, 456, 789, 1000, 2000, 3000]
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        
        a = [0]
        used = set()
        n_target = rng.randint(7, 12)
        for _ in range(n_target - 1):
            attempts = 0
            while attempts < 100:
                x = rng.randint(1, 100)
                if x in a:
                    continue
                ok = True
                new_diffs = set()
                for v in a:
                    d = abs(x - v)
                    if d in used or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    a.append(x)
                    a.sort()
                    used.update(new_diffs)
                    break
                attempts += 1
        
        sa_result, sa_cost, sa_valid = _sa_search(a, deadline - 5, seed=seed)
        if sa_valid and sa_result[-1] < best_diam:
            best_set = sa_result
            best_diam = sa_result[-1]
            best_valid = True
    
    return best_set, best_diam, best_valid

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    best_set = None
    best_diam = float('inf')
    
    # Phase 1: Tight backtracking for small diameter
    if time.time() < deadline - 60:
        bt_result, bt_diam = _backtrack_mrv(deadline - 60, max_diam=42)
        if bt_result is not None:
            best_set = bt_result
            best_diam = bt_diam
    
    # Phase 2: Widen diameter if needed
    if (best_set is None or best_diam > 40) and time.time() < deadline - 40:
        bt_result2, bt_diam2 = _backtrack_mrv(deadline - 40, max_diam=55)
        if bt_result2 is not None:
            if best_set is None or bt_diam2 < best_diam:
                best_set = bt_result2
                best_diam = bt_diam2
    
    # Phase 3: Even wider if still needed
    if (best_set is None or best_diam > 50) and time.time() < deadline - 25:
        bt_result3, bt_diam3 = _backtrack_mrv(deadline - 25, max_diam=70)
        if bt_result3 is not None:
            if best_set is None or bt_diam3 < best_diam:
                best_set = bt_result3
                best_diam = bt_diam3
    
    # Phase 4: SA refinement
    if time.time() < deadline - 15:
        if best_set is not None:
            sa_result, sa_cost, sa_valid = _sa_search(best_set, deadline - 15, seed=42)
        else:
            a0 = _greedy_build()
            sa_result, sa_cost, sa_valid = _sa_search(a0, deadline - 15, seed=42)
        if sa_valid:
            if best_set is None or sa_result[-1] < best_diam:
                best_set = sa_result
                best_diam = sa_result[-1]
    
    # Phase 5: Local optimization
    if best_set is not None and time.time() < deadline - 8:
        opt, opt_diam, opt_valid = _local_optimize(best_set, deadline - 8)
        if opt_valid and opt_diam < best_diam:
            best_set = opt
            best_diam = opt_diam
    
    # Phase 6: Final polish
    if best_set is not None and time.time() < deadline - 2:
        pol, pol_diam, pol_valid = _final_polish(best_set, deadline - 2)
        if pol_valid and pol_diam < best_diam:
            best_set = pol
            best_diam = pol_diam
    
    # Fallback
    if best_set is None:
        best_set = _greedy_build()
        best_diam = max(best_set)
    
    return best_set

# EVOLVE-BLOCK-END
