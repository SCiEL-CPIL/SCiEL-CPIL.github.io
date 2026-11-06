"""Sidon set construction: minimize max element while covering differences 1..30."""
import time
import random
import numpy as np

# Deadline guard: evaluator kills at 150s, we keep a safety margin
_DEADLINE = 145.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _all_diffs(s):
    """Return list of all positive differences of sorted set s."""
    n = len(s)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(s[j] - s[i])
    return diffs

def _is_sidon(s):
    """Check if sorted list s has all positive differences distinct."""
    n = len(s)
    seen = set()
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            d = s[j] - si
            if d in seen:
                return False
            seen.add(d)
    return True

def _covers_1_to_30(s):
    """Check if all differences 1..30 appear in the set."""
    diffs = set()
    for i in range(len(s)):
        si = s[i]
        for j in range(i + 1, len(s)):
            diffs.add(s[j] - si)
    for d in range(1, 31):
        if d not in diffs:
            return False
    return True

def _valid(s):
    """Full validity check: Sidon + covers 1..30 + size <= 32 + all in 0..10^6."""
    if len(s) > 32:
        return False
    if min(s) < 0 or max(s) > 10**6:
        return False
    if 0 not in s:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

def _score(s):
    """Objective: minimize max element."""
    return max(s)

def _greedy_construct():
    """Greedy construction: realize each missing difference d with a new element next to an existing one."""
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs_list(sorted(a)):
            continue
        found = False
        candidates = []
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in a:
                    candidates.append(x)
        candidates = sorted(set(candidates))
        for x in candidates:
            trial = sorted(a + [x])
            dl = diffs_list(trial)
            if len(set(dl)) == len(dl):
                a = trial
                found = True
                break
        if not found:
            y = max(a) + 1
            while True:
                trial = sorted(a + [y, y + d])
                dl = diffs_list(trial)
                if len(set(dl)) == len(dl):
                    a = trial
                    break
                y += 1
    return a

def _backtracking_search(D, time_limit):
    """
    Search for a valid Sidon set in [0, D] covering 1..30.
    Returns the set if found, None otherwise.
    """
    target_mask = (1 << 30) - 1
    best_sol = None
    
    def dfs(current_set, covered_mask, used_diffs):
        nonlocal best_sol
        if time.time() > time_limit:
            return False
        
        if covered_mask == target_mask:
            best_sol = current_set[:]
            return True
            
        if len(current_set) >= 32:
            return False
            
        last_val = current_set[-1]
        
        # Generate candidates x > last_val
        # Sort by improvement in coverage
        candidates = []
        for x in range(last_val + 1, D + 1):
            valid = True
            new_diffs = []
            for y in current_set:
                d = x - y
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
                
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))
                    
            improvement = bin(new_mask ^ covered_mask).count("1")
            if improvement == 0:
                continue
                
            candidates.append((improvement, x, new_diffs, new_mask))
            
        candidates.sort(key=lambda item: item[0], reverse=True)
        
        for imp, x, new_diffs, new_mask in candidates:
            new_set = current_set + [x]
            new_used = used_diffs.union(new_diffs)
            
            if dfs(new_set, new_mask, new_used):
                return True
                
        return False

    if dfs([0], 0, set()):
        return best_sol
    return None

def _local_search(start_set, max_iters=5000):
    """Local search: try to reduce max element by replacing elements."""
    best = sorted(start_set)
    best_score = _score(best)
    if not _valid(best):
        return best
    
    # Build difference set for quick lookup
    def diff_set(s):
        return set(s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s)))
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        
        improved = False
        # Try removing the largest element and finding a smaller replacement
        # that maintains all required differences
        n = len(best)
        if n <= 1:
            break
        
        # Try to replace each element with a smaller value
        for idx in range(n - 1, -1, -1):  # Start from largest
            old_val = best[idx]
            remaining = best[:idx] + best[idx + 1:]
            
            # If we remove old_val, which differences do we lose?
            lost_diffs = set()
            for v in remaining:
                lost_diffs.add(abs(old_val - v))
            
            # We need to find a new value that covers all lost_diffs (that are in 1..30)
            # and doesn't create duplicate differences
            required = [d for d in lost_diffs if 1 <= d <= 30]
            
            # Try values from 1 to old_val - 1
            for new_val in range(1, old_val):
                if new_val in remaining:
                    continue
                trial = sorted(remaining + [new_val])
                
                # Quick check: all differences distinct
                td = set(trial[j] - trial[i] for i in range(len(trial)) for j in range(i + 1, len(trial)))
                if len(td) != len(trial) * (len(trial) - 1) // 2:
                    continue
                
                # Check coverage
                if all(d in td for d in range(1, 31)):
                    if max(trial) < best_score:
                        best = trial
                        best_score = max(trial)
                        improved = True
                        break
            
            if improved:
                break
        else:
            if not improved:
                break
    
    return best

def _simulated_annealing(start_set, max_iters=100000):
    """Simulated annealing to reduce max element."""
    best = sorted(start_set)
    best_score = _score(best)
    if not _valid(best):
        return best
    
    current = list(best)
    current_score = best_score
    T = 100.0
    cooling = 0.999
    
    def diff_set(s):
        return set(s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s)))
    
    for it in range(max_iters):
        if _time_left() < 5:
            break
        
        T *= cooling
        if T < 0.01:
            break
        
        # Generate neighbor: move one element to a random nearby position
        idx = random.randint(0, len(current) - 1)
        if idx == 0:
            continue  # Keep 0 fixed
        old_val = current[idx]
        # Try moving to a random position in [1, max(current)+10]
        new_val = random.randint(1, max(current) + 10)
        if new_val == old_val or new_val in [current[j] for j in range(len(current)) if j != idx]:
            continue
        
        current[idx] = new_val
        trial = sorted(current)
        
        # Check Sidon
        td = set(trial[j] - trial[i] for i in range(len(trial)) for j in range(i + 1, len(trial)))
        if len(td) != len(trial) * (len(trial) - 1) // 2:
            current[idx] = old_val
            continue
        
        # Check coverage
        if not all(d in td for d in range(1, 31)):
            current[idx] = old_val
            continue
        
        new_score = max(trial)
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_score = new_score
            if new_score < best_score:
                best = list(trial)
                best_score = new_score
        else:
            current[idx] = old_val
    
    return best

def _differential_evolution(pop_size=20, max_gens=200, F=0.8, CR=0.9):
    """Differential evolution for Sidon set optimization."""
    # Initialize population with greedy + random perturbations
    base = _greedy_construct()
    
    def random_perturb(base_set):
        s = list(base_set)
        for _ in range(random.randint(1, 3)):
            idx = random.randint(1, len(s) - 1)
            s[idx] = max(1, s[idx] + random.randint(-50, 50))
        # Remove duplicates
        s = sorted(set(s))
        if 0 not in s:
            s.append(0)
            s = sorted(s)
        return s[:32]
    
    population = [list(base)]
    for _ in range(pop_size - 1):
        population.append(random_perturb(base))
    
    best = sorted(base)
    best_score = _score(best)
    
    def diff_set(s):
        return set(s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s)))
    
    for gen in range(max_gens):
        if _time_left() < 5:
            break
        
        new_pop = []
        for i in range(pop_size):
            # Select three distinct parents
            a_idx, b_idx, c_idx = random.sample(range(pop_size), 3)
            a, b, c = population[a_idx], population[b_idx], population[c_idx]
            
            # Mutation
            v = list(a)
            for j in range(len(v)):
                if random.random() < CR:
                    v[j] = a[j] + F * (b[j % len(b)] - c[j % len(c)])
                else:
                    v[j] = a[j]
            
            # Convert to integers, ensure non-negative, distinct, includes 0
            v = [max(0, int(round(x))) for x in v]
            if 0 not in v:
                v[0] = 0
            v = sorted(set(v))
            if len(v) < 2:
                v = [0, 1]
            v = v[:32]
            
            # Check validity
            td = diff_set(v)
            if len(td) == len(v) * (len(v) - 1) // 2 and all(d in td for d in range(1, 31)):
                v_score = max(v)
                if v_score < best_score:
                    best = list(v)
                    best_score = v_score
                
                # Tournament selection
                if i < pop_size:
                    new_pop.append(v)
                else:
                    new_pop.append(list(population[i]))
            else:
                new_pop.append(list(population[i]))
        
        population = new_pop
    
    return best

def _smart_backtracking():
    """Improved backtracking with better heuristics and pruning."""
    best = [0]
    best_max = 10**6
    
    def dfs(current, used_diffs):
        nonlocal best, best_max
        if _time_left() < 10:
            return
        if max(current) >= best_max:
            return
        
        # Check if we've covered all 1..30
        if all(d in used_diffs for d in range(1, 31)):
            if max(current) < best_max:
                best = list(current)
                best_max = max(current)
            return
        
        if len(current) >= 32:
            return
        
        missing = [d for d in range(1, 31) if d not in used_diffs]
        
        # Try adding next element
        # Heuristic: the new element should be small to keep diameter low
        # But it must create at least one missing difference
        max_current = max(current)
        
        # Try values from 1 to best_max - 1
        # But we can prune: if x - max_current > some missing d, then x can't create that d
        # Actually, x can create difference d if x - v = d for some v in current
        # So x = v + d for some v in current, d in missing
        # Or x = v - d for some v in current (but x > 0 and x not in current)
        
        candidates = set()
        for v in current:
            for d in missing:
                x = v + d
                if x > 0 and x not in current and x < best_max:
                    candidates.add(x)
                x2 = v - d
                if x2 > 0 and x2 not in current and x2 < best_max:
                    candidates.add(x2)
        
        # Sort candidates by value (prefer smaller)
        candidates = sorted(candidates)
        
        for x in candidates:
            # Check all new differences
            new_diffs = set()
            ok = True
            for v in current:
                d = abs(x - v)
                if d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            
            if not ok:
                continue
            
            if len(new_diffs) != len(current):
                continue
            
            if not any(d in new_diffs for d in missing):
                continue
            
            new_used = used_diffs | new_diffs
            new_current = sorted(current + [x])
            
            dfs(new_current, new_used)
    
    dfs([0], set())
    
    return best

def construct():
    """Main entry point: find Sidon set minimizing max element."""
    global _start_time
    _start_time = time.time()
    
    # Get a baseline solution to ensure we have something valid
    baseline = _greedy_construct()
    best_val = max(baseline)
    best_set = baseline
    
    # Try to find a smaller diameter using backtracking
    # Start from a lower bound.
    # We need at least 9 elements.
    # A safe lower bound for diameter is 50.
    # We iterate D upwards.
    
    start_D = 50
    end_D = best_val
    
    for D in range(start_D, end_D + 1):
        if time.time() > _DEADLINE - 5:
            break
            
        # Allow 5 seconds for each D
        t_start = time.time()
        limit = t_start + 5.0
        
        sol = _backtracking_search(D, limit)
        
        if sol:
            # This is the minimum diameter found so far
            # We can return immediately because we are iterating D in increasing order
            best_set = sol
            best_val = max(sol)
            break
    
    # Local search refinement
    best_set = _local_search(best_set)
    
    # Simulated annealing
    if _time_left() > 10:
        sa_result = _simulated_annealing(best_set, max_iters=50000)
        if _valid(sa_result) and _score(sa_result) < _score(best_set):
            best_set = sa_result
    
    # Differential evolution
    if _time_left() > 15:
        de_result = _differential_evolution(pop_size=15, max_gens=100)
        if _valid(de_result) and _score(de_result) < _score(best_set):
            best_set = de_result
    
    # Final local search polish
    best_set = _local_search(best_set, max_iters=3000)
    
    # Ensure validity
    if not _valid(best_set):
        # Last resort: use greedy again
        best_set = _greedy_construct()
    
    return sorted(best_set)
