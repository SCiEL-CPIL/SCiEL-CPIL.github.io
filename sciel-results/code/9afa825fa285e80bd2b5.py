"""Differential-evolution search for truncated perfect difference sets."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 148.0
_MARGIN = 3.0

def _valid(s):
    """Check Sidon + covers 1..30 + has 0 + size <= 32 + max <= 10^6."""
    if 0 not in s:
        return False
    n = len(s)
    if n > 32 or n < 2:
        return False
    if max(s) > 10**6:
        return False
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d in seen:
                return False
            seen.add(d)
    for d in range(1, 31):
        if d not in seen:
            return False
    return True

def _penalty(s):
    """Return (penalty, max_val) where penalty=0 means valid."""
    if 0 not in s:
        return (10**9, max(s))
    n = len(s)
    if n > 32 or n < 2:
        return (10**9, max(s))
    if max(s) > 10**6:
        return (10**9, max(s))
    seen = set()
    collisions = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d in seen:
                collisions += 1
            else:
                seen.add(d)
    uncovered = sum(1 for d in range(1, 31) if d not in seen)
    # penalty: collisions dominate, then uncovered, then diameter
    pen = 1000 * collisions + 100 * uncovered + max(s)
    return (pen, max(s))

def _normalize(s, max_size=32):
    """Ensure 0 is present, sort, trim to max_size, ensure distinct positive."""
    s = sorted(set(s))
    if 0 not in s:
        s = [0] + [x for x in s if x > 0]
    # Keep only positive distinct values
    s = sorted(set(x for x in s if x >= 0))
    if len(s) > max_size:
        s = s[:max_size]
    return s

def _rand_individual(rng, D):
    """Generate a random individual: 0 plus k random distinct values in [1, D]."""
    k = rng.randint(10, 20)
    vals = [0] + rng.sample(range(1, D + 1), k)
    return sorted(set(vals))

def _mutate(a, b, c, F, rng, D):
    """Differential evolution mutation: a + F*(b - c)."""
    n = len(a)
    # Pad or trim to same length
    if len(b) < n:
        b = b + [rng.randint(1, D) for _ in range(n - len(b))]
    if len(c) < n:
        c = c + [rng.randint(1, D) for _ in range(n - len(c))]
    if len(b) > n:
        b = b[:n]
    if len(c) > n:
        c = c[:n]
    
    mutated = []
    for i in range(n):
        v = a[i] + F * (b[i] - c[i])
        v = max(0, min(D, int(round(v))))
        mutated.append(v)
    return mutated

def _crossover(mutated, current, CR, rng):
    """Binary crossover between mutated and current."""
    n = len(current)
    trial = []
    for i in range(n):
        if rng.random() < CR:
            trial.append(mutated[i])
        else:
            trial.append(current[i])
    return trial

def _evaluate_and_sort(pop, D):
    """Evaluate population and return sorted by (penalty, max)."""
    scored = []
    for ind in pop:
        pen, mx = _penalty(ind)
        scored.append((pen, mx, ind))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored

def _de_search(rng, D, pop_size, max_iter, deadline, F=0.8, CR=0.7):
    """Differential evolution search within diameter D."""
    pop = [_rand_individual(rng, D) for _ in range(pop_size)]
    
    scored = _evaluate_and_sort(pop, D)
    best_pen, best_mx, best_ind = scored[0]
    
    for it in range(max_iter):
        if time.time() > deadline - _MARGIN:
            break
        
        # Check if we have a valid solution
        if best_pen == best_mx and best_pen < 10**6:
            # Valid: penalty = 0 collisions + 0 uncovered + max = max
            # Actually check properly
            if _valid(best_ind):
                pass  # keep searching for smaller max
        
        for i in range(pop_size):
            if time.time() > deadline - _MARGIN:
                break
            
            # Select three distinct individuals different from i
            candidates = [j for j in range(pop_size) if j != i]
            a_idx, b_idx, c_idx = rng.sample(candidates, 3)
            
            a = pop[a_idx]
            b = pop[b_idx]
            c = pop[c_idx]
            current = pop[i]
            
            # Match lengths
            target_len = len(current)
            if len(a) < target_len:
                a = a + [rng.randint(1, D) for _ in range(target_len - len(a))]
            elif len(a) > target_len:
                a = a[:target_len]
            if len(b) < target_len:
                b = b + [rng.randint(1, D) for _ in range(target_len - len(b))]
            elif len(b) > target_len:
                b = b[:target_len]
            if len(c) < target_len:
                c = c + [rng.randint(1, D) for _ in range(target_len - len(c))]
            elif len(c) > target_len:
                c = c[:target_len]
            
            mutated = _mutate(a, b, c, F, rng, D)
            trial = _crossover(mutated, current, CR, rng)
            
            # Ensure 0 is in trial
            if 0 not in trial:
                trial[0] = 0
            trial = sorted(set(trial))
            
            trial_pen, trial_mx = _penalty(trial)
            cur_pen, cur_mx = _penalty(current)
            
            if trial_pen < cur_pen or (trial_pen == cur_pen and trial_mx < cur_mx):
                pop[i] = trial
            
            # Update best
            if trial_pen < best_pen or (trial_pen == best_pen and trial_mx < best_mx):
                best_pen, best_mx, best_ind = trial_pen, trial_mx, trial
    
    return best_ind, best_mx

def _local_shrink(s, rng, deadline, D):
    """Try to reduce the maximum element via local moves."""
    s = sorted(set(s))
    best_mx = max(s)
    cur_pen, _ = _penalty(s)
    
    for _ in range(5000):
        if time.time() > deadline - _MARGIN:
            break
        
        idx = rng.randint(1, len(s) - 1)
        old = s[idx]
        
        # Try to move this element to a smaller random position
        new = rng.randint(1, min(old - 1, D))
        if new in s:
            continue
        
        s[idx] = new
        s = sorted(set(s))
        new_pen, new_mx = _penalty(s)
        
        if new_pen < cur_pen or (new_pen == cur_pen and new_mx < best_mx):
            cur_pen = new_pen
            if new_mx < best_mx:
                best_mx = new_mx
        else:
            s[idx] = old
            s = sorted(set(s))
    
    return s, best_mx

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)
    
    best = None
    best_max = 10**6
    
    # Try multiple diameter bounds
    D_list = [50, 60, 70, 80, 100, 120, 150]
    
    for D in D_list:
        if time.time() > deadline - _MARGIN:
            break
        
        # DE search
        pop_size = 30
        max_iter = 50
        ind, mx = _de_search(rng, D, pop_size, max_iter, deadline)
        
        if ind is not None and _valid(ind):
            if mx < best_max:
                best = list(ind)
                best_max = mx
        
        # Local shrink on best found
        if best is not None and time.time() < deadline - _MARGIN:
            shrunken, mx2 = _local_shrink(list(best), rng, deadline, D)
            if _valid(shrunken) and mx2 < best_max:
                best = list(shrunken)
                best_max = mx2
        
        # Also try local shrink on the DE result directly
        if ind is not None and _valid(ind) and time.time() < deadline - _MARGIN:
            shrunken, mx3 = _local_shrink(list(ind), rng, deadline, D)
            if _valid(shrunken) and mx3 < best_max:
                best = list(shrunken)
                best_max = mx3
    
    # If we have a good solution, do more aggressive local search
    if best is not None and time.time() < deadline - _MARGIN:
        # Multiple rounds of local shrink with different seeds
        for _ in range(5):
            if time.time() > deadline - _MARGIN:
                break
            shrunken, mx = _local_shrink(list(best), rng, deadline, best_max)
            if _valid(shrunken) and mx < best_max:
                best = list(shrunken)
                best_max = mx
        
        # Try removing elements
        s = list(best)
        changed = True
        while changed and time.time() < deadline - _MARGIN:
            changed = False
            for i in range(len(s) - 1, 0, -1):
                if s[i] == 0:
                    continue
                trial = [x for x in s if x != s[i]]
                if len(trial) >= 2 and _valid(trial):
                    s = trial
                    changed = True
                    break
        if _valid(s):
            m = max(s)
            if m < best_max:
                best = list(s)
                best_max = m
    
    # Fallback: known construction
    if best is None:
        # Try a simple greedy construction
        a = [0]
        for d in range(1, 31):
            placed = False
            for x in range(1, 200):
                if x in a:
                    continue
                trial = sorted(a + [x])
                diffs = []
                valid = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        dd = trial[j] - trial[i]
                        if dd in diffs:
                            valid = False
                            break
                        diffs.append(dd)
                    if not valid:
                        break
                if valid:
                    a = trial
                    placed = True
                    break
            if not placed:
                # Add two new elements to cover d
                for y in range(1, 500):
                    trial = sorted(a + [y, y + d])
                    diffs = []
                    valid = True
                    for i in range(len(trial)):
                        for j in range(i + 1, len(trial)):
                            dd = trial[j] - trial[i]
                            if dd in diffs:
                                valid = False
                                break
                            diffs.append(dd)
                        if not valid:
                            break
                    if valid:
                        a = trial
                        break
        if _valid(a):
            return a
        # Ultimate fallback
        return [0, 1, 4, 10, 12, 17, 20, 23, 25, 29]
    
    return best
# EVOLVE-BLOCK-END
