import time
import random
import math
from collections import Counter

_BEST = None
_DEADLINE = time.time() + 110.0

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _fast_greedy(k, n_rows, rng=None, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _sa_optimize(rows, k, iters, rng, deadline):
    """
    Simulated Annealing on the scope.
    State: list of rows (each row is list of k+1 ints, row[0]=0).
    Objective: scope (max entry). We want to minimize it.
    Uses efficient difference tracking for fast cost evaluation.
    """
    n_rows = len(rows)
    current_rows = [list(r) for r in rows]
    best_rows = [list(r) for r in current_rows]
    best_scope = _scope(current_rows)
    
    # Track differences with counts for efficient collision tracking
    diff_count = {}
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                diff_count[d] = diff_count.get(d, 0) + 1
    
    def get_scope_and_collisions(rows):
        scope = max(a for r in rows for a in r[1:])
        collisions = sum(max(0, c - 1) for c in diff_count.values())
        return scope, collisions
    
    current_scope, current_collisions = get_scope_and_collisions(current_rows)
    current_cost = current_scope + 1000 * current_collisions
    best_cost = current_cost
    
    T = 5.0
    cooling = 0.9995
    
    for it in range(iters):
        if time.time() > deadline:
            break
            
        # Pick a random element to change
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)  # index in row, 1..k
        
        old_val = current_rows[i][j]
        
        # Determine range for new value
        # Must be > prev and < next
        prev_val = current_rows[i][j-1]
        next_val = current_rows[i][j+1] if j < k else current_scope + 20
        
        lo = prev_val + 1
        hi = next_val - 1
        
        if lo > hi:
            continue
            
        # Propose new value
        if rng.random() < 0.8:
            span = max(1, hi - lo)
            delta = int(rng.gauss(0, max(1, span * 0.15)))
            new_val = old_val + delta
        else:
            new_val = rng.randint(lo, hi)
        
        if new_val == old_val:
            continue
        
        # Check if new value maintains strict increase
        if new_val <= prev_val:
            continue
        if j < k and new_val >= next_val:
            continue
        
        # Apply change
        current_rows[i][j] = new_val
        
        # Update diff_count efficiently
        # Remove old differences involving position j
        row = current_rows[i]
        for t in range(len(row)):
            if t == j:
                continue
            old_d = abs(row[j] - row[t])
            diff_count[old_d] = diff_count.get(old_d, 0) - 1
            if diff_count[old_d] == 0:
                del diff_count[old_d]
        
        # Add new differences involving position j
        for t in range(len(row)):
            if t == j:
                continue
            new_d = abs(row[j] - row[t])
            diff_count[new_d] = diff_count.get(new_d, 0) + 1
        
        new_scope = max(a for r in current_rows for a in r[1:])
        new_collisions = sum(max(0, c - 1) for c in diff_count.values())
        new_cost = new_scope + 1000 * new_collisions
        
        # Metropolis
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, 0.1)):
            current_cost = new_cost
            current_scope = new_scope
            current_collisions = new_collisions
            if new_cost < best_cost:
                best_cost = new_cost
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Revert
            current_rows[i][j] = old_val
            
            # Revert diff_count
            for t in range(len(row)):
                if t == j:
                    continue
                new_d = abs(row[j] - row[t])
                diff_count[new_d] = diff_count.get(new_d, 0) - 1
                if diff_count[new_d] == 0:
                    del diff_count[new_d]
            
            for t in range(len(row)):
                if t == j:
                    continue
                old_d = abs(row[j] - row[t])
                diff_count[old_d] = diff_count.get(old_d, 0) + 1
            
        T *= cooling
        
    return best_scope, best_rows

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 110.0
    
    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])
            
    rng = random.Random(42)
    
    # 1. Generate several initial candidates using different greedy strategies
    candidates = []
    
    # Standard greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    # Spread greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy with varied strategies
    for trial in range(8):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a range of candidates with varying step sizes
                found = False
                # Vary the search strategy based on trial
                if trial < 3:
                    max_step = rng.randint(1, 8)
                elif trial < 6:
                    max_step = rng.randint(2, 15)
                else:
                    max_step = rng.randint(1, 20)
                
                for t in range(x + 1, x + 1 + max_step * 4):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    # Fallback: find smallest valid
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
        
    # 2. Run SA on the best few candidates
    # Sort candidates by scope
    candidates.sort(key=_scope)
    
    # Run SA on the top 3 candidates with more iterations
    for cand in candidates[:3]:
        if time.time() > _DEADLINE:
            break
        # Run SA for more iterations since we have efficient cost evaluation
        iters = 15000
        s, rows = _sa_optimize(cand, k, iters, rng, _DEADLINE)
        consider(rows)
        
    # 3. If time remains, run multiple SA passes on the best result
    if _BEST is not None and time.time() < _DEADLINE:
        remaining_time = _DEADLINE - time.time()
        if remaining_time > 5:
            # Run multiple SA passes with different temperature schedules
            for pass_num in range(3):
                if time.time() > _DEADLINE:
                    break
                extra_iters = int(remaining_time * 8000)
                if extra_iters > 2000:
                    s, rows = _sa_optimize(_BEST[1], k, extra_iters, rng, _DEADLINE)
                    consider(rows)
                    remaining_time = _DEADLINE - time.time()
            
    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
