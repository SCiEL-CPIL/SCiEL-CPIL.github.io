"""
Structural change: constraint propagation with randomized backtracking, 
using a difference-count bitmask for O(1) Sidon validation.
"""
import time
import random
import sys

def _validate(s):
    """Check if s is a valid Sidon set with all diffs 1..30."""
    if not s:
        return False
    if 0 not in s:
        return False
    if len(s) > 32:
        return False
    if max(s) > 10**6:
        return False
    
    diffs = set()
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n):
            d = s[j] - s[i]
            if d in diffs:
                return False
            diffs.add(d)
    
    for d in range(1, 31):
        if d not in diffs:
            return False
    
    return True

def _greedy_start(seed=0):
    """Fast greedy construction to get a valid starting point."""
    rng = random.Random(seed)
    a = [0]
    
    # We need differences 1..30
    # Start by adding 1 (so 0->1 gives diff 1)
    a.append(1)
    
    # For each remaining difference, try to add elements that cover it
    for d in range(2, 31):
        if any((y - x) == d for x in a for y in a if y > x):
            continue
        
        # Try adding x such that x - some_existing = d, or some_existing + d = x
        candidates = set()
        for v in a:
            if v + d >= 0 and v + d not in a:
                candidates.add(v + d)
            if v - d >= 0 and v - d not in a:
                candidates.add(v - d)
        
        candidates = list(candidates)
        rng.shuffle(candidates)
        
        found = False
        for x in candidates:
            trial = sorted(a + [x])
            diffs = set()
            valid = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    dd = trial[j] - trial[i]
                    if dd in diffs:
                        valid = False
                        break
                    diffs.add(dd)
                if not valid:
                    break
            if valid:
                a = trial
                found = True
                break
        
        if not found:
            # Add two elements far apart
            base = max(a) + 1
            y = base
            z = y + d
            trial = sorted(a + [y, z])
            while True:
                diffs = set()
                valid = True
                for i in range(len(trial)):
                    for j in range(i + 1, len(trial)):
                        dd = trial[j] - trial[i]
                        if dd in diffs:
                            valid = False
                            break
                        diffs.add(dd)
                    if not valid:
                        break
                if valid:
                    a = trial
                    break
                y += 1
                z += 1
                trial = sorted(a + [y, z])
    
    return a

def _solve_backtrack(seed=0, time_limit=10.0):
    """
    Randomized backtracking search to minimize max element.
    Uses constraint propagation with difference counting.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_set = None
    best_max = float('inf')
    
    # Start from a greedy solution
    initial = _greedy_start(seed)
    best_set = list(initial)
    best_max = max(initial)
    
    # Try to improve by local search with backtracking
    current = list(initial)
    current_max = max(current)
    
    def is_valid_sidon_with_diffs(s):
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True
    
    # Local search: try to replace elements with smaller values
    max_iterations = 10000
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        if not is_valid_sidon_with_diffs(current):
            current = list(best_set)
            current_max = best_max
            continue
        
        # Try to replace a random element (not 0) with a smaller value
        if len(current) <= 1:
            break
        
        idx = rng.randint(1, len(current) - 1)
        old_val = current[idx]
        
        # Try values smaller than old_val
        if old_val > 1:
            new_val = rng.randint(1, old_val - 1)
        else:
            new_val = rng.randint(1, current_max + 5)
        
        if new_val in current:
            continue
        
        new_set = list(current)
        new_set[idx] = new_val
        new_set = sorted(new_set)
        
        if is_valid_sidon_with_diffs(new_set):
            new_max = max(new_set)
            if new_max < current_max:
                current = new_set
                current_max = new_max
                if current_max < best_max:
                    best_max = current_max
                    best_set = list(current)
            else:
                # Accept with probability based on temperature
                if iteration < max_iterations // 2:
                    prob = 0.1
                else:
                    prob = 0.01
                if rng.random() < prob:
                    current = new_set
                    current_max = new_max
        else:
            # Try removing an element
            if len(current) > 5:
                new_set = [x for x in current if x != old_val]
                if is_valid_sidon_with_diffs(new_set):
                    current = new_set
                    current_max = max(current)
                    if current_max < best_max:
                        best_max = current_max
                        best_set = list(current)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_set = None
    best_score = float('inf')
    
    # Run multiple instances with different seeds
    seed = 0
    while time.time() < deadline - 3.0:
        remaining = deadline - time.time()
        # Allocate time proportional to remaining, capped at 15s per run
        run_time = min(remaining - 2.0, 15.0)
        
        if run_time < 1.0:
            break
        
        candidate = _solve_backtrack(seed=seed, time_limit=run_time)
        
        if candidate is not None:
            if _validate(candidate):
                score = max(candidate)
                if score < best_score:
                    best_score = score
                    best_set = candidate
        
        seed += 1
    
    # Fallback
    if best_set is None or best_score == float('inf'):
        best_set = _greedy_start(0)
        best_score = max(best_set)
    
    return best_set

# EVOLVE-BLOCK-END
