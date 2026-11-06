# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _positive_diffs(s):
    n = len(s)
    return [s[j] - s[i] for i in range(n) for j in range(i + 1, n)]

def _is_sidon(s):
    d = _positive_diffs(s)
    return len(d) == len(set(d))

def _covers_1_to_30(s):
    d = set(_positive_diffs(s))
    return all(x in d for x in range(1, 31))

def _valid(s):
    if len(s) < 1 or len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

def _greedy_construct():
    a = [0]
    def diffs(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs(sorted(a)):
            continue
        found = False
        # Try adding a single element x such that x - v = d or v - x = d for some v in a
        # Prefer smallest max element
        candidates = []
        for v in a:
            x1 = v + d
            x2 = v - d
            if x1 >= 0 and x1 not in a:
                candidates.append(x1)
            if x2 >= 0 and x2 not in a:
                candidates.append(x2)
        
        # Sort by resulting max element to prefer compact sets
        candidates = sorted(set(candidates), key=lambda x: max(a + [x]))
        
        for x in candidates:
            trial = sorted(a + [x])
            if len(set(diffs(trial))) == len(diffs(trial)):
                a = trial
                found = True
                break
        
        if not found:
            # If no single element works, add two elements y, y+d
            y = max(a) + 1
            while time.time() < _DEADLINE:
                trial = sorted(a + [y, y + d])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    break
                y += 1
    return a

def _local_search():
    """
    Local search to minimize max element.
    """
    best = None
    best_val = 10**9
    rng = random.Random(42)
    
    # Start from greedy
    start_set = _greedy_construct()
    if not _valid(start_set):
        return None
    
    current = start_set[:]
    current_val = max(current)
    
    if current_val < best_val and _valid(current):
        best = current[:]
        best_val = current_val
        
    steps = 0
    max_steps = 50000
    
    while time.time() < _DEADLINE and steps < max_steps:
        steps += 1
        # Pick a random non-zero element to move
        idx = rng.randint(1, len(current) - 1)
        old_val = current[idx]
        
        # Try moving it to a nearby value
        # Bias towards smaller values to reduce max
        delta = rng.randint(-20, 10)
        new_val = old_val + delta
        
        if new_val <= 0:
            continue
            
        # Check for duplicates
        if new_val in current and new_val != old_val:
            continue
            
        # Create candidate
        candidate = current[:]
        candidate[idx] = new_val
        candidate.sort()
        
        # Check Sidon and Coverage
        if _is_sidon(candidate) and _covers_1_to_30(candidate):
            val = max(candidate)
            if val < current_val:
                current = candidate
                current_val = val
                if val < best_val:
                    best_val = val
                    best = current[:]
                
    return best

def _random_local_search():
    """
    Multiple random restarts with local search.
    """
    best = None
    best_val = 10**7
    rng = random.Random(12345)
    
    for attempt in range(200):
        if time.time() > _DEADLINE:
            break
        # random initial set containing 0, size 9-12
        size = rng.randint(9, 12)
        upper = rng.randint(60, 200)
        s = {0}
        while len(s) < size:
            s.add(rng.randint(1, upper))
        s = sorted(s)
        if not _is_sidon(s):
            continue
        cur = s
        cur_val = max(cur)
        for step in range(3000):
            if time.time() > _DEADLINE:
                break
            # mutate: add, remove, or shift an element
            op = rng.random()
            if op < 0.4 and len(cur) < 32:
                # add a random element
                cand = cur + [rng.randint(1, max(cur) + 50)]
                cand = sorted(cand)
                if len(set(cand)) == len(cand) and _is_sidon(cand):
                    cur = cand
            elif op < 0.7 and len(cur) > 9:
                # remove a random non-zero element
                idx = rng.randint(1, len(cur) - 1)
                cand = cur[:idx] + cur[idx + 1:]
                if _is_sidon(cand):
                    cur = cand
            else:
                # shift a random non-zero element
                idx = rng.randint(1, len(cur) - 1)
                delta = rng.randint(-10, 10)
                new_val = cur[idx] + delta
                if new_val <= 0:
                    continue
                cand = cur[:idx] + [new_val] + cur[idx + 1:]
                cand = sorted(cand)
                if len(set(cand)) == len(cand) and _is_sidon(cand):
                    cur = cand
            # check if covers 1..30
            if _covers_1_to_30(cur):
                val = max(cur)
                if val < best_val:
                    best_val = val
                    best = cur[:]
    return best

def construct():
    t0 = time.time()
    
    # 1. Greedy
    g = _greedy_construct()
    best = g
    best_val = max(g)
    
    # 2. Local Search (deterministic, from greedy)
    ls = _local_search()
    if ls is not None and max(ls) < best_val:
        best = ls
        best_val = max(ls)
        
    # 3. Random Local Search (multiple restarts)
    if time.time() < _DEADLINE:
        rls = _random_local_search()
        if rls is not None and max(rls) < best_val:
            best = rls
            best_val = max(rls)
            
    return best
# EVOLVE-BLOCK-END
