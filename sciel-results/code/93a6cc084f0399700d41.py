"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def _max_sidon_size(subset):
    """Compute the largest subset of 'subset' with all pair sums (i<=j) distinct."""
    n = len(subset)
    if n <= 1:
        return n
    elems = sorted(subset)
    best = [0]
    
    def backtrack(idx, current, current_sums):
        if idx >= n:
            best[0] = max(best[0], len(current))
            return
        # Pruning: if even taking all remaining doesn't beat best, stop
        if len(current) + (n - idx) <= best[0]:
            return
        
        # Option 1: Skip idx
        backtrack(idx + 1, current, current_sums)
        
        # Option 2: Take idx
        elem = elems[idx]
        new_sums = set(current_sums)
        valid = True
        # Check collisions with existing elements in current
        for j in range(len(current)):
            val = current[j] + elem
            if val in new_sums:
                valid = False
                break
            new_sums.add(val)
        
        if valid:
            val = elem + elem
            if val in new_sums:
                valid = False
            else:
                new_sums.add(val)
        
        if valid:
            backtrack(idx + 1, current + [elem], new_sums)
    
    backtrack(0, [], set())
    return best[0]

def _valid(cand):
    """Check candidate is 12 distinct integers in [0,128]."""
    if len(cand) != 12:
        return False
    if len(set(cand)) != 12:
        return False
    return all(0 <= x <= 128 for x in cand)

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin
    
    best_cand = None
    best_score = 99
    seen = set()
    
    def consider(cand):
        nonlocal best_cand, best_score
        if not _valid(cand):
            return
        key = tuple(sorted(cand))
        if key in seen:
            return
        seen.add(key)
        s = _max_sidon_size(cand)
        if s < best_score:
            best_score = s
            best_cand = list(cand)
    
    # Initial candidates: Dense intervals and APs
    # Hypothesis: Dense sets have many collisions, thus smaller max Sidon subsets
    candidates = [
        list(range(12)),
        list(range(1, 13)),
        list(range(2, 14)),
        list(range(0, 24, 2)),
        list(range(0, 24, 3)),
        list(range(0, 24, 4)),
        list(range(0, 36, 3)),
        [0, 1, 2, 4, 8, 16, 32, 64, 100, 110, 120, 128], # Mixed
    ]
    
    for cand in candidates:
        if time.time() > deadline:
            break
        consider(cand)
        
    if best_cand is None:
        best_cand = list(range(12))
        best_score = _max_sidon_size(best_cand)

    # Local Search
    rng = random.Random(42)
    
    # We want to minimize best_score.
    # Strategy: Random perturbations.
    # Since the landscape is discrete and small, hill climbing works well.
    
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        cand = list(best_cand)
        
        # Perturb 1-3 elements
        num_perturb = rng.randint(1, 3)
        for _ in range(num_perturb):
            idx = rng.randint(0, 11)
            cand[idx] = rng.randint(0, 128)
            
        if not _valid(cand):
            continue
            
        key = tuple(sorted(cand))
        if key in seen:
            continue
        seen.add(key)
        
        s = _max_sidon_size(cand)
        if s < best_score:
            best_score = s
            best_cand = cand
            
        # Occasional restart from a random point to escape local minima
        if iterations % 500 == 0:
             random_cand = rng.sample(range(129), 12)
             consider(random_cand)

    return best_cand
# EVOLVE-BLOCK-END
