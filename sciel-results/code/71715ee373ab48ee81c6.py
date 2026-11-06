"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    A = list(A)
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):  # only need i<=j since addition/multiplication are commutative
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    # Note: for i=j, we get 2*ai and ai*ai, which are included above
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def _generate_geometric_candidates(rng, time_limit):
    """Generate various geometric-like candidate sets."""
    candidates = []
    
    # Powers of 2
    p2 = [2**i for i in range(20)]
    if max(p2) <= 10000:
        candidates.append(p2)
    
    # Various geometric sequences
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        cand = []
        val = 1
        for i in range(20):
            cand.append(val)
            val *= base
            if val > 10000:
                # Fill remaining with large numbers
                while len(cand) < 20:
                    new_val = min(10000, val + rng.randint(1, 50))
                    if new_val not in cand:
                        cand.append(new_val)
                    else:
                        new_val += 1
                        if new_val <= 10000:
                            cand.append(new_val)
        if len(set(cand)) == 20 and all(1 <= x <= 10000 for x in cand):
            candidates.append(cand)
    
    # Random geometric with varying ratios
    for _ in range(30):
        ratio = rng.uniform(1.1, 3.0)
        start = rng.randint(1, 20)
        cand = []
        val = float(start)
        for i in range(20):
            iv = int(round(val))
            if iv < 1:
                iv = 1
            if iv > 10000:
                iv = 10000
            # Find next unique value
            while iv in cand:
                iv += 1
                if iv > 10000:
                    iv = 10000
                    break
            if iv not in cand:
                cand.append(iv)
            val *= ratio
        if len(cand) == 20 and all(1 <= x <= 10000 for x in cand):
            candidates.append(cand)
    
    # Mix: some small consecutive, then geometric
    for _ in range(20):
        small_count = rng.randint(3, 8)
        start_small = rng.randint(1, 5)
        cand = list(range(start_small, start_small + small_count))
        val = start_small + small_count
        for i in range(20 - small_count):
            iv = int(val)
            if iv < 1:
                iv = 1
            if iv > 10000:
                iv = 10000
            while iv in cand:
                iv += 1
                if iv > 10000:
                    iv = 10000
                    break
            if iv not in cand:
                cand.append(iv)
            val *= rng.uniform(1.5, 3.0)
        if len(cand) == 20 and all(1 <= x <= 10000 for x in cand):
            candidates.append(cand)
    
    # Random sets
    for _ in range(10):
        cand = list(rng.sample(range(1, 10001), 20))
        candidates.append(cand)
    
    return candidates

def _local_search(A, time_limit, rng, best_score_global):
    """Improve A via local search with simulated annealing."""
    A = list(A)
    current_score = _score(A)
    best_A = list(A)
    best_score = current_score
    deadline = time.time() + time_limit
    
    temp = 5.0  # Initial temperature
    
    while time.time() < deadline:
        # Randomly pick an element to perturb
        idx = rng.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Try several candidate replacements
        for _ in range(30):
            if A[idx] > 10:
                new_val = max(1, A[idx] + rng.randint(-30, 30))
            else:
                new_val = rng.randint(1, 200)
            
            if new_val == old_val:
                continue
            if new_val > 10000:
                continue
            if new_val in A and new_val != old_val:
                continue
            
            A[idx] = new_val
            s = _score(A)
            
            if s < best_score:
                best_score = s
                best_A = list(A)
            
            # Accept with probability if not worse
            if s > current_score:
                delta = s - current_score
                prob = np.exp(-delta / max(temp, 0.1))
                if rng.random() < prob:
                    current_score = s
                else:
                    A[idx] = old_val
            else:
                current_score = s
        
        A[idx] = old_val
        
        # Cool down
        temp *= 0.95
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    rng = random.Random(42)
    
    # Start with generated candidates
    candidates = _generate_geometric_candidates(rng, deadline - start_time)
    
    best_A = None
    best_score = float('inf')
    
    # Evaluate all candidates first
    for cand in candidates:
        if time.time() > deadline:
            break
        
        if not _is_valid(cand, 20, 10000):
            continue
        
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_A = list(cand)
    
    # Sort candidates by score and do local search on the best ones
    scored_candidates = []
    for cand in candidates:
        if not _is_valid(cand, 20, 10000):
            continue
        s = _score(cand)
        scored_candidates.append((s, cand))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # Do local search on top candidates
    for s, cand in scored_candidates[:5]:
        if time.time() > deadline - 5:
            break
        
        improved_A, improved_score = _local_search(cand, min(10, deadline - time.time() - 3), rng, best_score)
        if improved_score < best_score:
            best_score = improved_score
            best_A = improved_A
    
    if best_A is None:
        return list(range(1, 21))
    
    return best_A

# EVOLVE-BLOCK-END
