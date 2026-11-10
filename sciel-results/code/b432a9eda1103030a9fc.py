"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _check_dissociated_fast(elements):
    """Fast check using floats."""
    if not elements:
        return True
    if len(elements) > 20:
        return False
    sums = {0.0}
    for s in elements:
        inv = 1.0 / s
        new_sums = set()
        for v in sums:
            nv = v + inv
            if nv in sums or nv in new_sums:
                return False
            new_sums.add(nv)
        sums.update(new_sums)
    return True

def _check_dissociated_exact(elements):
    """Exact check using Fractions."""
    if not elements:
        return True
    if len(elements) > 17:
        return False
    sums = {Fraction(0, 1)}
    for s in elements:
        inv = Fraction(1, s)
        new_sums = set()
        for v in sums:
            nv = v + inv
            if nv in sums or nv in new_sums:
                return False
            new_sums.add(nv)
        sums.update(new_sums)
    return True

def _greedy_random(pool, time_limit):
    """Greedy construction with random ordering."""
    start = time.time()
    current = []
    remaining = list(pool)
    random.shuffle(remaining)
    
    while remaining and (time.time() - start) < time_limit:
        progress = False
        # Try each remaining element
        for i, elem in enumerate(remaining):
            if (time.time() - start) >= time_limit:
                break
            candidate = current + [elem]
            if _check_dissociated_fast(candidate):
                current = candidate
                remaining.pop(i)
                progress = True
                break
        if not progress:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best)
    all_elements = list(range(1, 23))
    
    # Phase 1: Random restart greedy
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        per_restart_time = max(0.1, (deadline - time.time() - 30) / max(1, num_restarts - r))
        candidate = _greedy_random(all_elements, per_restart_time)
        if len(candidate) > best_size:
            if _check_dissociated_exact(candidate):
                best = candidate
                best_size = len(best)
    
    # Phase 2: Local search on best
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        
        improved = False
        
        # Try adding one
        for elem in all_elements:
            if time.time() > deadline - 5:
                break
            if elem in best:
                continue
            candidate = best + [elem]
            if _check_dissociated_fast(candidate) and _check_dissociated_exact(candidate):
                best = candidate
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swap
        for rem in list(best):
            if time.time() > deadline - 5:
                break
            for elem in all_elements:
                if time.time() > deadline - 5:
                    break
                if elem in best:
                    continue
                candidate = [x for x in best if x != rem] + [elem]
                if _check_dissociated_fast(candidate) and _check_dissociated_exact(candidate):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best
# EVOLVE-BLOCK-END
