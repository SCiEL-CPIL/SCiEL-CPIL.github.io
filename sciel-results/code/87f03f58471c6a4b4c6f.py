"""Minimum maximal no-isosceles set on [0,N)^2."""
import random
import time
import itertools

N = 16

# Precompute all grid points
ALL_POINTS = list(itertools.product(range(N), repeat=2))
N_PTS = len(ALL_POINTS)

def _is_iso_fast(a, b, c):
    ab2 = (a[0]-b[0])**2 + (a[1]-b[1])**2
    ac2 = (a[0]-c[0])**2 + (a[1]-c[1])**2
    bc2 = (b[0]-c[0])**2 + (b[1]-c[1])**2
    return ab2 == ac2 or ab2 == bc2 or ac2 == bc2

def _can_add(p, chosen):
    for i in range(len(chosen)):
        for j in range(i+1, len(chosen)):
            if _is_iso_fast(p, chosen[i], chosen[j]):
                return False
    return True

def _is_maximal(chosen):
    chosen_set = set(chosen)
    for p in ALL_POINTS:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _greedy_random(order=None):
    if order is None:
        order = list(ALL_POINTS)
        random.shuffle(order)
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen

def _try_remove(current, deadline):
    """Try to remove points from current to get a smaller maximal set."""
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        for i in range(len(current)):
            if time.time() > deadline:
                return current
            test = current[:i] + current[i+1:]
            if _is_maximal(test):
                current = test
                improved = True
                break
    return current

def _local_search(current, deadline, max_iters=50):
    """Local search: try to swap points to find smaller maximal sets."""
    best = list(current)
    best_size = len(best)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        
        # Try removing each point
        for i in range(len(best)):
            if time.time() > deadline:
                return best
            test = best[:i] + best[i+1:]
            if _is_maximal(test):
                best = test
                best_size = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # If can't remove, try swaps: remove point i, add point q
        for i in range(len(best)):
            if time.time() > deadline:
                return best
            base = best[:i] + best[i+1:]
            base_set = set(base)
            for q in ALL_POINTS:
                if q in base_set:
                    continue
                if _can_add(q, base):
                    test = base + [q]
                    if _is_maximal(test):
                        # Same size, but maybe different structure allows further reduction
                        best = test
                        best_size = len(best)
                        improved = True
                        break
            if improved:
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    best = None
    best_size = float('inf')
    
    # Multiple random greedy restarts with improvement
    n_restarts = 0
    while time.time() < deadline - 30.0:
        candidate = _greedy_random()
        # Improve by removals
        improved = _try_remove(candidate, deadline - 10.0)
        if len(improved) < best_size:
            best_size = len(improved)
            best = improved
            # Try local search on the best
            if time.time() < deadline - 20.0:
                ls_result = _local_search(best, deadline - 10.0, max_iters=20)
                if len(ls_result) < best_size:
                    best_size = len(ls_result)
                    best = ls_result
        n_restarts += 1
    
    # Final improvement pass
    if best is not None:
        final = _try_remove(best, deadline)
        if len(final) < best_size:
            best = final
            best_size = len(final)
    
    if best is None:
        best = _greedy_random()
    
    return best
