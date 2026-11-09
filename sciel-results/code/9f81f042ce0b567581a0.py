"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

DEADLINE = 325  # seconds, safety margin before 330s kill

def _d2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(p, chosen):
    """Check if p can be added to chosen without creating an isosceles triangle."""
    if not chosen:
        return True
    # Vectorized check using numpy
    chosen_arr = np.array(chosen)
    dx = chosen_arr[:, 0] - p[0]
    dy = chosen_arr[:, 1] - p[1]
    dists_from_p = dx * dx + dy * dy
    
    # For each pair (i, j) in chosen, check if dist(p,i) == dist(p,j) or 
    # dist(p,i) == dist(i,j) or dist(p,j) == dist(i,j)
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            d1 = dists_from_p[i]
            d2 = dists_from_p[j]
            d3 = _d2(chosen[i], chosen[j])
            if d1 == d2 or d1 == d3 or d2 == d3:
                return False
    return True

def _is_maximal(chosen, all_points):
    """Check if no point outside chosen can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _greedy_random(all_points, rng):
    """Randomized greedy: pick points in random order, add if possible."""
    order = list(all_points)
    rng.shuffle(order)
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen

def _try_remove(chosen, all_points, rng, max_iters=200):
    """Try to remove points while maintaining maximality."""
    chosen = list(chosen)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        # Try removing each point in random order
        indices = list(range(len(chosen)))
        rng.shuffle(indices)
        for idx in indices:
            candidate = chosen[:idx] + chosen[idx+1:]
            if _is_maximal(candidate, all_points):
                chosen = candidate
                improved = True
                break
    return chosen

def _swap_search(chosen, all_points, rng, max_iters=50):
    """Try to swap points to reduce size while maintaining maximality."""
    chosen_set = set(chosen)
    chosen = list(chosen)
    remaining = [p for p in all_points if p not in chosen_set]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if not remaining:
            break
        
        # Try swapping: remove one from chosen, add one from remaining
        chosen_indices = list(range(len(chosen)))
        rng.shuffle(chosen_indices)
        
        for ci in chosen_indices:
            if time.time() - _swap_search.start_time > DEADLINE * 0.8:
                return chosen
            
            # Try removing chosen[ci]
            without_ci = chosen[:ci] + chosen[ci+1:]
            without_ci_set = set(without_ci)
            
            # Find points that can be added to without_ci
            addable = []
            for p in remaining:
                if p in without_ci_set:
                    continue
                if _can_add(p, without_ci):
                    addable.append(p)
            
            if not addable:
                continue
            
            # Try adding each addable point
            rng.shuffle(addable)
            for p in addable[:20]:  # Limit to keep it fast
                candidate = without_ci + [p]
                if _is_maximal(candidate, all_points):
                    # Check if this actually reduces something (it won't directly, 
                    # but might enable further removals)
                    # For now, just accept if maximal
                    chosen = candidate
                    chosen_set = set(chosen)
                    remaining = [p for p in all_points if p not in chosen_set]
                    improved = True
                    break
            
            if improved:
                break
    
    return chosen

def construct():
    start_time = time.time()
    _swap_search.start_time = start_time
    all_points = list(itertools.product(range(N), repeat=2))
    rng = random.Random(42)
    
    best = None
    best_size = N * N + 1
    
    # Phase 1: Multiple randomized greedy starts
    while time.time() - start_time < DEADLINE * 0.3:
        candidate = _greedy_random(all_points, rng)
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
    
    # Phase 2: Local search (removal + swap) on best and near-best candidates
    if best is not None:
        while time.time() - start_time < DEADLINE * 0.85:
            # Try removal on current best
            improved = _try_remove(best, all_points, rng, max_iters=100)
            if len(improved) < best_size:
                best = improved
                best_size = len(best)
            
            # Try swap search
            swapped = _swap_search(best, all_points, rng, max_iters=30)
            if len(swapped) < best_size:
                best = swapped
                best_size = len(best)
            
            # Occasionally try a fresh random start + removal
            if time.time() - start_time < DEADLINE * 0.6:
                candidate = _greedy_random(all_points, rng)
                candidate = _try_remove(candidate, all_points, rng, max_iters=50)
                if len(candidate) < best_size:
                    best = candidate
                    best_size = len(candidate)
    
    # Final: ensure maximality of best result
    if best is not None and time.time() - start_time < DEADLINE - 5:
        # If somehow not maximal (shouldn't happen), fix by adding points
        if not _is_maximal(best, all_points):
            # Add points in random order until maximal
            remaining = [p for p in all_points if p not in set(best)]
            rng.shuffle(remaining)
            for p in remaining:
                if _can_add(p, best):
                    best.append(p)
                if _is_maximal(best, all_points):
                    break
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
