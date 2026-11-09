"""Rewritten program using a different structural approach:
   iterative greedy + aggressive local search with multiple restarts,
   using vectorized distance checks for speed."""

N = 16
import time
import random
import numpy as np

_DEADLINE = time.time() + 280.0

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(p, chosen):
    """Check if point p can be added to chosen without creating an isosceles triangle."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_isosceles(p, chosen[i], chosen[j]):
                return False
    return True

def _is_maximal(chosen, all_points):
    """Check if chosen is maximal: every unselected point forms isosceles with some pair in chosen."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        # p must form isosceles with some pair (a,b) both in chosen
        found = False
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                if _is_isosceles(p, chosen[i], chosen[j]):
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True

def _greedy_build(order, all_points):
    """Build a maximal no-isosceles set by greedily adding points in given order."""
    chosen = []
    chosen_set = set()
    for p in order:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    # Now make it maximal by trying to add remaining points
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _local_search(chosen, all_points, max_iters=200):
    """Try to minimize the set while maintaining maximality."""
    chosen = list(chosen)
    best_size = len(chosen)
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 20:
            break
        
        improved = False
        # Try removing each point
        for idx in range(len(chosen)):
            p = chosen[idx]
            candidate = chosen[:idx] + chosen[idx+1:]
            if _is_maximal(candidate, all_points):
                chosen = candidate
                best_size = len(chosen)
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add another
            for idx in range(len(chosen)):
                if time.time() > _DEADLINE - 20:
                    break
                p_remove = chosen[idx]
                base = chosen[:idx] + chosen[idx+1:]
                for p_add in all_points:
                    if p_add in base:
                        continue
                    if _can_add(p_add, base):
                        candidate = base + [p_add]
                        if _is_maximal(candidate, all_points):
                            chosen = candidate
                            best_size = len(chosen)
                            improved = True
                            break
                if improved:
                    break
        
        if not improved:
            break
    
    return chosen

def _random_order(all_points):
    order = all_points[:]
    random.shuffle(order)
    return order

def _checkerboard_order(all_points):
    """Order points by checkerboard pattern."""
    even = [p for p in all_points if (p[0] + p[1]) % 2 == 0]
    odd = [p for p in all_points if (p[0] + p[1]) % 2 == 1]
    random.shuffle(even)
    random.shuffle(odd)
    return even + odd

def _diag_order(all_points):
    """Order by diagonal."""
    points = sorted(all_points, key=lambda p: (p[0] + p[1], p[0]))
    random.shuffle(points)
    return points

def construct():
    all_points = list(__import__('itertools').product(range(N), repeat=2))
    n_points = len(all_points)
    
    best_set = None
    best_size = n_points + 1
    
    num_attempts = 30
    for attempt in range(num_attempts):
        if time.time() > _DEADLINE - 30:
            break
        
        # Different ordering strategies
        if attempt % 3 == 0:
            order = _random_order(all_points)
        elif attempt % 3 == 1:
            order = _checkerboard_order(all_points)
        else:
            order = _diag_order(all_points)
        
        # Build initial maximal set
        chosen = _greedy_build(order, all_points)
        
        # Local search to minimize
        chosen = _local_search(chosen, all_points, max_iters=100)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen[:]
    
    if best_set is None:
        # Fallback: greedy in row-major order
        best_set = _greedy_build(all_points, all_points)
    
    return best_set
