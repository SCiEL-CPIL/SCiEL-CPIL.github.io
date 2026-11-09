import itertools
import time
import random
import sys
import numpy as np

N = 16
_DEADLINE = time.time() + 275  # Safety margin before 330s kill

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute(all_points):
    """Precompute distance matrix and for each point, the set of distances to all other points."""
    n = len(all_points)
    dist = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist[i, j] = d
            dist[j, i] = d
    
    # dist_sets[i] = set of distances from point i to all other points
    dist_sets = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_sets[i].add(int(dist[i, j]))
    
    return dist, dist_sets

def _can_add(x, chosen_set, dist, dist_sets):
    """
    Check if point x can be added to chosen_set without creating an isosceles triangle.
    Two conditions must hold:
    1. All distances from x to points in chosen_set must be distinct.
    2. For each p in chosen_set, d(x,p) must not equal d(p,q) for any q in chosen_set, q != p.
       (This prevents x,p,q from forming an isosceles triangle with apex at p.)
    """
    if not chosen_set:
        return True
    
    seen_distances = set()
    for p in chosen_set:
        d_xp = int(dist[x, p])
        if d_xp in seen_distances:
            return False
        seen_distances.add(d_xp)
    
    # Check condition 2: for each p, d(x,p) must not be in distances from p to other chosen points
    for p in chosen_set:
        d_xp = int(dist[x, p])
        # If d_xp is in dist_sets[p], it means there exists some q (possibly in chosen_set)
        # such that d(p,q) == d_xp. We need to check if that q is in chosen_set and q != p.
        # Actually, if d(x,p) == d(p,q) for q in chosen_set, q != p, then (x,p,q) is isosceles.
        # We need to verify that no such q exists in chosen_set.
        # A quick check: if d_xp is in dist_sets[p], there might be such a q.
        # We need to check if any q in chosen_set (q != p) has d(p,q) == d_xp.
        for q in chosen_set:
            if q == p:
                continue
            if int(dist[p, q]) == d_xp:
                return False
    
    return True

def _is_maximal(chosen_set, all_indices, dist, dist_sets):
    """Check if no point can be added to chosen_set."""
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add(x, chosen_set, dist, dist_sets):
            return False
    return True

def _greedy(all_indices, dist, dist_sets, rng):
    """Greedy construction with random ordering."""
    order = list(all_indices)
    rng.shuffle(order)
    chosen_set = set()
    for p in order:
        if _can_add(p, chosen_set, dist, dist_sets):
            chosen_set.add(p)
    return chosen_set

def _local_search_reduce(chosen_set, all_indices, dist, dist_sets, rng, deadline):
    """
    Try to reduce set size by removing 2 points and adding 1.
    This can reduce the set size by 1 if successful.
    """
    s = set(chosen_set)
    n = len(s)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        s_list = list(s)
        rng.shuffle(s_list)
        
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if time.time() > deadline:
                    break
                
                p, q = s_list[i], s_list[j]
                base = s - {p, q}
                
                # Try to find a point r that can be added to base
                for r in all_indices:
                    if r in base:
                        continue
                    
                    if not _can_add(r, base, dist, dist_sets):
                        continue
                    
                    candidate = base | {r}
                    if _is_maximal(candidate, all_indices, dist, dist_sets):
                        s = candidate
                        n = len(s)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return s

def _simulated_annealing(chosen_set, all_indices, dist, dist_sets, rng, deadline):
    """
    Simulated annealing: try to swap or remove/add points.
    Accept worse solutions with some probability.
    """
    s = set(chosen_set)
    best_s = set(s)
    best_size = len(s)
    current_size = len(s)
    temperature = 10.0
    cooling = 0.99
    
    while time.time() < deadline and temperature > 0.01:
        # Generate a neighbor
        move_type = rng.random()
        
        if move_type < 0.3:
            # Swap: remove one point, add another
            if not s:
                break
            p = rng.choice(list(s))
            base = s - {p}
            # Try random candidates
            found = False
            for _ in range(50):
                r = rng.choice(all_indices)
                if r in base:
                    continue
                if _can_add(r, base, dist, dist_sets):
                    candidate = base | {r}
                    if _is_maximal(candidate, all_indices, dist, dist_sets):
                        s = candidate
                        current_size = len(s)
                        if current_size < best_size:
                            best_size = current_size
                            best_s = set(s)
                        found = True
                        break
            if not found:
                # Try a random valid addition without removal (if size can grow temporarily)
                # Actually, let's try removing a point and not adding (size decreases by 1, but may not be maximal)
                # We'll only accept if maximal. For now, skip.
                pass
        
        elif move_type < 0.6:
            # Remove 2, add 1 (reduce by 1)
            if len(s) < 3:
                continue
            s_list = list(s)
            rng.shuffle(s_list)
            p, q = s_list[0], s_list[1]
            base = s - {p, q}
            found = False
            for _ in range(30):
                r = rng.choice(all_indices)
                if r in base:
                    continue
                if _can_add(r, base, dist, dist_sets):
                    candidate = base | {r}
                    if _is_maximal(candidate, all_indices, dist, dist_sets):
                        s = candidate
                        current_size = len(s)
                        if current_size < best_size:
                            best_size = current_size
                            best_s = set(s)
                        found = True
                        break
        
        else:
            # Add a point (increase size by 1, then maybe remove others)
            # This is less useful for minimization, but helps escape local optima
            found = False
            for _ in range(20):
                r = rng.choice(all_indices)
                if r in s:
                    continue
                if _can_add(r, s, dist, dist_sets):
                    candidate = s | {r}
                    # Accept with some probability
                    delta = 1.0  # size increase
                    if rng.random() < np.exp(-delta / temperature):
                        s = candidate
                        current_size = len(s)
                        found = True
                        break
        
        temperature *= cooling
    
    return best_s

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist, dist_sets = _precompute(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Phase 1: Multiple greedy constructions with different orderings
    while time.time() < _DEADLINE - 30:
        chosen = _greedy(all_indices, dist, dist_sets, rng)
        
        # Local search reduce
        pruned = _local_search_reduce(chosen, all_indices, dist, dist_sets, rng, _DEADLINE - 10)
        
        # Simulated annealing polish
        polished = _simulated_annealing(pruned, all_indices, dist, dist_sets, rng, _DEADLINE - 5)
        
        if len(polished) < best_size:
            best_size = len(polished)
            best = polished
    
    if best is None:
        best = _greedy(all_indices, dist, dist_sets, rng)
    
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END
