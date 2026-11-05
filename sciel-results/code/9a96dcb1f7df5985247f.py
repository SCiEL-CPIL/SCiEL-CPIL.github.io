"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import defaultdict


def _precompute_conflicts():
    """Precompute all pairs (i, j) that form an isosceles triangle with any third point k."""
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # For each pair (i, j), find all k such that {i, j, k} forms an isosceles triangle
    # Store as: conflicts[k] = set of (i, j) pairs that conflict with k
    # Actually, we want: for a set S, point k can be added iff no pair (i,j) in S conflicts with k
    
    # Better approach: precompute for each point k, which pairs (i, j) would create isosceles with k
    # conflict_with[k] = set of (i, j) tuples where i < j and {i,j,k} is isosceles
    
    conflict_with = [set() for _ in range(n_points)]
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            pi, pj = all_points[i], all_points[j]
            d_ij = (pi[0] - pj[0])**2 + (pi[1] - pj[1])**2
            
            for k in range(n_points):
                if k == i or k == j:
                    continue
                pk = all_points[k]
                d_ik = (pi[0] - pk[0])**2 + (pi[1] - pk[1])**2
                d_jk = (pj[0] - pk[0])**2 + (pj[1] - pk[1])**2
                
                if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                    conflict_with[k].add((i, j))
    
    return all_points, point_to_idx, conflict_with


def _can_add_fast(k, chosen_set, conflict_with):
    """Check if point k can be added to chosen_set using precomputed conflicts."""
    for pair in conflict_with[k]:
        if pair[0] in chosen_set and pair[1] in chosen_set:
            return False
    return True


def _is_maximal_fast(chosen_set, conflict_with):
    """Check if chosen_set is maximal using precomputed conflicts."""
    all_points = list(itertools.product(range(N), repeat=2))
    for p in all_points:
        # Find index of p
        idx = None
        for i, pp in enumerate(all_points):
            if pp == p:
                idx = i
                break
        if idx is None:
            continue
        if idx not in chosen_set:
            if _can_add_fast(idx, chosen_set, conflict_with):
                return False
    return True


def _make_maximal_fast(chosen_set, conflict_with):
    """Greedily expand chosen_set to be maximal."""
    chosen_set = set(chosen_set)
    all_points = list(itertools.product(range(N), repeat=2))
    
    # Sort by some heuristic - maybe by degree (number of conflicts)
    # Points with fewer conflicts are "safer" to add
    degrees = []
    for i in range(len(all_points)):
        if i not in chosen_set:
            # Count how many other unchosen points would conflict
            deg = 0
            for j in range(len(all_points)):
                if j not in chosen_set and j != i:
                    # Check if i and j would conflict with some common point
                    # Actually, just count conflicts
                    pass
            degrees.append((i, len(conflict_with[i])))
    
    # Try adding points in order of increasing conflict count
    degrees.sort(key=lambda x: x[1])
    
    for i, _ in degrees:
        if i not in chosen_set:
            if _can_add_fast(i, chosen_set, conflict_with):
                chosen_set.add(i)
    
    return chosen_set


def _greedy_random_order(all_points, conflict_with, random_order):
    """Greedy construction with a given random order."""
    chosen_set = set()
    for i in random_order:
        if _can_add_fast(i, chosen_set, conflict_with):
            chosen_set.add(i)
    return chosen_set


def _local_search_min_maximal(chosen_set, conflict_with, deadline):
    """Local search to minimize the size of a maximal set."""
    current = set(chosen_set)
    best = set(current)
    best_size = len(current)
    
    all_points = list(itertools.product(range(N), repeat=2))
    n_points = len(all_points)
    
    while time.time() < deadline:
        # Try removing each point and see if we can still make it maximal
        improved = False
        for i in list(current):
            if time.time() > deadline:
                break
            
            new_set = current - {i}
            # Try to make it maximal
            extended = _make_maximal_fast(new_set, conflict_with)
            
            if len(extended) < len(current):
                current = extended
                improved = True
                if len(current) < best_size:
                    best = set(current)
                    best_size = len(current)
                break  # Restart the loop
        
        if not improved:
            # Try swaps: remove one point, add another
            for i in list(current):
                if time.time() > deadline:
                    break
                
                new_set = current - {i}
                # Try adding each point not in new_set
                for j in range(n_points):
                    if j in new_set:
                        continue
                    if _can_add_fast(j, new_set, conflict_with):
                        candidate = new_set | {j}
                        extended = _make_maximal_fast(candidate, conflict_with)
                        if len(extended) < len(current):
                            current = extended
                            if len(current) < best_size:
                                best = set(current)
                                best_size = len(current)
                            break
                if time.time() > deadline:
                    break
    
    return best


def construct():
    deadline = time.time() + 280  # Safety margin before 300s limit
    
    # Precompute conflicts
    all_points, point_to_idx, conflict_with = _precompute_conflicts()
    n_points = len(all_points)
    
    best_set = None
    best_size = float('inf')
    
    # Initial solution: greedy with multiple random orders
    for restart in range(20):
        if time.time() > deadline:
            break
        
        random_order = list(range(n_points))
        random.shuffle(random_order)
        
        chosen = _greedy_random_order(all_points, conflict_with, random_order)
        # Make it maximal
        chosen = _make_maximal_fast(chosen, conflict_with)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = set(chosen)
        
        # Local search to minimize
        if time.time() < deadline:
            improved = _local_search_min_maximal(chosen, conflict_with, deadline)
            if len(improved) < best_size:
                best_size = len(improved)
                best_set = set(improved)
    
    # Convert back to list of tuples
    result = [all_points[i] for i in best_set]
    return result
# EVOLVE-BLOCK-END
