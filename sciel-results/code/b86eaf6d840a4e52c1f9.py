import itertools
import time
import random
import sys
from collections import defaultdict

N = 16
_DEADLINE = time.time() + 280

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _precompute(all_points):
    n = len(all_points)
    # dist_matrix[i][j] = squared distance
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _d2(all_points[i], all_points[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    # For each point i, and each distance d, list of points j such that dist(i, j) == d
    # dist_lists[i] = {d: [j1, j2, ...]}
    dist_lists = [dict() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist_matrix[i][j]
            if d not in dist_lists[i]:
                dist_lists[i][d] = []
            dist_lists[i][d].append(j)
    
    return dist_matrix, dist_lists

def _can_add(x, chosen_set, chosen_list, dist_matrix, dist_lists):
    """
    Check if point x can be added to the current set.
    chosen_set: set of indices in chosen
    chosen_list: list of indices in chosen
    """
    if not chosen_list:
        return True
        
    # 1. Distances from x to all p in chosen must be distinct
    seen_dists = set()
    for p in chosen_list:
        d = dist_matrix[x][p]
        if d in seen_dists:
            return False
        seen_dists.add(d)
        
    # 2. For each p in chosen, d(x, p) must not be a distance already used by p to other points in chosen
    for p in chosen_list:
        d_xp = dist_matrix[x][p]
        # Check if any other q in chosen has dist_matrix[p][q] == d_xp
        # We can check if p has any other point at distance d_xp in the chosen set
        # dist_lists[p][d_xp] contains all points at distance d_xp from p.
        # We need to see if any of them are in chosen_set (excluding p itself, but p is not in dist_lists[p])
        # Wait, dist_lists[p] is a dict of d -> list of points.
        # So we check if any point in dist_lists[p][d_xp] is in chosen_set.
        # Note: x is not in chosen_set yet.
        if d_xp in dist_lists[p]:
            for q in dist_lists[p][d_xp]:
                if q in chosen_set:
                    return False
    return True

def _is_maximal(chosen_set, chosen_list, all_indices, dist_matrix, dist_lists):
    """
    Check if chosen is maximal.
    i.e., no point in all_indices \ chosen can be added.
    """
    for x in all_indices:
        if x in chosen_set:
            continue
        if _can_add(x, chosen_set, chosen_list, dist_matrix, dist_lists):
            return False
    return True

def _construct_greedy(all_indices, dist_matrix, dist_lists, rng):
    order = list(all_indices)
    rng.shuffle(order)
    chosen = []
    chosen_set = set()
    for p in order:
        if _can_add(p, chosen_set, chosen, dist_matrix, dist_lists):
            chosen.append(p)
            chosen_set.add(p)
    return chosen, chosen_set

def _local_search_reduce(chosen, chosen_set, all_indices, dist_matrix, dist_lists, rng, deadline):
    """
    Try to reduce the size of the maximal set.
    Move: Remove p, q and add r.
    If (S \ {p, q}) U {r} is maximal, accept.
    """
    s = list(chosen)
    s_set = set(s)
    n = len(s)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs (p, q) to remove
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                p, q = s[i], s[j]
                # Create base set
                base = [x for k, x in enumerate(s) if k != i and k != j]
                base_set = set(base)
                
                # Try to add a point r not in base
                # We can restrict r to points that were "blocked" by p or q?
                # For now, try all points not in base.
                for r in all_indices:
                    if r in base_set:
                        continue
                    
                    # Check validity (no-isosceles)
                    if not _can_add(r, base_set, base, dist_matrix, dist_lists):
                        continue
                    
                    # Check maximality
                    candidate_set = base_set | {r}
                    candidate_list = base + [r]
                    if _is_maximal(candidate_set, candidate_list, all_indices, dist_matrix, dist_lists):
                        s = candidate_list
                        s_set = candidate_set
                        n = len(s)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
                
    return s, s_set

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    all_indices = list(range(len(all_points)))
    dist_matrix, dist_lists = _precompute(all_points)
    
    best = None
    best_size = float('inf')
    rng = random.Random(42)
    
    # Run multiple restarts
    while time.time() < _DEADLINE:
        # 1. Greedy Construction
        chosen, chosen_set = _construct_greedy(all_indices, dist_matrix, dist_lists, rng)
        
        # 2. Local Search: Try to reduce size
        pruned, pruned_set = _local_search_reduce(chosen, chosen_set, all_indices, dist_matrix, dist_lists, rng, _DEADLINE)
        
        # 3. If size didn't change, we could try swaps, but reduce is the main driver for minimum maximal.
        # If we can't reduce, we might be stuck in a local minimum of size k.
        # Swapping points of same size doesn't help minimize size.
        # So we just restart.
        
        # 4. Update best
        if len(pruned) < best_size:
            best_size = len(pruned)
            best = pruned
            
    if best is None:
        best, _ = _construct_greedy(all_indices, dist_matrix, dist_lists, rng)
        
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END
