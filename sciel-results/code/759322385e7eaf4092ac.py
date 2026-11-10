"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools
from collections import defaultdict


def _precompute_conflicts():
    """Precompute for each point, the set of pairs (i,j) that form an isosceles triangle with it."""
    all_points = list(itertools.product(range(N), repeat=2))
    M = N * N
    point_to_idx = {p: i for i, p in enumerate(all_points)}
    
    # conflicts[p] = list of (i, j) such that p, all_points[i], all_points[j] form isosceles triangle
    # where i < j and i, j != p
    conflicts = [[] for _ in range(M)]
    
    for p_idx, p in enumerate(all_points):
        for i in range(M):
            if i == p_idx:
                continue
            for j in range(i + 1, M):
                if j == p_idx:
                    continue
                a, b, c = p, all_points[i], all_points[j]
                d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
                d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    conflicts[p_idx].append((i, j))
    
    return all_points, conflicts


def _greedy_random(all_points, conflicts, rng):
    """Greedy construction with random point ordering."""
    M = len(all_points)
    order = list(range(M))
    rng.shuffle(order)
    chosen = []
    chosen_set = set()
    
    for p_idx in order:
        # Check if p_idx can be added
        can_add = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                # Check if p_idx, chosen[i], chosen[j] form isosceles
                a, b, c = p_idx, chosen[i], chosen[j]
                # We need to check if (a,b) or (a,c) or (b,c) are in conflicts
                # Actually, let's just compute directly for correctness
                pa = all_points[a]
                pb = all_points[b]
                pc = all_points[c]
                d1 = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                d2 = (pa[0] - pc[0]) ** 2 + (pa[1] - pc[1]) ** 2
                d3 = (pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    
    return chosen


def _is_maximal(chosen, all_points, conflicts):
    """Check if chosen is a maximal no-isosceles set."""
    chosen_set = set(chosen)
    M = len(all_points)
    
    for p_idx in range(M):
        if p_idx in chosen_set:
            continue
        # Check if p_idx can be added
        can_add = True
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                pa = all_points[p_idx]
                pb = all_points[chosen[i]]
                pc = all_points[chosen[j]]
                d1 = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                d2 = (pa[0] - pc[0]) ** 2 + (pa[1] - pc[1]) ** 2
                d3 = (pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            return False
    return True


def _local_search_reduce(chosen, all_points, conflicts, max_iterations=100):
    """Try to remove points from the set while maintaining maximality."""
    M = len(all_points)
    chosen_set = set(chosen)
    
    for _ in range(max_iterations):
        improved = False
        for v in list(chosen):
            # Try removing v
            new_chosen = [x for x in chosen if x != v]
            new_chosen_set = chosen_set - {v}
            
            # Check if removing v still maintains maximality
            # i.e., every point not in new_chosen_set is blocked by some pair in new_chosen_set
            maximal = True
            for p_idx in range(M):
                if p_idx in new_chosen_set:
                    continue
                can_add = True
                for i in range(len(new_chosen)):
                    for j in range(i + 1, len(new_chosen)):
                        pa = all_points[p_idx]
                        pb = all_points[new_chosen[i]]
                        pc = all_points[new_chosen[j]]
                        d1 = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                        d2 = (pa[0] - pc[0]) ** 2 + (pa[1] - pc[1]) ** 2
                        d3 = (pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2
                        if d1 == d2 or d1 == d3 or d2 == d3:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    maximal = False
                    break
            
            if maximal:
                chosen = new_chosen
                chosen_set = new_chosen_set
                improved = True
                break
        
        if not improved:
            break
    
    return chosen


def construct():
    all_points, conflicts = _precompute_conflicts()
    M = len(all_points)
    
    best = None
    best_size = float('inf')
    
    # Run multiple randomized trials with local search
    for trial in range(200):
        rng = random.Random(trial)
        chosen = _greedy_random(all_points, conflicts, rng)
        
        # Apply local search to reduce size
        chosen_reduced = _local_search_reduce(chosen, all_points, conflicts)
        
        # Verify maximality
        if not _is_maximal(chosen_reduced, all_points, conflicts):
            continue
        
        if len(chosen_reduced) < best_size:
            best_size = len(chosen_reduced)
            best = chosen_reduced
    
    # Also try deterministic greedy as fallback
    chosen_det = []
    for p_idx in range(M):
        can_add = True
        for i in range(len(chosen_det)):
            for j in range(i + 1, len(chosen_det)):
                pa = all_points[p_idx]
                pb = all_points[chosen_det[i]]
                pc = all_points[chosen_det[j]]
                d1 = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                d2 = (pa[0] - pc[0]) ** 2 + (pa[1] - pc[1]) ** 2
                d3 = (pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2
                if d1 == d2 or d1 == d3 or d2 == d3:
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            chosen_det.append(p_idx)
    
    # Reduce deterministic solution too
    chosen_det_reduced = _local_search_reduce(chosen_det, all_points, conflicts)
    
    if len(chosen_det_reduced) < best_size:
        best = chosen_det_reduced
        best_size = len(chosen_det_reduced)
    
    if best is None:
        best = chosen_det_reduced
    
    # Convert indices back to (x, y) tuples
    return [all_points[i] for i in best]
# EVOLVE-BLOCK-END
