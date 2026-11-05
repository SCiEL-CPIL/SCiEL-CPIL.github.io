"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
import random
import time
import itertools

# EVOLVE-BLOCK-START

def _sq_dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def _build_iso_pairs(all_points):
    """For each point i, precompute the set of unordered pairs {j,k} (j<k) such that (i,j,k) form an isosceles triangle."""
    n = len(all_points)
    iso_pairs = [set() for _ in range(n)]
    for i in range(n):
        pi = all_points[i]
        for j in range(n):
            if j == i:
                continue
            pj = all_points[j]
            for k in range(j + 1, n):
                if k == i:
                    continue
                pk = all_points[k]
                d1 = _sq_dist(pi, pj)
                d2 = _sq_dist(pi, pk)
                d3 = _sq_dist(pj, pk)
                if d1 == d2 or d1 == d3 or d2 == d3:
                    iso_pairs[i].add((j, k))
    return iso_pairs

def _can_add(p_idx, chosen_set, iso_pairs):
    """Check if point p_idx can be added to chosen_set."""
    if len(chosen_set) < 2:
        return True
    for (a, b) in iso_pairs[p_idx]:
        if a in chosen_set and b in chosen_set:
            return False
    return True

def _is_maximal(chosen_set, n, iso_pairs):
    """Check if the set is maximal."""
    for i in range(n):
        if i not in chosen_set:
            if _can_add(i, chosen_set, iso_pairs):
                return False
    return True

def _greedy_build(order, n, iso_pairs):
    """Build a maximal set by greedily adding points in given order."""
    chosen = []
    chosen_set = set()
    for p_idx in order:
        if _can_add(p_idx, chosen_set, iso_pairs):
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    # Ensure maximality
    for p_idx in range(n):
        if p_idx not in chosen_set and _can_add(p_idx, chosen_set, iso_pairs):
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    return chosen

def _minimize_deletion(chosen, n, iso_pairs):
    """Repeatedly remove points if the set remains maximal after removal."""
    chosen = list(chosen)
    changed = True
    while changed:
        changed = False
        for i in range(len(chosen)):
            trial_set = set(chosen)
            trial_set.discard(chosen[i])
            if _is_maximal(trial_set, n, iso_pairs):
                chosen.remove(chosen[i])
                changed = True
                break
    return chosen

def _conflict_graph(all_points):
    """
    Build a conflict graph where an edge (i, j) exists if there exists some k
    such that points i, j, k form an isosceles triangle.
    Actually, for the no-isosceles constraint, we need: a set S is valid if for all triples
    in S, no isosceles triangle is formed.
    
    The standard reduction to Independent Set isn't direct because the constraint is on triples,
    not pairs. However, we can use a different approach:
    
    Let's stick to the local search but improve it with a stronger local search:
    Simulated Annealing or Tabu Search on the set membership.
    
    Alternatively, we can define a "conflict" between a point p and a pair {a, b} in S.
    This is a hypergraph independent set problem.
    
    Let's implement a simple local search that tries to swap points or remove points
    to reduce size while maintaining maximality.
    """
    pass

def _local_search_improve(chosen, n, iso_pairs, all_points, deadline):
    """
    Try to improve the set by:
    1. Removing points if maximality is preserved.
    2. Trying to replace a point with a "better" point (one that allows more removals later).
    3. Random restarts with perturbation.
    """
    best = list(chosen)
    best_size = len(best)
    
    # First, do aggressive deletion
    best = _minimize_deletion(best, n, iso_pairs)
    best_size = len(best)
    
    # Now try local moves
    for step in range(500):
        if time.time() > deadline - 10:
            break
        
        current = list(best)
        current_set = set(current)
        
        # Try removing a random point
        if len(current) > 1:
            idx = random.randint(0, len(current) - 1)
            trial_set = set(current)
            trial_set.discard(current[idx])
            if _is_maximal(trial_set, n, iso_pairs):
                current.remove(current[idx])
                if len(current) < best_size:
                    best = current
                    best_size = len(current)
        
        # Try swapping: remove one, add another
        if len(current) > 1:
            remove_idx = random.randint(0, len(current) - 1)
            remove_point = current[remove_idx]
            
            # Find a point not in current that can be added if we remove remove_point
            candidates = [i for i in range(n) if i not in current_set]
            random.shuffle(candidates)
            
            for add_point in candidates[:10]:  # Try a few candidates
                trial_set = set(current)
                trial_set.discard(remove_point)
                if _can_add(add_point, trial_set, iso_pairs):
                    trial_set.add(add_point)
                    # Check if this swap leads to a smaller set after deletion
                    trial_list = list(trial_set)
                    trial_list = _minimize_deletion(trial_list, n, iso_pairs)
                    if len(trial_list) < best_size:
                        best = trial_list
                        best_size = len(trial_list)
                        current = trial_list
                        current_set = set(current)
                        break
        
        # Occasionally, do a full deletion pass
        if step % 50 == 0:
            current = _minimize_deletion(current, n, iso_pairs)
            if len(current) < best_size:
                best = current
                best_size = len(current)
    
    return best, best_size

def construct():
    all_points = list(itertools.product(range(N), repeat=2))
    n = len(all_points)
    start = time.time()
    deadline = start + 285  # safety margin
    
    iso_pairs = _build_iso_pairs(all_points)
    
    random.seed(42)
    
    best = None
    best_size = n + 1
    
    # Strategy 1: Heuristic ordering - points with more constraints first
    constraint_counts = [len(iso_pairs[i]) for i in range(n)]
    order = list(range(n))
    order.sort(key=lambda i: -constraint_counts[i])
    chosen = _greedy_build(order, n, iso_pairs)
    chosen = _minimize_deletion(chosen, n, iso_pairs)
    best = chosen
    best_size = len(chosen)
    
    # Strategy 2: Random restarts with local search
    for attempt in range(100):
        if time.time() > deadline - 30:
            break
        
        order = list(range(n))
        random.shuffle(order)
        chosen = _greedy_build(order, n, iso_pairs)
        
        if len(chosen) <= best_size + 2:  # Only do local search if it's competitive
            chosen, size = _local_search_improve(chosen, n, iso_pairs, all_points, deadline)
            if size < best_size:
                best_size = size
                best = chosen
                if best_size <= 8:
                    break
    
    # Strategy 3: Additional deletion passes on the best
    if best is not None and time.time() < deadline - 15:
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            trial = list(best)
            random.shuffle(trial)
            trial = _minimize_deletion(trial, n, iso_pairs)
            if len(trial) < best_size:
                best_size = len(trial)
                best = trial
    
    return [all_points[i] for i in best]

# EVOLVE-BLOCK-END
