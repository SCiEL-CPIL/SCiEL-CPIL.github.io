"""Improved program: randomized greedy + local search minimization with deadline guard."""
import random
import time
import itertools

N = 16
TIME_LIMIT = 300.0  # seconds, with safety margin
DEADLINE = time.time() + TIME_LIMIT - 10.0  # 10s safety margin


def _is_isosceles(a, b, c):
    """Check if triangle with vertices a, b, c is isosceles."""
    d1 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    d2 = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    d3 = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(point, chosen):
    """Check if adding point to chosen keeps the set no-isosceles."""
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_isosceles(point, chosen[i], chosen[j]):
                return False
    return True


def _is_maximal(chosen):
    """Check if no further grid point can be added to chosen."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p not in chosen_set:
            if _can_add(p, chosen):
                return False
    return True


def _randomized_greedy(rng, order):
    """Build a no-isosceles set by greedily adding points in a given order."""
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen


def _remove_redundant(chosen, rng):
    """Try to remove points that don't affect maximality.
    A point is redundant if removing it still leaves the set maximal
    (or we can re-add other points to maintain maximality)."""
    chosen = list(chosen)
    # Shuffle to randomize which points we try to remove
    indices = list(range(len(chosen)))
    rng.shuffle(indices)
    
    improved = True
    while improved:
        improved = False
        for idx in indices:
            if idx >= len(chosen):
                continue
            candidate = chosen[idx]
            # Try removing this point
            remaining = [c for i, c in enumerate(chosen) if i != idx]
            # Check if remaining is still no-isosceles (it should be, since it's a subset)
            # Check if remaining is maximal
            if _is_maximal(remaining):
                chosen = remaining
                improved = True
                # Reset indices since list changed
                indices = list(range(len(chosen)))
                rng.shuffle(indices)
                break
    return chosen


def _local_search(chosen, rng, time_budget):
    """Local search: try swapping points to reduce set size while maintaining maximality."""
    chosen = list(chosen)
    start_time = time.time()
    
    # Precompute all grid points
    all_points = list(itertools.product(range(N), repeat=2))
    
    while time.time() < start_time + time_budget and time.time() < DEADLINE:
        # Try to find a point to remove and a different point to add
        # that maintains no-isosceles property and maximality
        found = False
        indices = list(range(len(chosen)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() >= start_time + time_budget or time.time() >= DEADLINE:
                break
            candidate_remove = chosen[idx]
            remaining = [c for i, c in enumerate(chosen) if i != idx]
            
            # Try adding a random point from remaining grid
            grid_without_chosen = [p for p in all_points if p not in set(remaining)]
            if not grid_without_chosen:
                continue
            
            # Sample a few candidates
            samples = min(20, len(grid_without_chosen))
            candidates = rng.sample(grid_without_chosen, samples)
            
            for new_point in candidates:
                if _can_add(new_point, remaining):
                    new_set = remaining + [new_point]
                    # Check if this new set is maximal
                    if _is_maximal(new_set):
                        chosen = new_set
                        found = True
                        break
            if found:
                break
        
        if not found:
            # Try pure removal
            for idx in indices:
                if time.time() >= start_time + time_budget or time.time() >= DEADLINE:
                    break
                remaining = [c for i, c in enumerate(chosen) if i != idx]
                if _is_maximal(remaining):
                    chosen = remaining
                    found = True
                    break
        
        if not found:
            break
    
    return chosen


def construct():
    """Construct a minimal maximal no-isosceles set."""
    rng = random.Random(42)
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    best_set = None
    best_size = float('inf')
    
    # Multiple randomized greedy attempts with different orderings
    num_restarts = 50
    restart_time_budget = (TIME_LIMIT - 30) / num_restarts  # Leave 30s for local search
    
    for restart in range(num_restarts):
        if time.time() >= DEADLINE:
            break
        
        # Create a random ordering
        order = list(all_points)
        rng.shuffle(order)
        
        # Randomized greedy construction
        chosen = _randomized_greedy(rng, order)
        
        # Try to minimize by removing redundant points
        chosen = _remove_redundant(chosen, rng)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = list(chosen)
        
        # Check deadline
        elapsed = time.time() - (time.time() - (TIME_LIMIT - 10.0 - time.time()))
        remaining = DEADLINE - time.time()
        
        if remaining < 30:
            break
    
    # Local search on the best set found
    if best_set is not None:
        remaining = DEADLINE - time.time()
        if remaining > 5:
            best_set = _local_search(best_set, rng, min(remaining - 5, 60))
            best_size = len(best_set)
    
    # Final verification: ensure the set is maximal and no-isosceles
    if best_set is None:
        # Fallback: simple greedy
        chosen = []
        for p in all_points:
            if _can_add(p, chosen):
                chosen.append(p)
        best_set = chosen
    
    # Verify no-isosceles
    for i in range(len(best_set)):
        for j in range(i + 1, len(best_set)):
            for k in range(j + 1, len(best_set)):
                if _is_isosceles(best_set[i], best_set[j], best_set[k]):
                    # Should not happen, but if it does, fall back
                    chosen = []
                    for p in all_points:
                        if _can_add(p, chosen):
                            chosen.append(p)
                    best_set = chosen
                    break
    
    return best_set
