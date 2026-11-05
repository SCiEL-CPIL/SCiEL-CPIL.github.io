"""Minimum maximal no-isosceles set on 16x16 grid."""
import time
import random
import itertools

N = 16
DEADLINE = 280  # seconds, safety margin from 330s limit

# Precompute all grid points
ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_POINTS = N * N

# Precompute squared distances between all pairs
# dist[i][j] = squared distance between ALL_POINTS[i] and ALL_POINTS[j]
def _precompute():
    dist = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = ALL_POINTS[i]
        for j in range(i + 1, NUM_POINTS):
            xj, yj = ALL_POINTS[j]
            d = (xi - xj) ** 2 + (yi - yj) ** 2
            dist[i][j] = d
            dist[j][i] = d
    return dist

DIST = _precompute()


def _creates_isosceles(p_idx, chosen_set):
    """Check if adding point p_idx to chosen_set creates any isosceles triangle."""
    p = p_idx
    for i, a in enumerate(chosen_set):
        d_pa = DIST[p][a]
        for b in chosen_set[i + 1:]:
            d_pb = DIST[p][b]
            d_ab = DIST[a][b]
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _is_maximal(chosen_set):
    """Check if chosen_set is maximal: no point can be added without creating isosceles."""
    chosen_list = list(chosen_set)
    for p_idx in range(NUM_POINTS):
        if p_idx in chosen_set:
            continue
        if not _creates_isosceles(p_idx, chosen_list):
            return False
    return True


def _greedy_construct(order, chosen_set):
    """Add points from order to chosen_set if they don't create isosceles triangles."""
    chosen_list = list(chosen_set)
    for p_idx in order:
        if p_idx in chosen_set:
            continue
        if not _creates_isosceles(p_idx, chosen_list):
            chosen_set.add(p_idx)
            chosen_list.append(p_idx)
    return chosen_set


def _reduce_maximal(chosen_set):
    """Try to reduce a maximal no-isosceles set by removing points that don't break maximality."""
    chosen_set = set(chosen_set)
    changed = True
    while changed:
        changed = False
        # Try removing each point
        for p_idx in list(chosen_set):
            candidate = chosen_set - {p_idx}
            if not candidate:
                continue
            # Check if candidate is still no-isosceles (it is, since it's a subset)
            # Check if candidate is still maximal
            if _is_maximal(candidate):
                chosen_set = candidate
                changed = True
                break
    return chosen_set


def construct():
    start_time = time.time()
    best_size = NUM_POINTS + 1
    best_set = None
    
    # Multiple restarts with different orderings
    rng = random.Random(42)
    
    # Try several different strategies
    strategies = []
    
    # Strategy 1: Random order
    for _ in range(3):
        order = list(range(NUM_POINTS))
        rng.shuffle(order)
        strategies.append(order)
    
    # Strategy 2: Spatial clustering - start from corners/edges
    for _ in range(2):
        # Sort by distance from center
        cx, cy = N / 2, N / 2
        order = sorted(range(NUM_POINTS), key=lambda i: (
            (ALL_POINTS[i][0] - cx) ** 2 + (ALL_POINTS[i][1] - cy) ** 2
        ))
        # Randomize within same distance bands
        strategies.append(order)
    
    # Strategy 3: Diagonal order
    order = sorted(range(NUM_POINTS), key=lambda i: (
        ALL_POINTS[i][0] + ALL_POINTS[i][1],
        rng.random()
    ))
    strategies.append(order)
    
    # Strategy 4: Reverse diagonal
    order = sorted(range(NUM_POINTS), key=lambda i: (
        ALL_POINTS[i][0] + ALL_POINTS[i][1],
        -rng.random()
    ))
    strategies.append(order)
    
    # Strategy 5: Random with bias toward sparse regions
    for _ in range(3):
        # Weight points by how many isosceles they'd form with a random subset
        order = list(range(NUM_POINTS))
        # Shuffle with some spatial correlation
        for i in range(len(order) - 1, 0, -1):
            j = rng.randint(0, i)
            order[i], order[j] = order[j], order[i]
        strategies.append(order)
    
    for order in strategies:
        if time.time() - start_time > DEADLINE:
            break
        
        chosen_set = set()
        chosen_set = _greedy_construct(order, chosen_set)
        
        # Reduce the maximal set
        reduced = _reduce_maximal(chosen_set)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
        
        # If we found a very small set, try more aggressive reduction
        if best_size <= 8:
            if time.time() - start_time > DEADLINE:
                break
    
    # If no set found (shouldn't happen), fallback to greedy
    if best_set is None:
        chosen_set = set()
        order = list(range(NUM_POINTS))
        chosen_set = _greedy_construct(order, chosen_set)
        best_set = _reduce_maximal(chosen_set)
        best_size = len(best_set)
    
    # Convert back to list of (x, y) tuples
    result = [ALL_POINTS[i] for i in sorted(best_set)]
    return result


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Points: {result}")
