"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16
DEADLINE = 320  # seconds, safety margin before 330s kill


def _dist_sq(p, q):
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _creates_isosceles(p, chosen):
    """Check if adding p to chosen creates any isosceles triangle."""
    n = len(chosen)
    for i in range(n):
        a = chosen[i]
        d_pa = _dist_sq(p, a)
        for j in range(i + 1, n):
            b = chosen[j]
            d_pb = _dist_sq(p, b)
            d_ab = _dist_sq(a, b)
            if d_pa == d_pb or d_pa == d_ab or d_pb == d_ab:
                return True
    return False


def _is_maximal(chosen):
    """Check that no grid point can be added without creating an isosceles triangle."""
    chosen_set = set(chosen)
    for x in range(N):
        for y in range(N):
            p = (x, y)
            if p in chosen_set:
                continue
            if not _creates_isosceles(p, chosen):
                return False
    return True


def _greedy_construct(order):
    """Greedy construction following the given point order."""
    chosen = []
    for p in order:
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
    return chosen


def _try_remove(chosen, idx):
    """Try removing chosen[idx] and see if the set remains no-isosceles."""
    remaining = [chosen[i] for i in range(len(chosen)) if i != idx]
    # Check if remaining is still no-isosceles
    n = len(remaining)
    for i in range(n):
        a = remaining[i]
        for j in range(i + 1, n):
            b = remaining[j]
            d_ab = _dist_sq(a, b)
            for k in range(j + 1, n):
                c = remaining[k]
                d_ac = _dist_sq(a, c)
                d_bc = _dist_sq(b, c)
                if d_ab == d_ac or d_ab == d_bc or d_ac == d_bc:
                    return None
    return remaining


def _local_search(chosen, deadline):
    """Attempt to reduce set size by removing points."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(chosen)
        for idx in range(n):
            if time.time() >= deadline:
                break
            new_set = _try_remove(chosen, idx)
            if new_set is not None:
                # Check maximality of new set
                if _is_maximal(new_set):
                    chosen = new_set
                    improved = True
                    break
    return chosen


def _random_order(seed):
    """Generate a random point order."""
    rng = random.Random(seed)
    points = list(itertools.product(range(N), repeat=2))
    rng.shuffle(points)
    return points


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best = None
    best_size = float('inf')
    
    # Try multiple random orders
    base_order = list(itertools.product(range(N), repeat=2))
    
    # First, try the standard row-major order
    if time.time() < deadline - 10:
        chosen = _greedy_construct(base_order)
        if _is_maximal(chosen):
            best = chosen
            best_size = len(chosen)
    
    # Try several random orders
    seeds = range(50)
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        order = _random_order(seed)
        chosen = _greedy_construct(order)
        if not _is_maximal(chosen):
            continue
        if len(chosen) < best_size:
            best = chosen
            best_size = len(chosen)
        
        # Try local search to reduce size
        if time.time() < deadline - 10:
            improved = _local_search(chosen, deadline - 5)
            if _is_maximal(improved) and len(improved) < best_size:
                best = improved
                best_size = len(improved)
    
    if best is None:
        # Fallback: simple greedy
        best = _greedy_construct(base_order)
    
    return best
