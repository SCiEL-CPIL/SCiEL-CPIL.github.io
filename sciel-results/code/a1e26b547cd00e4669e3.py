"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import random
import itertools
import time

DEADLINE = time.time() + 280  # safety margin before 330s limit


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
    """Check if the set is maximal (no further point can be added)."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        if not _creates_isosceles(p, chosen):
            return False
    return True


def _greedy_construct(order, start_set=None):
    """Greedy construction: add points in given order if they don't create isosceles."""
    chosen = list(start_set) if start_set else []
    chosen_set = set(chosen)
    for p in order:
        if p in chosen_set:
            continue
        if not _creates_isosceles(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen


def _try_reduce(chosen, rng):
    """Try to reduce the maximal set by removing points and rebuilding."""
    if not chosen:
        return chosen
    
    best = list(chosen)
    n = len(best)
    
    # Try removing each point and see if we can rebuild a smaller maximal set
    for i in range(n):
        if time.time() > DEADLINE:
            break
        
        # Remove point i
        remaining = [best[j] for j in range(n) if j != i]
        removed_point = best[i]
        
        # Try greedy rebuild from remaining, with randomized order
        shuffled_remaining = remaining[:]
        rng.shuffle(shuffled_remaining)
        
        # Also try adding the removed point at a random position in the order
        pos = rng.randint(0, len(shuffled_remaining))
        order = shuffled_remaining[:pos] + [removed_point] + shuffled_remaining[pos:]
        
        candidate = _greedy_construct(order)
        
        # Check if candidate is maximal and smaller
        if len(candidate) < len(best) and _is_maximal(candidate):
            best = candidate
            n = len(best)
    
    return best


def construct():
    rng = random.Random(42)
    all_points = list(itertools.product(range(N), repeat=2))
    
    best = None
    best_size = float('inf')
    
    # Multiple randomized greedy constructions
    num_starts = 50
    for start in range(num_starts):
        if time.time() > DEADLINE:
            break
        
        # Random order
        order = all_points[:]
        rng.shuffle(order)
        
        # Greedy construct
        candidate = _greedy_construct(order)
        
        # Verify maximality
        if not _is_maximal(candidate):
            # Force maximality by adding points
            candidate_set = set(candidate)
            for p in all_points:
                if p in candidate_set:
                    continue
                if not _creates_isosceles(p, candidate):
                    candidate.append(p)
                    candidate_set.add(p)
        
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
        
        # Try local reduction
        if time.time() < DEADLINE - 60:
            reduced = _try_reduce(candidate, rng)
            if len(reduced) < best_size:
                best = reduced
                best_size = len(reduced)
    
    if best is None:
        # Fallback: simple greedy
        best = _greedy_construct(all_points)
    
    return best
# EVOLVE-BLOCK-END
