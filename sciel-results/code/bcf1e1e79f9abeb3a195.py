"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys

DEADLINE = time.time() + 280  # Safety margin before 330s kill

# Precompute all points and squared distances
POINTS = [(x, y) for y in range(N) for x in range(N)]
NUM_PTS = N * N

# dist[i][j] = squared distance between point i and point j
dist = [[0] * NUM_PTS for _ in range(NUM_PTS)]
for i in range(NUM_PTS):
    x1, y1 = POINTS[i]
    for j in range(i + 1, NUM_PTS):
        x2, y2 = POINTS[j]
        d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
        dist[i][j] = d
        dist[j][i] = d


def is_isosceles(i, j, k):
    """Check if points i, j, k form an isosceles triangle."""
    d1 = dist[i][j]
    d2 = dist[i][k]
    d3 = dist[j][k]
    return d1 == d2 or d1 == d3 or d2 == d3


def can_add(p, chosen):
    """Check if point p can be added to chosen without forming isosceles."""
    for a, b in itertools.combinations(chosen, 2):
        if is_isosceles(p, a, b):
            return False
    return True


def is_maximal(chosen):
    """Check if no further point can be added to chosen."""
    chosen_set = set(chosen)
    for p in range(NUM_PTS):
        if p not in chosen_set and can_add(p, chosen):
            return False
    return True


def greedy_construct(order):
    """Greedy construction following given order of points."""
    chosen = []
    for p in order:
        if can_add(p, chosen):
            chosen.append(p)
    return chosen


def local_search(chosen, max_iterations=500):
    """Local search: try removing a point and refilling greedily."""
    best = list(chosen)
    best_len = len(best)
    
    for iteration in range(max_iterations):
        if time.time() > DEADLINE:
            break
        
        # Try removing each point
        for i in range(len(best)):
            if time.time() > DEADLINE:
                break
            
            # Remove point i
            new_chosen = [p for j, p in enumerate(best) if j != i]
            
            # Try to add other points greedily
            removed_point = best[i]
            for p in range(NUM_PTS):
                if p not in new_chosen and can_add(p, new_chosen):
                    new_chosen.append(p)
            
            # If we got a smaller set, keep it
            if len(new_chosen) < best_len:
                best = new_chosen
                best_len = len(best)
                break  # Restart local search with new best
    
    return best


def randomized_greedy(seed):
    """Randomized greedy construction."""
    rng = random.Random(seed)
    order = list(range(NUM_PTS))
    rng.shuffle(order)
    chosen = greedy_construct(order)
    return chosen


def construct():
    """Construct a maximal no-isosceles set with minimum size."""
    best = None
    best_len = float('inf')
    
    # First, do a deterministic row-major greedy as baseline
    baseline_order = list(range(NUM_PTS))
    baseline = greedy_construct(baseline_order)
    best = baseline
    best_len = len(best)
    
    # Try randomized restarts
    seed = 42
    while time.time() < DEADLINE:
        if time.time() > DEADLINE:
            break
        
        # Randomized greedy
        chosen = randomized_greedy(seed)
        if len(chosen) < best_len:
            best = chosen
            best_len = len(chosen)
        
        # Local search improvement
        improved = local_search(chosen, max_iterations=100)
        if len(improved) < best_len:
            best = improved
            best_len = len(improved)
        
        seed += 1
    
    # Convert indices back to coordinates
    return [POINTS[i] for i in best]
# EVOLVE-BLOCK-END
