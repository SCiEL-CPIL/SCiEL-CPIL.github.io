"""Search for a Hamilton cycle minimizing the max edge arrival index."""
import random
import time

# Build the pinned edge order
def _build_edge_order():
    n = 24
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    rng = random.Random(614)
    rng.shuffle(edges)
    return edges

EDGE_ORDER = _build_edge_order()
N = 24

# Precompute edge index lookup: (u,v) -> 1-based index
EDGE_INDEX = {}
for idx, (u, v) in enumerate(EDGE_ORDER, 1):
    EDGE_INDEX[(u, v)] = idx
    EDGE_INDEX[(v, u)] = idx

def cycle_score(cycle):
    """Return the max 1-based edge index in the Hamilton cycle."""
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u = cycle[i]
        v = cycle[(i + 1) % n]
        idx = EDGE_INDEX[(u, v)]
        if idx > max_idx:
            max_idx = idx
    return max_idx

def two_opt_improve(cycle, max_iter=5000):
    """2-opt local search minimizing max edge index."""
    n = len(cycle)
    best = cycle_score(cycle)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Try reversing segment [i+1..j]
                new_cycle = cycle[:i+1] + cycle[i+1:j+1][::-1] + cycle[j+1:]
                sc = cycle_score(new_cycle)
                if sc < best:
                    cycle = new_cycle
                    best = sc
                    improved = True
                    break
            if improved:
                break
    return cycle, best

def random_cycle(seed):
    rng = random.Random(seed)
    c = list(range(N))
    rng.shuffle(c)
    return c

def construct():
    deadline = time.time() + 80  # 80s safety margin (limit is 90s)
    
    best_cycle = list(range(N))
    best_score = cycle_score(best_cycle)
    
    # Multiple random restarts with 2-opt
    seed_counter = 0
    while time.time() < deadline:
        seed_counter += 1
        c = random_cycle(seed_counter)
        c, sc = two_opt_improve(c, max_iter=2000)
        if sc < best_score:
            best_cycle = c
            best_score = sc
            # If we get very low, try more aggressive improvement
            if best_score <= 6:
                c, sc = two_opt_improve(c, max_iter=10000)
                if sc < best_score:
                    best_cycle = c
                    best_score = sc
    
    return best_cycle
