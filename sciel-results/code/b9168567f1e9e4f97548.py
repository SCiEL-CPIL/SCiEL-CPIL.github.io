"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def _edge_idx(i, j):
    """Return the index in the lexicographic edge list for edge (i, j) with i < j."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def _build_cycles():
    """Build all 3 four-cycles for each 4-vertex subset."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Three 4-cycles on vertices a,b,c,d
                    # Cycle 1: a-b-c-d-a
                    cycles.append((
                        _edge_idx(a, b),
                        _edge_idx(b, c),
                        _edge_idx(c, d),
                        _edge_idx(d, a)
                    ))
                    # Cycle 2: a-b-d-c-a
                    cycles.append((
                        _edge_idx(a, b),
                        _edge_idx(b, d),
                        _edge_idx(d, c),
                        _edge_idx(c, a)
                    ))
                    # Cycle 3: a-c-b-d-a
                    cycles.append((
                        _edge_idx(a, c),
                        _edge_idx(c, b),
                        _edge_idx(b, d),
                        _edge_idx(d, a)
                    ))
    return cycles

def _is_valid(colors, cycles):
    """Check no four-cycle has four distinct edge colors."""
    for e1, e2, e3, e4 in cycles:
        c1, c2, c3, c4 = colors[e1], colors[e2], colors[e3], colors[e4]
        if len({c1, c2, c3, c4}) == 4:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _algebraic_init(a, b, c, M):
    """Generate algebraic coloring (a*i + b*j + c) % M."""
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = _edge_idx(i, j)
            colors[idx] = (a * i + b * j + c) % M
    return colors

def _local_search(colors, cycles, deadline, seed):
    """Local search: try changing individual edge colors to maximize distinct colors."""
    rng = random.Random(seed)
    best = list(colors)
    best_count = _count_colors(best)
    
    # Precompute for each edge, which cycles it belongs to
    edge_to_cycles = {e: [] for e in range(NUM_EDGES)}
    for ci, (e1, e2, e3, e4) in enumerate(cycles):
        for e in (e1, e2, e3, e4):
            edge_to_cycles[e].append(ci)
    
    # Try each edge, each candidate color
    edges = list(range(NUM_EDGES))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rng.shuffle(edges)
        for e in edges:
            if time.time() >= deadline:
                break
            old_c = best[e]
            # Try all possible colors
            for new_c in range(MAX_COLORS):
                if new_c == old_c:
                    continue
                # Quick check: does this create a rainbow cycle?
                valid = True
                for ci in edge_to_cycles[e]:
                    e1, e2, e3, e4 = cycles[ci]
                    c1 = best[e1] if e1 != e else new_c
                    c2 = best[e2] if e2 != e else new_c
                    c3 = best[e3] if e3 != e else new_c
                    c4 = best[e4] if e4 != e else new_c
                    if len({c1, c2, c3, c4}) == 4:
                        valid = False
                        break
                if valid:
                    # Check if this increases color count
                    best[e] = new_c
                    new_count = _count_colors(best)
                    if new_count > best_count:
                        best_count = new_count
                        improved = True
                    else:
                        best[e] = old_c
                    break  # Only try first valid color that doesn't decrease count
    
    return best, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    cycles = _build_cycles()
    
    best_colors = None
    best_count = 0
    
    # Try multiple algebraic parameter combinations
    params = []
    for M in [10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 80, 100, 120]:
        for a in range(1, min(M, 16) + 1):
            for b in range(1, min(M, 16) + 1):
                if a == b and M <= 20:
                    continue
                params.append((a, b, 0, M))
                if len(params) > 200:
                    break
            if len(params) > 200:
                break
        if len(params) > 200:
            break
    
    # Also try some random initializations
    rng = random.Random(42)
    for _ in range(20):
        M = rng.choice([10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 80, 100, 120])
        a = rng.randint(1, M - 1)
        b = rng.randint(1, M - 1)
        c = rng.randint(0, M - 1)
        params.append((a, b, c, M))
    
    for pi, (a, b, c, M) in enumerate(params):
        if time.time() >= deadline - 5:
            break
        
        colors = _algebraic_init(a, b, c, M)
        
        # Check validity
        if not _is_valid(colors, cycles):
            continue
        
        count = _count_colors(colors)
        if count > best_count:
            best_count = count
            best_colors = list(colors)
        
        # Local search to improve
        if time.time() < deadline - 3:
            ls_colors, ls_count = _local_search(colors, cycles, deadline - 1, seed=pi)
            if ls_count > best_count:
                best_count = ls_count
                best_colors = ls_colors
    
    # If no valid coloring found, return all zeros (valid but score 1)
    if best_colors is None:
        return [0] * NUM_EDGES
    
    return best_colors
# EVOLVE-BLOCK-END
