"""Combined program: vectorized violation checking + simulated-annealing local search."""
import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120
MAX_COLOR = 119

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def _build_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    cycles.append((_edge_idx(a, b), _edge_idx(b, c), _edge_idx(c, d), _edge_idx(d, a)))
                    cycles.append((_edge_idx(a, b), _edge_idx(b, d), _edge_idx(d, c), _edge_idx(c, a)))
                    cycles.append((_edge_idx(a, c), _edge_idx(c, b), _edge_idx(b, d), _edge_idx(d, a)))
    return np.array(cycles, dtype=np.int32)

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# Precompute edge-to-cycles mapping
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci in range(NUM_CYCLES):
    for e in CYCLES[ci]:
        EDGE_TO_CYCLES[e].append(ci)

def _count_violations_vectorized(colors):
    cycle_colors = colors[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    distinct = (sorted_colors[:, 1] != sorted_colors[:, 0]) & \
               (sorted_colors[:, 2] != sorted_colors[:, 1]) & \
               (sorted_colors[:, 3] != sorted_colors[:, 2])
    return int(np.sum(distinct))

def _is_valid_fast(colors):
    return _count_violations_vectorized(colors) == 0

def _count_colors(colors):
    return len(set(colors.tolist() if isinstance(colors, np.ndarray) else colors))

def _algebraic_init(a, b, c, M):
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            colors[idx] = (a * i + b * j + c) % M
            idx += 1
    return colors

def _rank1_coloring(u, v, K):
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            colors[idx] = (u[i] * v[j] + u[j] * v[i]) % K
            idx += 1
    return colors

def _fix_violations(colors, rng, max_iters=2000):
    """Fix violations by reassigning random edges."""
    colors = colors.copy()
    for _ in range(max_iters):
        violated = False
        # Check for violations
        cycle_colors = colors[CYCLES]
        sorted_colors = np.sort(cycle_colors, axis=1)
        distinct = (sorted_colors[:, 1] != sorted_colors[:, 0]) & \
                   (sorted_colors[:, 2] != sorted_colors[:, 1]) & \
                   (sorted_colors[:, 3] != sorted_colors[:, 2])
        violating = np.where(distinct)[0]
        if len(violating) == 0:
            break
        violated = True
        # Pick a random violating cycle and fix one of its edges
        vi = rng.randint(0, len(violating) - 1)
        cyc = CYCLES[violating[vi]]
        edge = int(cyc[rng.randint(0, 3)])
        colors[edge] = rng.randint(0, MAX_COLOR)
    return colors, _is_valid_fast(colors)

def _simulated_annealing(colors, deadline, seed_val=42):
    """Simulated annealing with swap moves to maximize distinct colors."""
    rng = np.random.RandomState(seed_val)
    current = colors.copy()
    current_distinct = len(set(current.tolist()))
    best = current.copy()
    best_distinct = current_distinct
    
    color_counts = np.zeros(MAX_COLOR + 1, dtype=np.int32)
    for c in current:
        color_counts[c] += 1
    
    T = 1.0
    cooling = 0.99998
    min_T = 0.001
    no_improve = 0
    
    while time.time() < deadline and T > min_T:
        T *= cooling
        no_improve += 1
        
        # Pick a random edge
        e_idx = rng.randint(0, NUM_EDGES)
        old_color = int(current[e_idx])
        
        # Pick new color: bias toward unused colors
        if rng.random() < 0.5:
            unused_mask = color_counts == 0
            unused_indices = np.where(unused_mask)[0]
            if len(unused_indices) > 0:
                new_color = int(unused_indices[rng.randint(0, len(unused_indices) - 1)])
            else:
                new_color = int(rng.randint(0, MAX_COLOR + 1))
        else:
            new_color = int(rng.randint(0, MAX_COLOR + 1))
        
        if new_color == old_color:
            continue
        
        # Check if change creates violations
        affected_cycles = EDGE_TO_CYCLES[e_idx]
        safe = True
        for ci in affected_cycles:
            cyc = CYCLES[ci]
            c0 = int(current[cyc[0]])
            c1 = int(current[cyc[1]])
            c2 = int(current[cyc[2]])
            c3 = int(current[cyc[3]])
            
            nc0, nc1, nc2, nc3 = c0, c1, c2, c3
            if cyc[0] == e_idx:
                nc0 = new_color
            elif cyc[1] == e_idx:
                nc1 = new_color
            elif cyc[2] == e_idx:
                nc2 = new_color
            elif cyc[3] == e_idx:
                nc3 = new_color
            
            if len({nc0, nc1, nc2, nc3}) == 4:
                safe = False
                break
        
        if not safe:
            continue
        
        # Apply change
        current[e_idx] = new_color
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        
        # Update distinct count
        new_distinct = current_distinct
        if color_counts[old_color] == 0:
            new_distinct -= 1
        if color_counts[new_color] == 1:
            new_distinct += 1
        
        # Accept or reject
        if new_distinct > current_distinct:
            current_distinct = new_distinct
            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best = current.copy()
                no_improve = 0
        elif rng.random() < np.exp((new_distinct - current_distinct) / T):
            current_distinct = new_distinct
        else:
            # Reject
            current[e_idx] = old_color
            color_counts[old_color] += 1
            color_counts[new_color] -= 1
    
    return best, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_colors = None
    best_count = 0
    
    rng = random.Random(42)
    
    # Strategy 1: Linear algebraic colorings
    linear_configs = []
    for M in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]:
        for a in range(1, min(M, 6) + 1):
            for b in range(1, min(M, 6) + 1):
                if a != b:
                    linear_configs.append((a, b, 0, M))
    
    # Strategy 2: Rank-1 algebraic colorings
    rank1_configs = []
    for K in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        for _ in range(2):
            u = [rng.randint(1, K - 1) for _ in range(N)]
            v = [rng.randint(1, K - 1) for _ in range(N)]
            rank1_configs.append((u, v, K))
        u_struct = [(i + 1) % K for i in range(N)]
        v_struct = [(j + 2) % K for j in range(N)]
        rank1_configs.append((u_struct, v_struct, K))
    
    all_configs = []
    for a, b, c, M in linear_configs:
        all_configs.append(('linear', a, b, c, M))
    for u, v, K in rank1_configs:
        all_configs.append(('rank1', u, v, K))
    
    rng.shuffle(all_configs)
    
    # Try each configuration
    for config in all_configs:
        if time.time() >= deadline - 5.0:
            break
        
        if config[0] == 'linear':
            _, a, b, c, M = config
            colors = _algebraic_init(a, b, c, M)
        else:  # rank1
            _, u, v, K = config
            colors = _rank1_coloring(u, v, K)
        
        # Fix violations if any
        colors, valid = _fix_violations(colors, rng)
        if not valid:
            continue
        
        count = _count_colors(colors)
        if count > best_count:
            best_count = count
            best_colors = colors.copy()
        
        # Simulated annealing to improve
        if time.time() < deadline - 3.0:
            sa_colors, sa_count = _simulated_annealing(colors, deadline - 1.0, seed_val=rng.randint(0, 2**31))
            if sa_count > best_count:
                best_count = sa_count
                best_colors = sa_colors.copy()
    
    # Final aggressive improvement on the best solution
    if best_colors is not None and time.time() < deadline - 2.0:
        final_colors, final_count = _simulated_annealing(best_colors, deadline - 0.5, seed_val=rng.randint(0, 2**31))
        if final_count > best_count:
            best_count = final_count
            best_colors = final_colors
    
    if best_colors is None:
        return [0] * NUM_EDGES
    
    return best_colors.tolist()
