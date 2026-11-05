import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
NUM_COLORS = 120

# Precompute edge indices
edges = []
edge_index = {}
idx = 0
for i in range(N):
    for j in range(i + 1, N):
        edges.append((i, j))
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
        idx += 1

# Precompute 4-cycles
cycles = []
edge_to_cycles = [[] for _ in range(NUM_EDGES)]

for v0 in range(N):
    for v1 in range(v0 + 1, N):
        for v2 in range(v1 + 1, N):
            for v3 in range(v2 + 1, N):
                # Cycle 1: v0-v1-v2-v3-v0
                e1 = edge_index[(v0, v1)]
                e2 = edge_index[(v1, v2)]
                e3 = edge_index[(v2, v3)]
                e4 = edge_index[(v3, v0)]
                cycles.append((e1, e2, e3, e4))
                edge_to_cycles[e1].append(len(cycles) - 1)
                edge_to_cycles[e2].append(len(cycles) - 1)
                edge_to_cycles[e3].append(len(cycles) - 1)
                edge_to_cycles[e4].append(len(cycles) - 1)
                
                # Cycle 2: v0-v1-v3-v2-v0
                e1 = edge_index[(v0, v1)]
                e2 = edge_index[(v1, v3)]
                e3 = edge_index[(v3, v2)]
                e4 = edge_index[(v2, v0)]
                cycles.append((e1, e2, e3, e4))
                edge_to_cycles[e1].append(len(cycles) - 1)
                edge_to_cycles[e2].append(len(cycles) - 1)
                edge_to_cycles[e3].append(len(cycles) - 1)
                edge_to_cycles[e4].append(len(cycles) - 1)
                
                # Cycle 3: v0-v2-v1-v3-v0
                e1 = edge_index[(v0, v2)]
                e2 = edge_index[(v2, v1)]
                e3 = edge_index[(v1, v3)]
                e4 = edge_index[(v3, v0)]
                cycles.append((e1, e2, e3, e4))
                edge_to_cycles[e1].append(len(cycles) - 1)
                edge_to_cycles[e2].append(len(cycles) - 1)
                edge_to_cycles[e3].append(len(cycles) - 1)
                edge_to_cycles[e4].append(len(cycles) - 1)

# Precompute cycle edge lists for faster lookup
cycle_edges = [list(c) for c in cycles]

def is_valid_assignment_fast(edge_idx, new_color, colors):
    """Check if assigning new_color to edge_idx results in any rainbow 4-cycle."""
    for c_idx in edge_to_cycles[edge_idx]:
        e1, e2, e3, e4 = cycle_edges[c_idx]
        # Find the other 3 edges
        if e1 == edge_idx:
            other = [e2, e3, e4]
        elif e2 == edge_idx:
            other = [e1, e3, e4]
        elif e3 == edge_idx:
            other = [e1, e2, e4]
        else:
            other = [e1, e2, e3]
        
        c1 = colors[other[0]]
        c2 = colors[other[1]]
        c3 = colors[other[2]]
        
        # If any of the other edges has the new color, not rainbow
        if c1 == new_color or c2 == new_color or c3 == new_color:
            continue
            
        # If the other 3 are all distinct, we'd have 4 distinct colors
        if c1 != c2 and c1 != c3 and c2 != c3:
            return False
            
    return True

def greedy_coloring(order, rng):
    """Greedy: assign colors in given edge order, preferring new colors."""
    colors = [0] * NUM_EDGES
    used_colors = {0}
    available = list(range(1, NUM_COLORS))
    rng.shuffle(available)
    
    for e_idx in order:
        assigned = False
        for c in available:
            if is_valid_assignment_fast(e_idx, c, colors):
                colors[e_idx] = c
                used_colors.add(c)
                available.remove(c)
                assigned = True
                break
        if not assigned:
            colors[e_idx] = 0  # Keep as 0
    
    return colors, len(used_colors)

def local_improve(colors, rng, deadline, max_iterations=50000):
    """Aggressive local search: try to recolor edges to new unused colors."""
    colors = colors[:]
    used = set(colors)
    current_count = len(used)
    best_count = current_count
    best_colors = colors[:]
    
    unused = [c for c in range(NUM_COLORS) if c not in used]
    
    for _ in range(max_iterations):
        if time.time() >= deadline:
            break
        
        e = rng.randint(0, NUM_EDGES - 1)
        old = colors[e]
        
        # Prefer trying new unused colors
        if unused:
            # Try a few random unused colors
            candidates = rng.sample(unused, min(5, len(unused)))
        else:
            break
        
        found = False
        for new_c in candidates:
            if new_c == old:
                continue
            if is_valid_assignment_fast(e, new_c, colors):
                colors[e] = new_c
                used.add(new_c)
                unused.remove(new_c)
                current_count = len(used)
                if current_count > best_count:
                    best_count = current_count
                    best_colors = colors[:]
                found = True
                break
        
        if not found:
            # Try recoloring to a different used color to potentially free up constraints
            if rng.random() < 0.3:
                other_used = [c for c in used if c != old]
                if other_used:
                    new_c = rng.choice(other_used)
                    if is_valid_assignment_fast(e, new_c, colors):
                        colors[e] = new_c
                        # Check if old color is now unused
                        if colors.count(old) == 0:
                            used.discard(old)
                            unused.append(old)
                            rng.shuffle(unused)
    
    return best_colors, best_count

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin before 90s kill
    
    best_result = [0] * NUM_EDGES
    best_score = 1
    
    rng = random.Random(42)
    
    # Generate diverse edge orderings
    edge_indices = list(range(NUM_EDGES))
    orderings = []
    
    # Natural order
    orderings.append(list(edge_indices))
    
    # Reverse
    orderings.append(list(reversed(edge_indices)))
    
    # Degree-based (edges in fewer cycles first)
    edge_cycle_count = [len(edge_to_cycles[e]) for e in range(NUM_EDGES)]
    degree_order = sorted(edge_indices, key=lambda e: edge_cycle_count[e])
    orderings.append(degree_order)
    orderings.append(list(reversed(degree_order)))
    
    # Random shuffles
    for _ in range(20):
        shuffled = list(edge_indices)
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Run greedy on each ordering, then local improve
    candidates = []
    for order in orderings:
        if time.time() - start_time > time_limit - 20:
            break
        colors, count = greedy_coloring(order, rng)
        candidates.append((count, colors[:]))
    
    # Sort by count descending
    candidates.sort(key=lambda x: -x[0])
    
    # Local improve on top candidates
    for cnt, c in candidates[:10]:
        if time.time() - start_time > time_limit - 5:
            break
        improved_c, improved_cnt = local_improve(c, rng, time_limit - 2)
        if improved_cnt > best_score:
            best_score = improved_cnt
            best_result = improved_c[:]
    
    # Additional local search rounds on the best
    if time.time() - start_time < time_limit - 10:
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            improved_c, improved_cnt = local_improve(best_result, rng, time_limit - 2)
            if improved_cnt > best_score:
                best_score = improved_cnt
                best_result = improved_c[:]
    
    return best_result
