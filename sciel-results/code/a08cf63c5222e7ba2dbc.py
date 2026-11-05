import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2

def get_edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - (i * (i + 1)) // 2 + (j - i - 1)

# Precompute cycles
cycles = []
for v1 in range(N):
    for v2 in range(v1 + 1, N):
        for v3 in range(v2 + 1, N):
            for v4 in range(v3 + 1, N):
                cycles.append([
                    get_edge_idx(v1, v2),
                    get_edge_idx(v2, v3),
                    get_edge_idx(v3, v4),
                    get_edge_idx(v4, v1)
                ])
                cycles.append([
                    get_edge_idx(v1, v2),
                    get_edge_idx(v2, v4),
                    get_edge_idx(v4, v3),
                    get_edge_idx(v3, v1)
                ])
                cycles.append([
                    get_edge_idx(v1, v3),
                    get_edge_idx(v3, v2),
                    get_edge_idx(v2, v4),
                    get_edge_idx(v4, v1)
                ])

def is_valid(coloring):
    for cycle in cycles:
        c1 = coloring[cycle[0]]
        c2 = coloring[cycle[1]]
        c3 = coloring[cycle[2]]
        c4 = coloring[cycle[3]]
        if len({c1, c2, c3, c4}) == 4:
            return False
    return True

def count_colors(coloring):
    return len(set(coloring))

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s margin
    
    # Base construction: 15 colors
    coloring = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            coloring[get_edge_idx(i, j)] = i
    
    best_coloring = coloring[:]
    best_count = count_colors(best_coloring)
    
    # Local search
    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        # Pick a random new color
        # Bias towards new colors
        if random.random() < 0.5:
            new_color = random.randint(best_count, 119)
        else:
            new_color = random.randint(0, 119)
            
        old_color = best_coloring[e]
        if old_color == new_color:
            continue
            
        best_coloring[e] = new_color
        if is_valid(best_coloring):
            new_count = count_colors(best_coloring)
            if new_count >= best_count:
                best_count = new_count
            else:
                # Revert if worse
                best_coloring[e] = old_color
        else:
            best_coloring[e] = old_color
            
    return best_coloring
