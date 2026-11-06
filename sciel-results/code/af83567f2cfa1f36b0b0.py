import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 119

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute edge endpoints
EDGE_ENDPOINTS = []
for i in range(N):
    for j in range(i + 1, N):
        EDGE_ENDPOINTS.append((i, j))

# Precompute cycles for each edge
CYCLE_EDGES = [[] for _ in range(NUM_EDGES)]

for idx in range(NUM_EDGES):
    u, v = EDGE_ENDPOINTS[idx]
    others = [k for k in range(N) if k != u and k != v]
    for wi in range(len(others)):
        for xi in range(wi + 1, len(others)):
            w = others[wi]
            x = others[xi]
            e1 = edge_idx(v, w)
            e2 = edge_idx(w, x)
            e3 = edge_idx(x, u)
            CYCLE_EDGES[idx].append((e1, e2, e3))

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial solution: The known 16-color solution
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    colors[edge_idx(13, 15)] = 15
    colors[edge_idx(14, 15)] = 16
    
    used_colors = set(colors)
    color_counts = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
        
    best_colors = colors[:]
    best_count = len(used_colors)
    
    T = 10.0
    cooling = 0.9995
    
    def is_valid_change(e_idx, new_color):
        for (e1, e2, e3) in CYCLE_EDGES[e_idx]:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            if new_color != c1 and new_color != c2 and new_color != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def apply_change(e_idx, new_color):
        old_color = colors[e_idx]
        colors[e_idx] = new_color
        color_counts[old_color] -= 1
        if color_counts[old_color] == 0:
            del color_counts[old_color]
            used_colors.discard(old_color)
        
        if new_color not in color_counts:
            color_counts[new_color] = 1
            used_colors.add(new_color)
        else:
            color_counts[new_color] += 1
        return old_color

    def revert_change(e_idx, old_color, new_color):
        apply_change(e_idx, old_color)

    while time.time() < deadline:
        candidates = []
        if len(used_colors) < MAX_COLOR:
            for e_idx in range(NUM_EDGES):
                c = colors[e_idx]
                if color_counts[c] > 1:
                    candidates.append(e_idx)
        
        if candidates:
            e_idx = random.choice(candidates)
            available_new = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if available_new:
                new_color = random.choice(available_new)
            else:
                new_color = random.randint(0, MAX_COLOR)
        else:
            e_idx = random.randint(0, NUM_EDGES - 1)
            new_color = random.randint(0, MAX_COLOR)
        
        old_color = colors[e_idx]
        if old_color == new_color:
            continue
            
        if is_valid_change(e_idx, new_color):
            old_count = len(used_colors)
            apply_change(e_idx, new_color)
            new_count = len(used_colors)
            delta = new_count - old_count
            
            if delta > 0:
                if new_count > best_count:
                    best_count = new_count
                    best_colors = colors[:]
            elif delta == 0:
                if random.random() >= 0.5:
                    revert_change(e_idx, old_color, new_color)
            else:
                prob = math.exp(-delta / T)
                if random.random() >= prob:
                    revert_change(e_idx, old_color, new_color)
            
            T *= cooling
        else:
            pass

    return best_colors
