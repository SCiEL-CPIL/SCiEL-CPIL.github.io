import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 85.0

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def construct():
    start = time.time()
    
    # Build edge index mapping
    # Precompute all 4-cycles
    cycles = []
    for a, b, c, d in combinations(range(N), 4):
        c1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        c2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)]
        c3 = [edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)]
        cycles.append(c1)
        cycles.append(c2)
        cycles.append(c3)
    
    # Map each edge to cycles containing it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    
    # Base: star coloring
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i+1, N):
            colors[edge_idx(i,j)] = i
    
    best_colors = colors[:]
    best_count = len(set(best_colors))
    
    def valid_new_color(e, new_c, cur_colors):
        """Check if assigning new_c to edge e keeps all cycles non-rainbow."""
        for ci in edge_to_cycles[e]:
            cyc = cycles[ci]
            other_colors = set()
            for ce in cyc:
                if ce != e:
                    other_colors.add(cur_colors[ce])
            if new_c not in other_colors:
                return False
        return True
    
    # Greedy pass 1: for each edge, try to increase its color
    for e in range(NUM_EDGES):
        if time.time() - start > DEADLINE:
            break
        old_c = colors[e]
        # Try colors from high to low to maximize distinct count
        for new_c in range(119, -1, -1):
            if new_c == old_c:
                continue
            if valid_new_color(e, new_c, colors):
                colors[e] = new_c
                new_count = len(set(colors))
                if new_count > best_count:
                    best_count = new_count
                    best_colors = colors[:]
                break
    
    if time.time() - start > DEADLINE:
        return best_colors
    
    # Multiple greedy passes
    for _ in range(20):
        if time.time() - start > DEADLINE:
            break
        improved = False
        for e in range(NUM_EDGES):
            if time.time() - start > DEADLINE:
                break
            old_c = colors[e]
            for new_c in range(119, -1, -1):
                if new_c == old_c:
                    continue
                if valid_new_color(e, new_c, colors):
                    colors[e] = new_c
                    new_count = len(set(colors))
                    if new_count > best_count:
                        best_count = new_count
                        best_colors = colors[:]
                    improved = True
                    break
        if not improved:
            break
    
    if time.time() - start > DEADLINE:
        return best_colors
    
    # Randomized search
    random.seed(123)
    for _ in range(20000):
        if time.time() - start > DEADLINE:
            break
        e = random.randint(0, NUM_EDGES - 1)
        new_c = random.randint(0, 119)
        if new_c == colors[e]:
            continue
        if valid_new_color(e, new_c, colors):
            old_c = colors[e]
            colors[e] = new_c
            new_count = len(set(colors))
            if new_count > best_count:
                best_count = new_count
                best_colors = colors[:]
            else:
                colors[e] = old_c
    
    return best_colors

