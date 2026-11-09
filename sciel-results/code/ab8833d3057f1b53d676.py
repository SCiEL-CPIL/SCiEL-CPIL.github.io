import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def construct():
    """
    Constructs a coloring of K_16 maximizing distinct colors with no rainbow C4.
    Uses initial Gallai construction + local search with efficient cycle checking.
    """
    deadline = time.time() + 80
    
    # Precompute edge indices
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    # Precompute all 4-cycles and which edges they contain
    all_cycles = []  # list of lists of 4 edge indices
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]  # for each edge, which cycles contain it
    
    cycle_id = 0
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cyc1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        all_cycles.append(cyc1)
        for e in cyc1:
            edge_to_cycles[e].append(cycle_id)
        cycle_id += 1
        # Cycle 2: a-b-d-c-a
        cyc2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
        all_cycles.append(cyc2)
        for e in cyc2:
            edge_to_cycles[e].append(cycle_id)
        cycle_id += 1
        # Cycle 3: a-c-b-d-a
        cyc3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
        all_cycles.append(cyc3)
        for e in cyc3:
            edge_to_cycles[e].append(cycle_id)
        cycle_id += 1
    
    # Build initial Gallai coloring
    colors = [0] * NUM_EDGES
    next_color = [0]
    
    def assign_gallai(vertices, color_offset=0):
        nonlocal colors
        n = len(vertices)
        if n <= 1:
            return
        if n == 2:
            u, v = vertices
            e = edge_idx(u, v)
            colors[e] = next_color[0]
            next_color[0] += 1
            return
        if n == 3:
            u, v, w = vertices
            e1 = edge_idx(u, v)
            e2 = edge_idx(v, w)
            e3 = edge_idx(u, w)
            colors[e1] = next_color[0]
            colors[e2] = next_color[0] + 1
            colors[e3] = next_color[0] + 2
            next_color[0] += 3
            return
        
        SPLIT = {4:3, 5:3, 6:3, 7:3, 8:4, 9:3, 10:4, 11:5, 12:6, 13:6, 14:7, 15:6, 16:7}
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        
        cross_color = next_color[0]
        for u in v1:
            for v in v2:
                e = edge_idx(u, v)
                colors[e] = cross_color
        next_color[0] += 1
        
        assign_gallai(v1)
        assign_gallai(v2)
    
    assign_gallai(list(range(N)))
    
    current = colors[:]
    best = colors[:]
    best_num_colors = len(set(current))
    
    def check_valid_fast(flat, changed_edge=None):
        """Check validity, optionally only checking cycles containing changed_edge."""
        if changed_edge is None:
            for cyc in all_cycles:
                if len(set(flat[e] for e in cyc)) == 4:
                    return False
            return True
        else:
            for cid in edge_to_cycles[changed_edge]:
                cyc = all_cycles[cid]
                if len(set(flat[e] for e in cyc)) == 4:
                    return False
            return True
    
    random.seed(42)
    
    while time.time() < deadline:
        used_colors = set(current)
        available = [c for c in range(120) if c not in used_colors]
        
        if not available:
            break
        
        found_improvement = False
        for attempt in range(2000):
            if time.time() >= deadline:
                break
            
            e = random.randint(0, NUM_EDGES - 1)
            old_c = current[e]
            new_c = random.choice(available)
            if new_c == old_c:
                continue
            
            current[e] = new_c
            if check_valid_fast(current, e):
                new_used = set(current)
                if len(new_used) > best_num_colors:
                    best = current[:]
                    best_num_colors = len(new_used)
                    found_improvement = True
                    available = [c for c in range(120) if c not in new_used]
            else:
                current[e] = old_c
        
        if not found_improvement:
            # Try more aggressive: try all available colors on a random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_c = current[e]
            for new_c in available:
                if time.time() >= deadline:
                    break
                current[e] = new_c
                if check_valid_fast(current, e):
                    new_used = set(current)
                    if len(new_used) > best_num_colors:
                        best = current[:]
                        best_num_colors = len(new_used)
                        found_improvement = True
                        break
            if not found_improvement:
                current[e] = old_c
    
    return best

if __name__ == "__main__":
    res = construct()
    print(f"Number of colors used: {len(set(res))}")
    print(f"First 10 colors: {res[:10]}")
