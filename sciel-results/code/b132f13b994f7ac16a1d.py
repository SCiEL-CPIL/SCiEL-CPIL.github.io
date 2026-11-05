import time
import random
import math

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    e1 = (edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a))
                    cycles.append(e1)
                    e2 = (edge_index(a,b), edge_index(b,d), edge_index(d,c), edge_index(c,a))
                    cycles.append(e2)
                    e3 = (edge_index(a,c), edge_index(c,b), edge_index(b,d), edge_index(d,a))
                    cycles.append(e3)
    return cycles

def construct():
    deadline = time.time() + 55
    
    cycles = build_four_cycles()
    cycles_by_edge = [[] for _ in range(NUM_EDGES)]
    for cycle in cycles:
        for e in cycle:
            cycles_by_edge[e].append(cycle)
    
    # Greedy initialization: assign colors to maximize distinct colors
    colors = [0] * NUM_EDGES
    used_colors = set([0])
    
    # Process edges in order of constraint tightness (most constrained first)
    edge_order = sorted(range(NUM_EDGES), key=lambda e: len(cycles_by_edge[e]), reverse=True)
    
    for e in edge_order:
        # Try to find a new color that is valid
        found = False
        for c in range(1, NUM_EDGES):
            if c in used_colors:
                continue
            valid = True
            for cycle in cycles_by_edge[e]:
                other_colors = [colors[oe] for oe in cycle if oe != e]
                if len(set(other_colors)) == 3 and c not in other_colors:
                    valid = False
                    break
            if valid:
                colors[e] = c
                used_colors.add(c)
                found = True
                break
    
    best_colors = colors[:]
    best_count = len(set(colors))
    
    # Local search with simulated annealing
    rng = random.Random(42)
    temperature = 10.0
    min_temperature = 0.1
    cooling_rate = 0.995
    
    def is_valid_recolor(colors_list, edge, new_color):
        """Check if recoloring edge to new_color maintains validity."""
        for cycle in cycles_by_edge[edge]:
            others = [colors_list[oe] for oe in cycle if oe != edge]
            if len(set(others)) == 3 and new_color not in others:
                return False
        return True
    
    def count_distinct(colors_list):
        return len(set(colors_list))
    
    while time.time() < deadline and temperature > min_temperature:
        improved = False
        used_colors = set(colors)
        
        # Try each edge, try all new colors to find one that increases color count
        for e in edge_order:
            if time.time() > deadline:
                break
            cur = colors[e]
            # Try new colors first (not in used_colors)
            for nc in range(NUM_EDGES):
                if nc == cur or nc in used_colors:
                    continue
                if is_valid_recolor(colors, e, nc):
                    colors[e] = nc
                    used_colors.add(nc)
                    improved = True
                    break
        
        # Try swap moves: exchange colors between two edges
        if not improved:
            for _ in range(50):  # Limited swap attempts per iteration
                if time.time() > deadline:
                    break
                e1 = rng.randint(0, NUM_EDGES - 1)
                e2 = rng.randint(0, NUM_EDGES - 1)
                if e1 == e2:
                    continue
                c1 = colors[e1]
                c2 = colors[e2]
                if c1 == c2:
                    continue
                # Check if swapping is valid
                valid1 = is_valid_recolor(colors, e1, c2)
                if valid1:
                    # Temporarily swap to check e2
                    colors[e1] = c2
                    valid2 = is_valid_recolor(colors, e2, c1)
                    colors[e1] = c1
                    if valid2:
                        colors[e1] = c2
                        colors[e2] = c1
                        improved = True
                        break
        
        # Simulated annealing: accept worse states with probability
        if not improved:
            for _ in range(20):
                if time.time() > deadline:
                    break
                e = rng.randint(0, NUM_EDGES - 1)
                nc = rng.randint(0, NUM_EDGES - 1)
                if nc == colors[e]:
                    continue
                if is_valid_recolor(colors, e, nc):
                    old_count = count_distinct(colors)
                    colors[e] = nc
                    new_count = count_distinct(colors)
                    delta = new_count - old_count
                    # Accept if improves or with probability exp(delta/T)
                    if delta >= 0 or rng.random() < math.exp(delta / temperature):
                        improved = True
                        break
                    else:
                        colors[e] = nc  # Keep the worse state (annealing)
                        break
        
        # Update best
        cur_count = len(set(colors))
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        if not improved:
            # Perturbation: randomly recolor some edges
            num_perturb = rng.randint(3, 8)
            for _ in range(num_perturb):
                if time.time() > deadline:
                    break
                e = rng.randint(0, NUM_EDGES - 1)
                nc = rng.randint(0, NUM_EDGES - 1)
                if nc == colors[e]:
                    continue
                if is_valid_recolor(colors, e, nc):
                    colors[e] = nc
                    improved = True
        
        # Cool down
        temperature *= cooling_rate
    
    # Final polish: try to improve from best_colors
    colors = best_colors[:]
    used_colors = set(colors)
    for e in edge_order:
        if time.time() > deadline - 2:
            break
        cur = colors[e]
        for nc in range(NUM_EDGES):
            if nc == cur or nc in used_colors:
                continue
            if is_valid_recolor(colors, e, nc):
                colors[e] = nc
                used_colors.add(nc)
                break
    
    cur_count = len(set(colors))
    if cur_count > best_count:
        best_colors = colors[:]
        best_count = cur_count
    
    return best_colors
