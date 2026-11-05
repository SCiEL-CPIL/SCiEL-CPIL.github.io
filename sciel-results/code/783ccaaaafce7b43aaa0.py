import time
import random

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
    
    # Precompute for each edge, the other edges in each cycle
    # cycle_others[e] = list of (other_edge1, other_edge2, other_edge3) for each cycle containing e
    cycle_others = [[] for _ in range(NUM_EDGES)]
    for e in range(NUM_EDGES):
        for cycle in cycles_by_edge[e]:
            others = [oe for oe in cycle if oe != e]
            cycle_others[e].append(tuple(others))
    
    # Greedy initialization: assign colors to maximize distinct colors
    colors = [0] * NUM_EDGES
    used_colors = set([0])
    
    # Process edges in order of constraint tightness (most constrained first)
    edge_order = sorted(range(NUM_EDGES), key=lambda e: len(cycles_by_edge[e]), reverse=True)
    
    for e in edge_order:
        # Try to find a new color that is valid
        for c in range(1, NUM_EDGES):
            if c in used_colors:
                continue
            valid = True
            for others in cycle_others[e]:
                oc = [colors[oe] for oe in others]
                if len(set(oc)) == 3 and c not in oc:
                    valid = False
                    break
            if valid:
                colors[e] = c
                used_colors.add(c)
                break
    
    best_colors = colors[:]
    best_count = len(set(colors))
    
    # Local search with efficient tracking
    rng = random.Random(42)
    
    # Helper: check if assigning color nc to edge e is valid
    def is_valid_assign(e, nc):
        for others in cycle_others[e]:
            oc = [colors[oe] for oe in others]
            if len(set(oc)) == 3 and nc not in oc:
                return False
        return True
    
    # Helper: try to assign a new color to edge e
    def try_new_color(e, used_set):
        cur = colors[e]
        for nc in range(NUM_EDGES):
            if nc == cur or nc in used_set:
                continue
            if is_valid_assign(e, nc):
                return nc
        return -1
    
    while time.time() < deadline:
        improved = False
        used_colors = set(colors)
        
        # Phase 1: Try to add new colors by recoloring edges
        for e in edge_order:
            if time.time() > deadline:
                break
            nc = try_new_color(e, used_colors)
            if nc >= 0:
                colors[e] = nc
                used_colors.add(nc)
                improved = True
        
        # Phase 2: Try 2-opt moves
        if not improved:
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            for e1 in edge_order:
                if time.time() > deadline:
                    break
                c1 = colors[e1]
                if color_counts.get(c1, 0) != 1:
                    continue
                for e2 in edge_order:
                    if e2 == e1:
                        continue
                    c2 = colors[e2]
                    if c2 == c1:
                        continue
                    
                    # Check if e1 can take color c2 (with e2 temporarily changed)
                    valid1 = True
                    for others in cycle_others[e1]:
                        oc = [colors[oe] for oe in others]
                        # If e2 is in this cycle, its color would be c1 instead of c2
                        if e2 in others:
                            oc = [c1 if x == c2 else x for x in oc]
                        if len(set(oc)) == 3 and c2 not in oc:
                            valid1 = False
                            break
                    
                    if not valid1:
                        continue
                    
                    # Try new color for e2
                    for nc in range(NUM_EDGES):
                        if nc == c2 or nc == c1 or nc in used_colors:
                            continue
                        valid2 = True
                        for others in cycle_others[e2]:
                            oc = [colors[oe] for oe in others]
                            if e1 in others:
                                oc = [c1 if x == c2 else x for x in oc]
                            if len(set(oc)) == 3 and nc not in oc:
                                valid2 = False
                                break
                        if valid2:
                            colors[e1] = c2
                            colors[e2] = nc
                            used_colors.discard(c1)
                            used_colors.add(nc)
                            improved = True
                            break
                    if improved:
                        break
        
        cur_count = len(set(colors))
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        if not improved:
            # Perturbation
            num_perturb = rng.randint(3, 8)
            for _ in range(num_perturb):
                if time.time() > deadline:
                    break
                e = rng.randint(0, NUM_EDGES - 1)
                nc = rng.randint(0, NUM_EDGES - 1)
                if nc == colors[e]:
                    continue
                if is_valid_assign(e, nc):
                    colors[e] = nc
                    improved = True
    
    return best_colors
