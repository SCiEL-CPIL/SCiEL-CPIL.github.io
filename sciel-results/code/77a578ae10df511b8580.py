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
                other_set = set(other_colors)
                if len(other_set) == 3 and c not in other_set:
                    valid = False
                    break
            if valid:
                colors[e] = c
                used_colors.add(c)
                found = True
                break
    
    best_colors = colors[:]
    best_count = len(set(colors))
    
    # Local search: systematically try to improve
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        used_colors = set(colors)
        
        # Try each edge, try all colors to find one that increases color count
        for e in edge_order:
            if time.time() > deadline:
                break
            cur = colors[e]
            # Try new colors first (not in used_colors), then existing
            for nc in range(NUM_EDGES):
                if nc == cur:
                    continue
                valid = True
                for cycle in cycles_by_edge[e]:
                    others = [colors[oe] for oe in cycle if oe != e]
                    other_set = set(others)
                    if len(other_set) == 3 and nc not in other_set:
                        valid = False
                        break
                if valid:
                    # Check if this increases distinct color count
                    if nc not in used_colors:
                        # This will definitely increase count
                        colors[e] = nc
                        used_colors.add(nc)
                        improved = True
                        break
                    else:
                        # Same color count, but might enable future improvements
                        # Skip for now to focus on count-increasing moves
                        pass
        
        # Try swap moves: exchange colors between two edges to potentially free up a color
        if not improved:
            for e1 in edge_order:
                if time.time() > deadline:
                    break
                c1 = colors[e1]
                for e2 in edge_order:
                    if e2 == e1:
                        continue
                    c2 = colors[e2]
                    if c1 == c2:
                        continue
                    # Try swapping: e1 gets c2, e2 gets c1
                    # This is valid if both assignments are valid
                    valid1 = True
                    for cycle in cycles_by_edge[e1]:
                        others = [colors[oe] for oe in cycle if oe != e1]
                        if oe == e2:
                            others = [x if x != c1 else c2 for x in others]
                        other_set = set(others)
                        if len(other_set) == 3 and c2 not in other_set:
                            valid1 = False
                            break
                    if not valid1:
                        continue
                    valid2 = True
                    for cycle in cycles_by_edge[e2]:
                        others = [colors[oe] for oe in cycle if oe != e2]
                        if oe == e1:
                            others = [x if x != c2 else c1 for x in others]
                        other_set = set(others)
                        if len(other_set) == 3 and c1 not in other_set:
                            valid2 = False
                            break
                    if not valid2:
                        continue
                    # Check if swap increases distinct colors
                    # After swap: e1 has c2, e2 has c1
                    # If c1 is used elsewhere and c2 is used elsewhere, no change
                    # If c1 was only on e1 and c2 only on e2, no change
                    # If c1 was only on e1 and c2 used elsewhere, we lose c1
                    # If c2 was only on e2 and c1 used elsewhere, we lose c2
                    # So swaps don't directly increase count, but can enable future moves
                    # Skip swaps for now
                    pass
        
        # Update best
        cur_count = len(set(colors))
        if cur_count > best_count:
            best_count = cur_count
            best_colors = colors[:]
        
        if not improved:
            # Stronger perturbation: random restart of a subset of edges
            num_perturb = rng.randint(5, 15)
            for _ in range(num_perturb):
                if time.time() > deadline:
                    break
                e = rng.randint(0, NUM_EDGES - 1)
                nc = rng.randint(0, NUM_EDGES - 1)
                if nc == colors[e]:
                    continue
                valid = True
                for cycle in cycles_by_edge[e]:
                    others = [colors[oe] for oe in cycle if oe != e]
                    other_set = set(others)
                    if len(other_set) == 3 and nc not in other_set:
                        valid = False
                        break
                if valid:
                    colors[e] = nc
                    improved = True
    
    return best_colors
