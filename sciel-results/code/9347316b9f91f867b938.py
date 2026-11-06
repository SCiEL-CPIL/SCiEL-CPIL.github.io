import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 108.0  # Safety margin for 120s limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    vertices = list(range(n))
    for comb in itertools.combinations(vertices, 4):
        u, v, w, x = comb
        # Three distinct 4-cycles on 4 vertices
        c1 = [(u,v), (v,w), (w,x), (x,u)]
        c2 = [(u,v), (v,x), (x,w), (w,u)]
        c3 = [(u,w), (w,v), (v,x), (x,u)]
        
        for cycle in [c1, c2, c3]:
            norm_cycle = []
            for e in cycle:
                norm_e = (min(e[0], e[1]), max(e[0], e[1]))
                norm_cycle.append(norm_e)
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    # State
    edge_colors = {}
    
    # Start with complete graph K_20
    for u in range(n):
        for v in range(u+1, n):
            edge_colors[(u,v)] = random.randint(0, num_colors - 1)
    
    def get_bad_cycles():
        """Return list of cycle indices where all 4 edges present and colors not all distinct."""
        bad = []
        for i, cycle in enumerate(all_cycles):
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    bad.append(i)
        return bad
    
    # Phase 1: Remove edges to eliminate all bad cycles
    max_removals = 500
    for _ in range(max_removals):
        if time.time() > deadline:
            break
        
        bad = get_bad_cycles()
        if not bad:
            break
        
        # Pick a bad cycle, remove one of its edges
        ci = random.choice(bad)
        cycle = all_cycles[ci]
        e_to_remove = random.choice(cycle)
        del edge_colors[e_to_remove]
    
    # Verify no bad cycles remain
    bad = get_bad_cycles()
    while bad:
        if time.time() > deadline:
            break
        ci = random.choice(bad)
        cycle = all_cycles[ci]
        e_to_remove = random.choice(cycle)
        del edge_colors[e_to_remove]
        bad = get_bad_cycles()
    
    # Phase 2: Try to add back removed edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    current_edges = set(edge_colors.keys())
    missing_edges = [e for e in all_edges if e not in current_edges]
    
    def can_add_with_color(e, c):
        """Check if adding edge e with color c keeps all 4-cycles rainbow."""
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            # Check if all other edges in this cycle exist
            if not all(other_e in edge_colors for other_e in cycle if other_e != e):
                continue
            other_colors = [edge_colors[other_e] for other_e in cycle if other_e != e]
            if len(set(other_colors)) < 3:
                return False
            if c in other_colors:
                return False
        return True
    
    # Try to add missing edges
    for _ in range(200):
        if time.time() > deadline:
            break
        if not missing_edges:
            break
        
        e_add = random.choice(missing_edges)
        colors = list(range(num_colors))
        random.shuffle(colors)
        added = False
        for c in colors:
            if can_add_with_color(e_add, c):
                edge_colors[e_add] = c
                missing_edges.remove(e_add)
                added = True
                break
    
    # Phase 3: Local search - try recoloring to enable adding more edges
    for _ in range(100):
        if time.time() > deadline:
            break
        if not missing_edges:
            break
        
        # Try to add a random missing edge
        e_add = random.choice(missing_edges)
        colors = list(range(num_colors))
        random.shuffle(colors)
        added = False
        for c in colors:
            if can_add_with_color(e_add, c):
                edge_colors[e_add] = c
                missing_edges.remove(e_add)
                added = True
                break
        
        if added:
            continue
        
        # Try recoloring a random existing edge
        if edge_colors:
            e_recolor = random.choice(list(edge_colors.keys()))
            old_color = edge_colors[e_recolor]
            new_color = random.randint(0, num_colors - 1)
            if new_color != old_color:
                del edge_colors[e_recolor]
                if can_add_with_color(e_recolor, new_color):
                    edge_colors[e_recolor] = new_color
                else:
                    edge_colors[e_recolor] = old_color
    
    # Final verification and cleanup
    bad = get_bad_cycles()
    while bad:
        if time.time() > deadline:
            break
        ci = random.choice(bad)
        cycle = all_cycles[ci]
        e_to_remove = random.choice(cycle)
        if e_to_remove in edge_colors:
            del edge_colors[e_to_remove]
        bad = get_bad_cycles()
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
        
    return result
