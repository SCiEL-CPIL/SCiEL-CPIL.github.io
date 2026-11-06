"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    n = 20
    num_colors = 20

    # Precompute all 4-cycles
    cycles = []
    edge_to_cycles = {}
    
    def get_edge(u, v):
        if u > v:
            u, v = v, u
        return (u, v)

    for verts in itertools.combinations(range(n), 4):
        a, b, c, d = verts
        # 3 distinct 4-cycles
        # 1. a-b-c-d-a
        e1 = get_edge(a, b); e2 = get_edge(b, c); e3 = get_edge(c, d); e4 = get_edge(d, a)
        cyc = (e1, e2, e3, e4)
        cycles.append(cyc)
        idx = len(cycles) - 1
        for e in cyc:
            if e not in edge_to_cycles: edge_to_cycles[e] = []
            edge_to_cycles[e].append(idx)

        # 2. a-b-d-c-a
        e1 = get_edge(a, b); e2 = get_edge(b, d); e3 = get_edge(d, c); e4 = get_edge(c, a)
        cyc = (e1, e2, e3, e4)
        cycles.append(cyc)
        idx = len(cycles) - 1
        for e in cyc:
            if e not in edge_to_cycles: edge_to_cycles[e] = []
            edge_to_cycles[e].append(idx)

        # 3. a-c-b-d-a
        e1 = get_edge(a, c); e2 = get_edge(c, b); e3 = get_edge(b, d); e4 = get_edge(d, a)
        cyc = (e1, e2, e3, e4)
        cycles.append(cyc)
        idx = len(cycles) - 1
        for e in cyc:
            if e not in edge_to_cycles: edge_to_cycles[e] = []
            edge_to_cycles[e].append(idx)

    num_cycles = len(cycles)
    all_possible_edges = [get_edge(i, j) for i in range(n) for j in range(i+1, n)]
    
    # State
    edge_present = set()
    edge_color = {}
    cycle_active = [False] * num_cycles
    cycle_conflict = [False] * num_cycles
    conflict_count = 0
    num_edges = 0

    def update_cycle(idx):
        nonlocal conflict_count
        cyc = cycles[idx]
        active = all(e in edge_present for e in cyc)
        was_active = cycle_active[idx]
        was_conflict = cycle_conflict[idx]
        
        cycle_active[idx] = active
        if active:
            cols = [edge_color[e] for e in cyc]
            is_conflict = len(set(cols)) < 4
            cycle_conflict[idx] = is_conflict
        else:
            cycle_conflict[idx] = False
            
        if was_conflict and not cycle_conflict[idx]:
            conflict_count -= 1
        elif not was_conflict and cycle_conflict[idx]:
            conflict_count += 1

    def init_state():
        nonlocal edge_present, edge_color, cycle_active, cycle_conflict, conflict_count, num_edges
        edge_present = set()
        edge_color = {}
        cycle_active = [False] * num_cycles
        cycle_conflict = [False] * num_cycles
        conflict_count = 0
        num_edges = 0

    def add_edge(e, c):
        nonlocal num_edges
        if e in edge_present:
            return False
        for idx in edge_to_cycles.get(e, []):
            cyc = cycles[idx]
            others_present = True
            for oe in cyc:
                if oe != e and oe not in edge_present:
                    others_present = False
                    break
            if not others_present:
                continue
            cols = [c]
            for oe in cyc:
                if oe != e:
                    cols.append(edge_color[oe])
            if len(set(cols)) < 4:
                return False
        
        edge_present.add(e)
        edge_color[e] = c
        num_edges += 1
        for idx in edge_to_cycles.get(e, []):
            update_cycle(idx)
        return True

    def remove_edge(e):
        nonlocal num_edges
        if e not in edge_present:
            return
        edge_present.remove(e)
        del edge_color[e]
        num_edges -= 1
        for idx in edge_to_cycles.get(e, []):
            update_cycle(idx)

    def recolor_edge(e, new_c):
        if e not in edge_present:
            return False
        old_c = edge_color[e]
        if old_c == new_c:
            return True
        
        for idx in edge_to_cycles.get(e, []):
            if not cycle_active[idx]:
                continue
            cyc = cycles[idx]
            cols = [new_c]
            for oe in cyc:
                if oe != e:
                    cols.append(edge_color[oe])
            if len(set(cols)) < 4:
                return False
                
        edge_color[e] = new_c
        for idx in edge_to_cycles.get(e, []):
            update_cycle(idx)
        return True

    best_score = 0
    best_edges = set()
    best_colors = {}
    
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        
        random.seed(42 + restart)
        init_state()
        
        # Phase 1: Greedy Add
        random.shuffle(all_possible_edges)
        for e in all_possible_edges:
            if time.time() > deadline - 5:
                break
            # Try all colors
            for c in range(num_colors):
                if add_edge(e, c):
                    break
        
        # Phase 2: Local Search (Recolor/Remove/Add)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try to add more edges
            random.shuffle(all_possible_edges)
            for e in all_possible_edges:
                if e in edge_present:
                    continue
                for c in range(num_colors):
                    if add_edge(e, c):
                        improved = True
                        break
                if time.time() > deadline - 2:
                    break
            
            # If stuck, try recoloring
            if not improved and edge_present:
                # Try random recolors
                for _ in range(100):
                    e = random.choice(list(edge_present))
                    new_c = random.randint(0, num_colors - 1)
                    if recolor_edge(e, new_c):
                        # Try to add an edge after recoloring
                        random.shuffle(all_possible_edges)
                        for e2 in all_possible_edges:
                            if e2 in edge_present:
                                continue
                            for c2 in range(num_colors):
                                if add_edge(e2, c2):
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break

        if num_edges > best_score:
            best_score = num_edges
            best_edges = set(edge_present)
            best_colors = dict(edge_color)

    result = []
    for e in best_edges:
        u, v = e
        c = best_colors[e]
        result.append([u, v, c])
        
    return result
# EVOLVE-BLOCK-END
