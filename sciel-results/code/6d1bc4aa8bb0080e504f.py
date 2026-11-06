import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s kill limit
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles and edge-to-cycle incidence
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    for u, v, w, x in itertools.combinations(range(n), 4):
        # Three distinct 4-cycles on 4 vertices
        cycles = [
            [(u, v), (v, w), (w, x), (x, u)],
            [(u, v), (v, x), (x, w), (w, u)],
            [(u, w), (w, v), (v, x), (x, u)],
        ]
        for cycle in cycles:
            norm_cycle = tuple(sorted((min(e), max(e)) for e in cycle))
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    def can_add_edge(e, c, edge_colors):
        """Check if adding edge e with color c keeps all 4-cycles containing e rainbow."""
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_edges = [oe for oe in cycle if oe != e]
            if not all(oe in edge_colors for oe in other_edges):
                continue
            other_colors = [edge_colors[oe] for oe in other_edges]
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True
    
    def find_valid_color(e, edge_colors):
        """Find a valid color for edge e, or None. Prioritize colors with less pressure."""
        colors = list(range(num_colors))
        # Sort colors by how often they appear in cycles containing e (fewer = better)
        color_pressure = defaultdict(int)
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            for oe in cycle:
                if oe != e and oe in edge_colors:
                    color_pressure[edge_colors[oe]] += 1
        colors.sort(key=lambda c: color_pressure.get(c, 0))
        for c in colors:
            if can_add_edge(e, c, edge_colors):
                return c
        return None
    
    def get_conflict_edges(edge_colors):
        conflict = set()
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict.add(e)
        return conflict
    
    # Phase 1: Structured initial solution using a near-complete graph
    # Use a systematic coloring: color = (u * 7 + v * 3) % num_colors for edge (u,v)
    # This tends to avoid color conflicts on 4-cycles
    edge_colors = {}
    for u in range(n):
        for v in range(u + 1, n):
            c = (u * 7 + v * 3) % num_colors
            edge_colors[(u, v)] = c
    
    # Repair: remove edges causing conflicts, prioritizing high-degree edges
    max_repair = 2000
    for _ in range(max_repair):
        if time.time() > deadline:
            break
        conflicts = get_conflict_edges(edge_colors)
        if not conflicts:
            break
        # Remove edge in most conflict cycles
        e_to_remove = max(conflicts, key=lambda e: len(edge_to_cycles[e]))
        del edge_colors[e_to_remove]
    
    # Phase 2: Local search with O(1) delta evaluation for recoloring
    best_count = len(edge_colors)
    best_state = dict(edge_colors)
    
    def save_if_better():
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_state = dict(edge_colors)
    
    # Precompute for each edge, the set of other edges in each cycle containing it
    # cycle_neighbors[edge_idx] = list of (other_edge, cycle_idx)
    cycle_neighbors = defaultdict(list)
    for idx, cycle in enumerate(all_cycles):
        for e in cycle:
            for oe in cycle:
                if oe != e:
                    cycle_neighbors[e].append((oe, idx))
    
    # Count how many 4-cycles each edge participates in
    edge_cycle_count = {e: len(edge_to_cycles[e]) for e in all_edges}
    
    # Fast recolor check: for edge e with new color c, check all cycles containing e
    def can_recolor(e, new_c, edge_colors):
        """O(k) check where k = number of cycles containing e."""
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_edges = [oe for oe in cycle if oe != e]
            if not all(oe in edge_colors for oe in other_edges):
                continue
            other_colors = [edge_colors[oe] for oe in other_edges]
            if new_c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True
    
    # Iterative improvement loop with fast moves
    while time.time() < deadline:
        missing = [e for e in all_edges if e not in edge_colors]
        
        # Move type 1: Add a missing edge (prioritize edges with more available colors)
        if missing:
            # Score missing edges by how many valid colors they have
            scored_missing = []
            for e in missing:
                valid_count = 0
                for c in range(num_colors):
                    if can_add_edge(e, c, edge_colors):
                        valid_count += 1
                        if valid_count >= 3:  # Early exit
                            break
                scored_missing.append((valid_count, e))
            
            if scored_missing and scored_missing[0][0] > 0:
                # Pick from top-scoring edges
                scored_missing.sort(reverse=True)
                top_edges = [e for _, e in scored_missing[:10]]
                e_add = random.choice(top_edges)
                c = find_valid_color(e_add, edge_colors)
                if c is not None:
                    edge_colors[e_add] = c
                    save_if_better()
                    continue
        
        # Move type 2: Recolor an existing edge (fast O(k) check)
        if edge_colors:
            e_recolor = random.choice(list(edge_colors.keys()))
            old_c = edge_colors[e_recolor]
            # Try a few random new colors
            for _ in range(3):
                new_c = random.randint(0, num_colors - 1)
                if new_c != old_c and can_recolor(e_recolor, new_c, edge_colors):
                    edge_colors[e_recolor] = new_c
                    break
        
        # Move type 3: Swap - remove one edge, add another
        if edge_colors and missing and random.random() < 0.4:
            e_remove = random.choice(list(edge_colors.keys()))
            e_add = random.choice(missing)
            old_c = edge_colors[e_remove]
            del edge_colors[e_remove]
            c_add = find_valid_color(e_add, edge_colors)
            if c_add is not None:
                edge_colors[e_add] = c_add
                save_if_better()
                # Try to re-add e_remove
                c_rem = find_valid_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                    save_if_better()
            else:
                # Revert
                c_rem = find_valid_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                else:
                    edge_colors[e_remove] = old_c
        
        # Move type 4: Double swap - remove 2, add 2
        if random.random() < 0.15 and len(edge_colors) >= 2 and len(missing) >= 2:
            e1, e2 = random.sample(list(edge_colors.keys()), 2)
            e3, e4 = random.sample(missing, 2)
            c1 = edge_colors[e1]
            c2 = edge_colors[e2]
            del edge_colors[e1]
            del edge_colors[e2]
            c3 = find_valid_color(e3, edge_colors)
            c4 = find_valid_color(e4, edge_colors)
            if c3 is not None and c4 is not None:
                edge_colors[e3] = c3
                edge_colors[e4] = c4
                save_if_better()
                # Try to re-add e1, e2
                c1_new = find_valid_color(e1, edge_colors)
                c2_new = find_valid_color(e2, edge_colors)
                if c1_new is not None:
                    edge_colors[e1] = c1_new
                if c2_new is not None:
                    edge_colors[e2] = c2_new
                save_if_better()
            else:
                # Revert
                edge_colors[e1] = c1
                edge_colors[e2] = c2
    
    # Use the best state found
    edge_colors = best_state
    
    # Final cleanup: ensure validity, then try to re-add edges
    for _ in range(500):
        if time.time() > deadline:
            break
        conflicts = get_conflict_edges(edge_colors)
        if not conflicts:
            break
        e_to_remove = max(conflicts, key=lambda e: len(edge_to_cycles[e]))
        del edge_colors[e_to_remove]
    
    # Try to re-add missing edges after cleanup
    for _ in range(100):
        if time.time() > deadline:
            break
        missing = [e for e in all_edges if e not in edge_colors]
        if not missing:
            break
        # Try each missing edge
        added = False
        for e in missing:
            c = find_valid_color(e, edge_colors)
            if c is not None:
                edge_colors[e] = c
                added = True
                break
        if not added:
            break
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    return result
