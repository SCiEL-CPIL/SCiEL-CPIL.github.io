import random
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit (evaluator kills at 150s)
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles and their edge sets
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    for u, v, w, x in itertools.combinations(range(n), 4):
        # Three distinct 4-cycles on 4 vertices (edge sets)
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
    
    num_cycles = len(all_cycles)
    
    def get_conflict_edges(edge_colors):
        """Return set of edges involved in any non-rainbow 4-cycle."""
        conflict = set()
        for idx, cycle in enumerate(all_cycles):
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict.add(e)
        return conflict
    
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
    
    def find_valid_color(e, edge_colors, color_order=None):
        """Find a valid color for edge e, or None."""
        if color_order is None:
            color_order = list(range(num_colors))
            random.shuffle(color_order)
        for c in color_order:
            if can_add_edge(e, c, edge_colors):
                return c
        return None
    
    def count_conflicts(edge_colors):
        """Count number of non-rainbow 4-cycles."""
        count = 0
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    count += 1
        return count
    
    def objective(edge_colors):
        """Objective: maximize edges, minimize conflicts. Higher is better."""
        return len(edge_colors) * 10000 - count_conflicts(edge_colors)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # === Phase 1: Construct initial solution using structured approach ===
    
    # Strategy: Start with a complete graph and greedily assign colors,
    # removing edges that can't be colored validly.
    
    def construct_initial():
        """Build a valid solution by greedily adding edges with valid colors."""
        edge_colors = {}
        
        # Order edges to process: prioritize edges that appear in fewer cycles first
        # This gives more flexibility for later edges
        edges_by_constraint = sorted(all_edges, key=lambda e: len(edge_to_cycles[e]))
        
        # Try multiple random orderings and keep the best
        best = {}
        
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            
            ec = {}
            # Process edges in a random-ish order but biased toward less-constrained
            shuffled_edges = list(edges_by_constraint)
            random.shuffle(shuffled_edges)
            
            for e in shuffled_edges:
                c = find_valid_color(e, ec)
                if c is not None:
                    ec[e] = c
            
            if len(ec) > len(best):
                best = dict(ec)
        
        return best
    
    # Also try starting from random dense graph and repairing
    def construct_repair():
        """Start with random dense graph, remove conflicting edges."""
        edge_colors = {}
        p_init = 0.8
        for e in all_edges:
            if random.random() < p_init:
                edge_colors[e] = random.randint(0, num_colors - 1)
        
        # Repair: iteratively remove edges in most conflicts
        for _ in range(500):
            if time.time() > deadline - 30:
                break
            conflicts = get_conflict_edges(edge_colors)
            if not conflicts:
                break
            # Remove edge in most conflicting cycles
            e_to_remove = max(conflicts, key=lambda e: len(edge_to_cycles[e]))
            del edge_colors[e_to_remove]
        
        return edge_colors
    
    # Try both approaches and keep the best
    sol1 = construct_initial()
    sol2 = construct_repair()
    
    edge_colors = sol1 if len(sol1) >= len(sol2) else sol2
    
    # Final repair to ensure validity
    for _ in range(200):
        if time.time() > deadline - 20:
            break
        conflicts = get_conflict_edges(edge_colors)
        if not conflicts:
            break
        e_to_remove = max(conflicts, key=lambda e: len(edge_to_cycles[e]))
        del edge_colors[e_to_remove]
    
    best_count = len(edge_colors)
    best_state = dict(edge_colors)
    
    def save_if_better():
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_state = dict(edge_colors)
    
    # === Phase 2: Local search to maximize edges ===
    
    # Use a more systematic approach:
    # 1. Try to add every missing edge with best possible color
    # 2. Recolor edges to free up color constraints
    # 3. Swap edges to improve packing
    
    while time.time() < deadline - 5:
        missing = [e for e in all_edges if e not in edge_colors]
        
        # Move 1: Add a missing edge (try all, keep best)
        if missing:
            # Prioritize edges with fewer cycle constraints
            missing_sorted = sorted(missing, key=lambda e: len(edge_to_cycles[e]))
            added = False
            for e_add in missing_sorted:
                if time.time() > deadline - 5:
                    break
                c = find_valid_color(e_add, edge_colors)
                if c is not None:
                    edge_colors[e_add] = c
                    save_if_better()
                    added = True
                    break
            
            if added:
                continue
        
        # Move 2: Recolor an edge to enable adding another
        if edge_colors and missing:
            # Find an edge whose recoloring would help
            e_recolor = random.choice(list(edge_colors.keys()))
            old_c = edge_colors[e_recolor]
            
            # Try all colors for this edge
            for new_c in range(num_colors):
                if new_c == old_c:
                    continue
                del edge_colors[e_recolor]
                if can_add_edge(e_recolor, new_c, edge_colors):
                    edge_colors[e_recolor] = new_c
                    break
                else:
                    edge_colors[e_recolor] = old_c
                    break
        
        # Move 3: Swap - remove one edge, add another (net zero or +1)
        if edge_colors and missing and random.random() < 0.5:
            e_remove = random.choice(list(edge_colors.keys()))
            e_add = random.choice(missing)
            
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
        
        # Move 4: Double swap - remove 2, add 2
        if random.random() < 0.2 and len(edge_colors) >= 2 and len(missing) >= 2:
            e1, e2 = random.sample(list(edge_colors.keys()), 2)
            e3, e4 = random.sample(missing, 2)
            c1 = edge_colors[e1]
            c2 = edge_colors[e2]
            del edge_colors[e1]
            del edge_colors[e2]
            c3 = find_valid_color(e3, edge_colors)
            c4 = find_valid_color(e4, edge_colors) if c3 is not None else None
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
        
        # Move 5: Recolor multiple edges to free up constraints
        if random.random() < 0.3 and len(edge_colors) >= 3:
            # Pick 3 edges and try to recolor them
            edges_to_recolor = random.sample(list(edge_colors.keys()), min(3, len(edge_colors)))
            old_colors = {e: edge_colors[e] for e in edges_to_recolor}
            
            # Remove all
            for e in edges_to_recolor:
                del edge_colors[e]
            
            # Try to re-add with new colors
            success = True
            new_colors = {}
            for e in edges_to_recolor:
                c = find_valid_color(e, edge_colors)
                if c is None:
                    success = False
                    break
                new_colors[e] = c
                edge_colors[e] = c
            
            if not success:
                # Revert
                for e, c in old_colors.items():
                    edge_colors[e] = c
    
    # Use the best state found
    edge_colors = best_state
    
    # Final cleanup: ensure validity
    for _ in range(500):
        if time.time() > deadline:
            break
        conflicts = get_conflict_edges(edge_colors)
        if not conflicts:
            break
        e_to_remove = max(conflicts, key=lambda e: len(edge_to_cycles[e]))
        del edge_colors[e_to_remove]
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    return result
