import time
from itertools import combinations
from collections import defaultdict

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return True
    
    # Precompute for each vertex, which edges it belongs to
    vertex_edges = defaultdict(list)
    for i, edge in enumerate(edges):
        for v in edge:
            vertex_edges[v].append(i)
    
    # Use bitmasks for efficiency
    # colors: -1 unassigned, 0, 1
    colors = [-1] * n
    count_0 = [0] * len(edges)
    count_1 = [0] * len(edges)
    
    def assign(idx, color, c, c0, c1, trail):
        if c[idx] != -1:
            return c[idx] == color
        c[idx] = color
        trail.append((idx, color))
        for e_idx in vertex_edges[idx]:
            if color == 0:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    return False
            else:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    return False
            
            # Unit propagation
            if c0[e_idx] == 4:
                for v in edges[e_idx]:
                    if c[v] == -1:
                        if not assign(v, 1, c, c0, c1, trail):
                            return False
            elif c1[e_idx] == 4:
                for v in edges[e_idx]:
                    if c[v] == -1:
                        if not assign(v, 0, c, c0, c1, trail):
                            return False
        return True
    
    def undo_trail(trail, c, c0, c1):
        for idx, color in reversed(trail):
            c[idx] = -1
            for e_idx in vertex_edges[idx]:
                if color == 0:
                    c0[e_idx] -= 1
                else:
                    c1[e_idx] -= 1
    
    def backtrack(c, c0, c1):
        # Find most constrained unassigned vertex
        idx = -1
        max_constrained = -1
        for i in range(n):
            if c[i] == -1:
                # Count how many edges are close to being monochromatic
                constrained = 0
                for e_idx in vertex_edges[i]:
                    if c0[e_idx] >= 3 or c1[e_idx] >= 3:
                        constrained += 1
                if constrained > max_constrained:
                    max_constrained = constrained
                    idx = i
        
        if idx == -1:
            return True  # All assigned, no contradiction
        
        # Try 0
        trail = []
        if assign(idx, 0, c, c0, c1, trail):
            if backtrack(c, c0, c1):
                return True
            undo_trail(trail, c, c0, c1)
        
        # Try 1
        trail = []
        if assign(idx, 1, c, c0, c1, trail):
            if backtrack(c, c0, c1):
                return True
            undo_trail(trail, c, c0, c1)
        
        return False
    
    return backtrack([-1]*n, [0]*len(edges), [0]*len(edges))

def construct():
    """
    Find a minimum non-2-colorable 5-uniform hypergraph.
    Uses branch-and-bound with ILP-inspired approach:
    - Enumerate all 2-colorings of vertices
    - Each edge "hits" colorings where it becomes monochromatic
    - Find minimum edge set that hits all colorings
    """
    start_time = time.time()
    time_limit = 110  # Leave buffer for 120s limit
    
    # Try different n values
    best_solution = None
    best_edge_count = 200
    
    for n in range(5, 14):
        if time.time() - start_time > time_limit:
            break
        
        all_possible_edges = list(combinations(range(n), 5))
        num_edges_total = len(all_possible_edges)
        
        if num_edges_total > 500:
            continue  # Too many edges to enumerate colorings efficiently
        
        # Generate all 2^n colorings (represent as bitmask)
        # A coloring is represented by a bitmask where bit i is 1 if vertex i is color 1
        all_colorings = []
        for mask in range(1 << n):
            # Check if this coloring is valid (not all same color for any edge we might pick)
            # Actually, we need to track which edges are monochromatic under this coloring
            all_colorings.append(mask)
        
        # For each edge, determine which colorings make it monochromatic
        # An edge {v1,v2,v3,v4,v5} is monochromatic under coloring mask if:
        # All bits are 0 (all color 0) OR all bits are 1 (all color 1)
        edge_hits = [set() for _ in range(num_edges_total)]
        
        for i, edge in enumerate(all_possible_edges):
            mask_0 = 0  # All vertices color 0
            mask_1 = 0  # All vertices color 1
            for v in edge:
                mask_1 |= (1 << v)
            
            # Colorings where this edge is all 0s: bits at positions in edge are all 0
            # This means mask & mask_1 == 0
            # Colorings where this edge is all 1s: mask & mask_1 == mask_1
            
            for coloring in all_colorings:
                if (coloring & mask_1) == 0 or (coloring & mask_1) == mask_1:
                    edge_hits[i].add(coloring)
        
        # Now we want to find minimum subset of edges such that every coloring is hit by at least one edge
        # This is a set cover problem
        # Use branch-and-bound
        
        # Sort edges by number of colorings they hit (descending) - greedy heuristic
        edge_order = sorted(range(num_edges_total), key=lambda i: len(edge_hits[i]), reverse=True)
        
        # Branch and bound
        selected = []
        covered = set()
        
        def bnb(idx_in_order, current_covered, selected_edges, depth=0):
            nonlocal best_solution, best_edge_count
            
            if time.time() - start_time > time_limit:
                return
            
            # Pruning: if we've already selected more than best, stop
            if len(selected_edges) >= best_edge_count:
                return
            
            # If all colorings are covered, we have a solution
            if current_covered == set(all_colorings):
                if len(selected_edges) < best_edge_count:
                    best_edge_count = len(selected_edges)
                    best_solution = [list(all_possible_edges[i]) for i in selected_edges]
                return
            
            # If no more edges to consider, fail
            if idx_in_order >= num_edges_total:
                return
            
            # Lower bound: estimate minimum additional edges needed
            # Simple bound: uncovered colorings / max hits per edge
            uncovered_count = len(set(all_colorings) - current_covered)
            if uncovered_count == 0:
                if len(selected_edges) < best_edge_count:
                    best_edge_count = len(selected_edges)
                    best_solution = [list(all_possible_edges[i]) for i in selected_edges]
                return
            
            # Greedy lower bound
            remaining_edges = edge_order[idx_in_order:]
            if not remaining_edges:
                return
            
            max_hits = max(len(edge_hits[e] - current_covered) for e in remaining_edges) if remaining_edges else 0
            if max_hits == 0:
                return
            
            lb = len(selected_edges) + (uncovered_count + max_hits - 1) // max_hits
            if lb >= best_edge_count:
                return
            
            # Branch: include or exclude edge at idx_in_order
            edge_idx = edge_order[idx_in_order]
            
            # Option 1: Include this edge
            new_covered = current_covered | edge_hits[edge_idx]
            new_selected = selected_edges + [edge_idx]
            bnb(idx_in_order + 1, new_covered, new_selected, depth + 1)
            
            # Option 2: Exclude this edge
            bnb(idx_in_order + 1, current_covered, selected_edges, depth + 1)
        
        # Start branch and bound
        bnb(0, set(), [])
        
        # If we found a solution, verify it
        if best_solution and len(best_solution) < 200:
            if not is_2_colorable(n, best_solution):
                return {'n': n, 'edges': best_solution}
        
        # If no solution found for this n, try next n
        best_solution = None
        best_edge_count = 200
    
    # Fallback: return a known valid construction
    # Use a dense hypergraph on 9 vertices
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)][:126]}

# EVOLVE-BLOCK-END
