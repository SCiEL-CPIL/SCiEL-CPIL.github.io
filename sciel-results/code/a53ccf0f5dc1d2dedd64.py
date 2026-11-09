"""Construct a dense edge-coloured graph on 20 vertices with rainbow 4-cycles."""
import time
import itertools
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    n = 20
    num_colors = 20
    
    # 1. Generate all possible edges in K_20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # 2. Assign algebraic coloring: c(u, v) = (u + v) % 20
    # This coloring ensures adjacent edges in a 4-cycle have different colors.
    # We only need to worry about opposite edges having the same color.
    color_map = {}
    for u, v in all_edges:
        color_map[(u, v)] = (u + v) % num_colors

    # 3. Precompute all 4-cycles in K_n
    # A 4-cycle is defined by 4 distinct vertices a, b, c, d such that 
    # (a,b), (b,c), (c,d), (d,a) are edges.
    # In a simple graph K_n, any set of 4 vertices forms 3 distinct 4-cycles (up to rotation/reflection).
    # However, for checking rainbow property, we just need to check the set of edges.
    # Let's iterate through all combinations of 4 vertices.
    
    def get_4cycle_edges(a, b, c, d):
        """
        Returns the 3 distinct sets of 4 edges forming 4-cycles on vertices {a,b,c,d}.
        Cycle 1: a-b-c-d-a -> edges (a,b), (b,c), (c,d), (d,a)
        Cycle 2: a-b-d-c-a -> edges (a,b), (b,d), (d,c), (c,a)
        Cycle 3: a-c-b-d-a -> edges (a,c), (c,b), (b,d), (d,a)
        """
        e1 = {(min(a,b), max(a,b)), (min(b,c), max(b,c)), (min(c,d), max(c,d)), (min(d,a), max(d,a))}
        e2 = {(min(a,b), max(a,b)), (min(b,d), max(b,d)), (min(c,d), max(c,d)), (min(a,c), max(a,c))}
        e3 = {(min(a,c), max(a,c)), (min(b,c), max(b,c)), (min(b,d), max(b,d)), (min(a,d), max(a,d))}
        return [e1, e2, e3]

    # 4. Identify all 4-cycles that violate the rainbow condition
    violating_cycles = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        cycles = get_4cycle_edges(a, b, c, d)
        for cycle_edges in cycles:
            # Check if all 4 edges have distinct colors
            colors = [color_map[e] for e in cycle_edges]
            if len(set(colors)) < 4:
                # This cycle is not rainbow
                violating_cycles.append(cycle_edges)

    # 5. Greedy removal of edges to resolve violations
    # We want to remove the minimum number of edges such that no violating cycle remains.
    # This is a Hitting Set problem.
    
    # Count frequency of each edge in violating cycles
    edge_violation_count = {e: 0 for e in all_edges}
    for cycle in violating_cycles:
        for e in cycle:
            edge_violation_count[e] += 1

    # Active edges
    active_edges = set(all_edges)
    
    # List of violating cycles that are still "active" (all their edges are present)
    # We will update this list as we remove edges.
    # For efficiency, we can just re-check or maintain a counter.
    # Given n=20, the number of 4-cycles is C(20,4)*3 = 4845 * 3 = 14535.
    # This is small enough to re-scan or use a simple list.
    
    current_violating = list(violating_cycles)
    
    while current_violating:
        if time.time() > deadline:
            break
        
        # Pick a violating cycle. 
        # Strategy: Pick the one with the highest "pressure" or just the first one.
        # To be smarter, pick a cycle, then pick the edge in it with the highest global violation count.
        
        # We can iterate through current_violating and find the "best" edge to remove.
        # But scanning all cycles every time is O(V * E).
        # Let's pick a random violating cycle to avoid bias, or the first one.
        # Actually, a better heuristic: Find the edge that appears in the most *currently active* violating cycles.
        
        # Since the list is small, let's just pick the first violating cycle, 
        # then among its 4 edges, pick the one that appears in the most other violating cycles (from the current list).
        
        # To make it efficient, we can just pick the first cycle, 
        # and remove the edge in that cycle with the highest count in the *original* count? 
        # No, counts change.
        
        # Let's do a simple pass:
        # For each edge in the first violating cycle, count how many cycles in current_violating contain it.
        # This is O(4 * len(current_violating)).
        
        sample_cycle = current_violating[0]
        
        best_edge_to_remove = None
        max_hits = -1
        
        for e in sample_cycle:
            if e not in active_edges:
                continue
            hits = 0
            for c in current_violating:
                if e in c:
                    hits += 1
            if hits > max_hits:
                max_hits = hits
                best_edge_to_remove = e
        
        if best_edge_to_remove is None:
            # Should not happen if sample_cycle is in current_violating
            break
            
        # Remove the edge
        active_edges.discard(best_edge_to_remove)
        
        # Update current_violating: remove any cycle that contains best_edge_to_remove
        # We can do this by filtering the list.
        new_violating = []
        for c in current_violating:
            if best_edge_to_remove not in c:
                new_violating.append(c)
        current_violating = new_violating

    # 6. Construct the result
    result = []
    for u, v in active_edges:
        result.append([u, v, color_map[(u, v)]])
        
    # Optional: Local search to see if we can add back any edges?
    # With the algebraic coloring, adding back an edge might create new violations.
    # Given the time budget, the greedy removal is likely sufficient for a high score.
    # The resulting graph is a subgraph of K_20.
    
    return result
# EVOLVE-BLOCK-END
