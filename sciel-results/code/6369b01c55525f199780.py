"""
Minimal 5-uniform hypergraph without Property B.
Strategy: Use a known construction (e.g., the 51-edge example) and then
attempt to remove edges while maintaining non-2-colorability.
We implement a fast backtracking solver with unit propagation for 2-colorability.
"""
import time
import random
from itertools import combinations

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 120.0  # seconds
SAFETY_MARGIN = 5.0  # seconds

def time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

def is_2_colorable(n, edges):
    """
    Check if the hypergraph is 2-colorable.
    Returns True if a proper 2-coloring exists, False otherwise.
    Uses backtracking with unit propagation.
    """
    # Preprocess: remove duplicate vertices in edges? No, they are sets of distinct vertices.
    # We represent the hypergraph as a list of edges, each a frozenset or tuple.
    # For speed, we use bitmasks for vertex sets if n <= 64.
    
    if not edges:
        return True

    # Convert edges to bitmasks for fast intersection checks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)

    # We maintain a set of unassigned vertices and a list of active edges.
    # Unit propagation: If an edge has 4 vertices of the same color, the 5th must be the other color.
    # If an edge has 5 vertices of the same color, it's a contradiction (not 2-colorable).
    
    # State:
    # color: list of length n, -1 for uncolored, 0 or 1 for colored
    # We'll use a recursive backtracking solver.
    
    # To make it faster, we can use a simple heuristic: pick the vertex with the highest degree in the remaining constraints.
    # Or just pick the first uncolored vertex.
    
    # Let's implement a simple DPLL-like solver.
    
    # Pre-compute for each vertex, which edges contain it.
    vertex_edges = [[] for _ in range(n)]
    for i, mask in enumerate(edge_masks):
        for v in range(n):
            if mask & (1 << v):
                vertex_edges[v].append(i)

    # We will use a stack for backtracking.
    # However, for simplicity and robustness within the time limit, we'll use a recursive function with memoization on the state of colored vertices?
    # State space is 2^n, which is too large.
    # But unit propagation helps.
    
    # Let's try a simple recursive backtracking with unit propagation.
    
    def solve(color, active_edges):
        # active_edges: list of edge indices that are not yet satisfied (i.e., not monochromatic)
        # Actually, an edge is "active" if it has at least one uncolored vertex.
        # An edge is "satisfied" if it has both colors.
        # An edge is "failed" if it has all 5 vertices of the same color.
        
        # Unit propagation
        changed = True
        while changed:
            changed = False
            # Check for failures and unit constraints
            # We need to track the count of 0s and 1s in each active edge.
            # Let's maintain counts dynamically or recompute?
            # Recomputing is O(E * 5) which is fast enough for small E.
            
            # Let's just check all active edges.
            new_active_edges = []
            for idx in active_edges:
                mask = edge_masks[idx]
                # Count colors in this edge
                count_0 = 0
                count_1 = 0
                uncolored = 0
                for v in range(n):
                    if mask & (1 << v):
                        if color[v] == 0:
                            count_0 += 1
                        elif color[v] == 1:
                            count_1 += 1
                        else:
                            uncolored += 1
                
                if count_0 == 5 or count_1 == 5:
                    return False, None # Failure
                
                if uncolored == 0:
                    # Edge is satisfied (has both colors)
                    continue
                
                if count_0 == 4:
                    # The uncolored vertex must be 1
                    for v in range(n):
                        if mask & (1 << v) and color[v] == -1:
                            color[v] = 1
                            changed = True
                            break
                elif count_1 == 4:
                    # The uncolored vertex must be 0
                    for v in range(n):
                        if mask & (1 << v) and color[v] == -1:
                            color[v] = 0
                            changed = True
                            break
                
                new_active_edges.append(idx)
            
            active_edges = new_active_edges
            
            if not active_edges:
                return True, color # All edges satisfied or no uncolored vertices left in active edges?
                # Wait, if active_edges is empty, it means all edges are satisfied.
        
        # If no unit propagation happened, pick a variable to branch on.
        # Heuristic: pick an uncolored vertex involved in the most active edges.
        uncolored_vertices = [v for v in range(n) if color[v] == -1]
        if not uncolored_vertices:
            return True, color
        
        # Pick the vertex with the highest "impact" (degree in active edges)
        best_v = -1
        max_deg = -1
        for v in uncolored_vertices:
            deg = 0
            for idx in active_edges:
                if edge_masks[idx] & (1 << v):
                    deg += 1
            if deg > max_deg:
                max_deg = deg
                best_v = v
        
        # Branch on best_v
        # Try 0
        color[best_v] = 0
        # We don't need to copy the list if we backtrack carefully, but Python lists are mutable.
        # Let's pass a copy or restore the value.
        success, _ = solve(color, active_edges)
        if success:
            return True, color
        
        # Try 1
        color[best_v] = 1
        success, _ = solve(color, active_edges)
        if success:
            return True, color
        
        # Backtrack
        color[best_v] = -1
        return False, None

    color = [-1] * n
    active_edges = list(range(len(edge_masks)))
    # Initial unit propagation
    # The solve function handles it.
    return solve(color, active_edges)[0]

def construct():
    """
    Returns a 5-uniform hypergraph with no proper 2-coloring.
    We start with a known non-2-colorable hypergraph and try to minimize edges.
    """
    # The complete hypergraph on 9 vertices has C(9,5) = 126 edges.
    # We know m(5) <= 51.
    # Let's try to find a smaller one or use a known 51-edge construction.
    # Since we don't have the specific 51-edge construction hardcoded, we will:
    # 1. Start with the complete hypergraph on a small n (e.g., 9 or 10).
    # 2. Randomly remove edges and check if it's still non-2-colorable.
    # 3. If it is, keep removing.
    # 4. If time runs out, return the best found.
    
    # Let's try n=9 first.
    # C(9,5) = 126.
    # We want to get down to ~50.
    
    n = 9
    all_edges = list(combinations(range(n), 5))
    current_edges = all_edges[:]
    
    # Shuffle edges to make removal order random
    random.shuffle(current_edges)
    
    # We will try to remove edges one by one.
    # To be efficient, we can try to remove multiple edges at once if possible,
    # but single removal is safer.
    
    # Optimization: If removing an edge makes it 2-colorable, we must put it back.
    # This is like finding a minimal unsatisfiable core.
    
    # Let's start with all edges.
    # We know it's not 2-colorable.
    
    # We'll iterate through the edges and try to remove each one.
    # We'll keep a list of edges to remove.
    
    edges_to_remove = []
    
    # To speed up, we can group edges.
    # Let's just iterate.
    
    # We need a fast solver. The recursive one above might be slow for n=9, E=126.
    # Let's optimize the solver or use a different approach.
    
    # Alternative: Use a SAT solver if available? No, standard library only.
    # The recursive solver with unit propagation should be okay for n=9.
    
    # Let's refine the solver to be iterative or use a better heuristic.
    # For now, let's try the removal process.
    
    # We'll maintain the current list of edges.
    # current_edges is a list of tuples.
    
    # We'll try to remove each edge in current_edges.
    # If removing it keeps the graph non-2-colorable, we remove it.
    
    # To avoid O(N^2) checks, we can just iterate and check.
    
    # Let's create a mutable list of edges.
    mutable_edges = current_edges[:]
    
    # We'll try to remove edges in a random order.
    # We'll use a while loop.
    
    # To speed up, we can check if the graph is 2-colorable with a timeout.
    # But the solver should be fast enough.
    
    # Let's try a different strategy: Start with a random subset of edges?
    # No, start with all and remove.
    
    # Let's implement a faster check.
    # For n=9, 2^9 = 512 colorings. We can just brute force all colorings!
    # 512 * 126 edges * 5 vertices = 322,560 operations. This is very fast.
    
    def is_2_colorable_bruteforce(n, edges):
        # Try all 2^n colorings
        for mask in range(1 << n):
            # Check if this coloring is proper
            valid = True
            for e in edges:
                # Check if all vertices in e have the same color
                # Get the colors of vertices in e
                colors = []
                for v in e:
                    if (mask >> v) & 1:
                        colors.append(1)
                    else:
                        colors.append(0)
                if len(set(colors)) == 1:
                    valid = False
                    break
            if valid:
                return True
        return False

    # Brute force is much simpler and faster for n <= 12.
    # n=9 is perfect.
    
    # Let's use brute force for the check.
    
    # Start with all edges.
    current_edges = all_edges[:]
    random.shuffle(current_edges)
    
    # Try to remove edges.
    # We'll iterate through the edges and try to remove them.
    # If removing an edge makes it 2-colorable, we skip it (keep it).
    # Otherwise, we remove it.
    
    # We'll do multiple passes until no more edges can be removed.
    
    changed = True
    while changed and time_remaining() > 0:
        changed = False
        # Shuffle the current edges to randomize the order
        random.shuffle(current_edges)
        
        for i in range(len(current_edges)):
            if time_remaining() < 0:
                break
            
            # Try to remove current_edges[i]
            # Create a new list without this edge
            # To be efficient, we can just check if the graph without this edge is 2-colorable.
            
            # We can optimize by checking if the edge is "critical".
            # If the graph without this edge is 2-colorable, the edge is critical.
            
            # Let's just do the check.
            test_edges = current_edges[:i] + current_edges[i+1:]
            
            if is_2_colorable_bruteforce(n, test_edges):
                # It is 2-colorable, so we cannot remove this edge.
                pass
            else:
                # It is not 2-colorable, so we can remove this edge.
                # Remove it from current_edges
                current_edges.pop(i)
                changed = True
                # We don't need to check the next edge immediately, just continue the loop
                # But we modified the list, so i is now pointing to the next element.
    
    # Return the result
    return {'n': n, 'edges': [list(e) for e in current_edges]}

# We should also consider trying larger n if n=9 doesn't yield a small enough edge count?
# The problem states m(5) <= 51.
# With n=9, we have 126 edges. Can we get down to 51?
# It's possible.
# If we run out of time, we return what we have.

# Let's add a fallback: if the result is too large, maybe try a different n?
# But n=9 is the smallest n for which non-2-colorable 5-uniform hypergraphs exist?
# Actually, for n < 9, all 5-uniform hypergraphs are 2-colorable?
# Let's check: n=8, C(8,5)=56. Is there a non-2-colorable one?
# If n=8, max edges is 56.
# If we can find a non-2-colorable hypergraph on n=8 with fewer edges, that would be better.
# But the problem says "best known construction has 51 edges".
# So 51 is a good target.

# Let's stick with n=9 for now, as it's a safe bet for non-2-colorability.

if __name__ == "__main__":
    result = construct()
    print(f"Found hypergraph with n={result['n']}, edges={len(result['edges'])}")
