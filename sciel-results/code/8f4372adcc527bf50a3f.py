"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""

import itertools
import random
from collections import deque


def construct():
    """
    Return a 5-uniform hypergraph on n vertices with no proper 2-colouring,
    minimizing the number of edges.

    We use a systematic search:
    1. Start with small n and build candidate edge sets.
    2. Use backtracking with unit propagation to check 2-colourability.
    3. Minimize edge count.

    Known: m(5) <= 51. We aim to find a construction with as few edges as possible.
    """
    
    def check_not_2colorable(n, edges):
        """
        Check if the hypergraph is NOT 2-colorable.
        2-colorable means: exists a 2-coloring where every edge has both colors.
        NOT 2-colorable means: for every 2-coloring, some edge is monochromatic.
        
        We search for a coloring where every edge is mixed (has both colors).
        If no such coloring exists, return True (not 2-colorable).
        """
        m = len(edges)
        if m == 0:
            return False  # empty hypergraph is trivially 2-colorable
        
        # Convert edges to sets for faster lookup
        edge_sets = [set(e) for e in edges]
        
        # For each vertex, list of edges containing it
        vertex_edges = [[] for _ in range(n)]
        for ei, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(ei)
        
        # State: assignment[v] = -1 (unassigned), 0, or 1
        # We use backtracking with unit propagation
        
        def backtrack(assignment):
            """
            assignment: list of length n with -1, 0, or 1
            Returns True if we find a valid mixed coloring
            """
            # Find first unassigned vertex
            v = -1
            for i in range(n):
                if assignment[i] == -1:
                    v = i
                    break
            if v == -1:
                # All assigned - check all edges are mixed
                for ei in range(m):
                    colors = set(assignment[x] for x in edge_sets[ei])
                    if len(colors) < 2:
                        return False
                return True
            
            # Try both colors
            for c in [0, 1]:
                assignment[v] = c
                if propagate(assignment):
                    if backtrack(assignment):
                        return True
                assignment[v] = -1
            return False
        
        def propagate(assignment):
            """
            Unit propagation: if an edge has all but one vertex assigned the same color,
            the last must be the opposite color.
            Also detect immediate violations.
            Returns False if contradiction detected.
            """
            changed = True
            while changed:
                changed = False
                for ei in range(m):
                    ev = edge_sets[ei]
                    assigned_vals = []
                    unassigned = []
                    for x in ev:
                        if assignment[x] == -1:
                            unassigned.append(x)
                        else:
                            assigned_vals.append(assignment[x])
                    
                    if len(unassigned) == 0:
                        # All assigned - check if mixed
                        if len(set(assigned_vals)) < 2:
                            return False
                    elif len(unassigned) == 1:
                        # Only one unassigned - if all assigned are same color, force opposite
                        if len(set(assigned_vals)) == 1:
                            forced_color = 1 - assigned_vals[0]
                            if assignment[unassigned[0]] == -1:
                                assignment[unassigned[0]] = forced_color
                                changed = True
                            elif assignment[unassigned[0]] != forced_color:
                                return False
                    else:
                        # Multiple unassigned - no forced assignment yet
                        pass
            return True
        
        # Initial propagation with all unassigned
        assignment = [-1] * n
        if not propagate(assignment):
            return True  # Contradiction -> not 2-colorable
        
        return not backtrack(assignment)
    
    # Strategy: Search for small non-2-colorable 5-uniform hypergraphs
    # Start with n=9, try to find minimal edge sets
    
    # Known lower bound: at least 29 edges needed (since <=28 is always 2-colorable)
    # Best known: 51 edges
    
    # We'll use a heuristic search:
    # 1. Generate random edge sets
    # 2. Check if not 2-colorable
    # 3. Try to minimize
    
    # First, let's try known constructions
    
    # Construction 1: All 5-subsets of [9] -> 126 edges (works but too many)
    # Construction 2: Use a subset of edges
    
    # Let's try a systematic approach:
    # For n=9, we need at least 29 edges
    # Try random subsets of increasing size
    
    best = None
    best_edges_count = float('inf')
    
    # Try n from 9 to 12
    for n in range(9, 13):
        all_possible = list(itertools.combinations(range(n), 5))
        random.shuffle(all_possible)
        
        # Try different sizes
        for num_edges in range(29, min(60, len(all_possible) + 1)):
            # Random sampling
            found = False
            for _ in range(50):  # 50 random attempts per size
                random_edges = random.sample(all_possible, num_edges)
                if check_not_2colorable(n, random_edges):
                    if best is None or num_edges < best_edges_count:
                        best = (n, [list(e) for e in random_edges])
                        best_edges_count = num_edges
                    found = True
                    break
        
        # If we found one, stop trying larger n
        if best is not None and best_edges_count <= 51:
            break
    
    # If we didn't find anything, fall back to a known construction
    if best is None:
        # Use the complete 5-uniform on 9 vertices
        n = 9
        edges = [list(e) for e in itertools.combinations(range(n), 5)]
        best = (n, edges)
    
    return {'n': best[0], 'edges': best[1]}
