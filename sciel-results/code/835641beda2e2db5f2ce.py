"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safe margin before 90s limit

    # Fallback: 7-edge Fano Plane (Score 1.4286)
    fano_edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    fano_colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    def is_2_colorable(edges):
        """
        Checks if the hypergraph defined by 'edges' is 2-colorable.
        Returns True if 2-colorable, False if not (i.e., 3-chromatic or higher).
        Uses backtracking with unit propagation.
        """
        n = 9
        # Precompute edge vertices for speed
        edge_verts = [[e[0], e[1], e[2]] for e in edges]
        
        def check_conflict(colors, v):
            # Check if assigning color `colors[v]` to `v` creates a monochromatic edge
            # where all other vertices are already assigned the same color.
            cv = colors[v]
            for e in edge_verts:
                if v in e:
                    others = [u for u in e if u != v]
                    # If all others are assigned and have the same color as v, it's a conflict
                    if all(colors[u] == cv for u in others):
                        return True
            return False

        def backtrack(idx, colors):
            if idx == n:
                # If we reach here, no conflict was found during assignment.
                # However, we must ensure no edge is monochromatic.
                # Actually, the conflict check during assignment handles this.
                # If we didn't prune, it's valid.
                return True
            
            v = idx
            
            # Try color 0
            colors[v] = 0
            if not check_conflict(colors, v):
                # Unit propagation check:
                # If an edge has 2 vertices with color 0, the 3rd MUST be 1.
                # If an edge has 2 vertices with color 1, the 3rd MUST be 0.
                # This is handled implicitly by the next step's check_conflict if we just proceed,
                # but explicit propagation is faster. For simplicity and correctness in this small case,
                # the standard backtracking with conflict check is sufficient.
                if backtrack(idx + 1, colors):
                    return True
            
            # Try color 1
            colors[v] = 1
            if not check_conflict(colors, v):
                if backtrack(idx + 1, colors):
                    return True
            
            colors[v] = -1
            return False

        colors = [-1] * n
        return backtrack(0, colors)

    def find_3_coloring(edges):
        """
        Finds a valid 3-coloring for the hypergraph defined by 'edges'.
        Returns list of colors or None.
        """
        n = 9
        edge_verts = [[e[0], e[1], e[2]] for e in edges]
        
        def check_mono(colors, v, c):
            cv = c
            for e in edge_verts:
                if v in e:
                    others = [u for u in e if u != v]
                    if all(colors[u] == cv for u in others if colors[u] != -1):
                        # If all assigned others are cv, and v is cv, it's mono.
                        # If some are unassigned, we can't say it's invalid yet.
                        if all(colors[u] != -1 for u in others):
                            return True
            return False

        def bt(idx, colors):
            if idx == n:
                # Final verification
                for e in edge_verts:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        return False
                return True
            
            v = idx
            for c in range(3):
                colors[v] = c
                if not check_mono(colors, v, c):
                    if bt(idx + 1, colors):
                        return True
            colors[v] = -1
            return False

        colors = [-1] * n
        if bt(0, colors):
            return colors
        return None

    # Target: Find a 6-edge 3-chromatic hypergraph
    # Strategy: Systematic search over edge sets.
    # We iterate through potential edge sets of size 6.
    # To make it tractable, we don't iterate all C(84, 6).
    # Instead, we use a heuristic construction:
    # Start with a known 2-colorable base, add edges that force 3rd color.
    
    # However, given the low probability of finding a 6-edge solution (if it exists) 
    # via random search in p7, we will try a more directed search.
    # If we find one, score is 1.5. Otherwise, we return the 7-edge Fano plane.
    
    # Let's do a limited random search with a better distribution or specific structure.
    # Actually, a simple random search is what p7 did. 
    # To be distinct and potentially more effective, we can try a "greedy" approach:
    # 1. Pick 3 edges that form a cycle or complex.
    # 2. Add edges that constrain colors.
    
    # For the sake of robustness and distinct code structure, we will attempt 
    # to find a 6-edge solution using a random search but with a time cap.
    # If found, great. If not, fallback.
    
    import random
    random.seed(123) # Different seed from p7 to explore different space
    
    all_triples = list(itertools.combinations(range(9), 3))
    
    found_6 = False
    best_edges_6 = None
    best_colors_6 = None
    
    # Limit iterations to fit within time budget while allowing for fast exit if found
    max_iters = 5000
    for i in range(max_iters):
        if time.time() > deadline - 5.0:
            break
            
        # Random sample of 6 distinct edges
        edges = random.sample(all_triples, 6)
        edges_list = [list(e) for e in edges]
        
        if not is_2_colorable(edges_list):
            colors = find_3_coloring(edges_list)
            if colors is not None:
                # Verify
                valid = True
                for e in edges_list:
                    if colors[e[0]] == colors[e[1]] == colors[e[2]]:
                        valid = False
                        break
                if valid:
                    best_edges_6 = edges_list
                    best_colors_6 = colors
                    found_6 = True
                    break
    
    if found_6:
        # Ensure types are Python int and list
        final_edges = [list(map(int, e)) for e in best_edges_6]
        final_colors = list(map(int, best_colors_6))
        return {'edges': final_edges, 'colors': final_colors}
    else:
        # Fallback to 7-edge Fano Plane
        final_edges = [list(map(int, e)) for e in fano_edges]
        final_colors = list(map(int, fano_colors))
        return {'edges': final_edges, 'colors': final_colors}

# EVOLVE-BLOCK-END
