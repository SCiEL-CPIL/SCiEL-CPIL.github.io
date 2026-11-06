<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # K9 split by circular distance: distances 1, 2 and 4 give nonagons, distance 3 gives three triangles.
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precomputed known good colouring for n=17 (best known R_4(C6) > 17)
# We'll use a construction based on Z_17 or search for it.
# For now, provide a valid n=9 as fallback, and try to search for larger.

def _has_c6_in_color(color_graph, n):
    """Check if the subgraph of a single color contains a 6-cycle (C6).
    color_graph: set of edges (frozenset of 2-tuples) for one color class.
    n: number of vertices.
    """
    # Build adjacency for this color
    adj = [set() for _ in range(n)]
    for (u, v) in color_graph:
        adj[u].add(v)
        adj[v].add(u)
    
    # Look for 6-cycles: v0-v1-v2-v3-v4-v5-v0
    # Enumerate v0 < v1 < v2 to avoid duplicates, then find paths of length 4 from v2 back to v0
    # More efficient: for each pair of vertices at distance 2, check if they share a common neighbor at distance 4
    # Simplest: enumerate all 6-tuples with v0 < v1 < v2 < v3 < v4 < v5 and check cycle edges
    # But that's O(n^6). Better: for each v0, v1, v2, v3, check if v3 connects to v4 which connects to v5 which connects to v0
    
    # For each starting vertex v0
    for v0 in range(n):
        # v1 is a neighbor of v0
        for v1 in adj[v0]:
            if v1 <= v0:
                continue  # to reduce symmetry
            # v2 is a neighbor of v1, v2 != v0
            for v2 in adj[v1]:
                if v2 == v0:
                    continue
                # v3 is a neighbor of v2, v3 not in {v0, v1}
                for v3 in adj[v2]:
                    if v3 in (v0, v1):
                        continue
                    # v4 is a neighbor of v3, v4 not in {v0, v1, v2}
                    for v4 in adj[v3]:
                        if v4 in (v0, v1, v2):
                            continue
                        # v5 must be a neighbor of v4 AND a neighbor of v0, v5 not in {v0,v1,v2,v3,v4}
                        common = adj[v4] & adj[v0]
                        for v5 in common:
                            if v5 not in (v1, v2, v3, v4):
                                return True
    return False

def _check_no_c6(colors, n):
    """Check that no color class contains a C6."""
    color_edges = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            color_edges[c].add((i, j))
            idx += 1
    for c in range(4):
        if _has_c6_in_color(color_edges[c], n):
            return False
    return True

def _backtrack_search(n, deadline):
    """Try to find a C6-free 4-coloring of K_n via backtracking."""
    total_edges = n * (n - 1) // 2
    colors = [None] * total_edges
    
    def edge_idx(i, j):
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    def check_partial_c6():
        """Check if any color class in the partial coloring already contains a C6."""
        color_edges = [set() for _ in range(4)]
        for idx in range(total_edges):
            if colors[idx] is not None:
                # decode idx back to (i,j)
                # find i,j such that edge_idx(i,j) == idx
                # i is largest such that i*(n-1) - i*(i+1)//2 <= idx
                # Simpler: precompute
                pass
        # This is getting complicated. Let me use a different approach.
        return True  # skip for now
    
    # Simpler approach: assign edges in order, and after each assignment check for C6
    # But checking C6 each time is expensive.
    
    # Alternative: use a known construction or randomized search
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Fallback: known valid n=9
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find larger colourings using randomized local search
    # Start with n=10, 11, ..., up to 17 or 20
    
    for target_n in range(10, 21):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initial coloring
            colors = [random.randint(0, 3) for _ in range(target_n * (target_n - 1) // 2)]
            
            # Local search: try to fix C6 violations
            improved = True
            iterations = 0
            while improved and iterations < 1000:
                iterations += 1
                if time.time() > deadline - 5:
                    break
                
                # Check for C6
                if _check_no_c6(colors, target_n):
                    found = True
                    break
                
                # Find a C6 and recolor one of its edges
                color_edges = [set() for _ in range(4)]
                for idx in range(target_n * (target_n - 1) // 2):
                    c = colors[idx]
                    # decode idx to (i,j)
                    i = 0
                    while i * (target_n - 1) - i * (i + 1) // 2 < idx:
                        i += 1
                    # Actually let me precompute the edge list
                    break
                
                # This is getting too complex. Let me use a simpler randomized approach.
                break
            
            if found:
                best_n = target_n
                best_colors = colors
                break
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, my implementation is getting too complex and buggy. Let me rewrite it more carefully with a cleaner approach.

