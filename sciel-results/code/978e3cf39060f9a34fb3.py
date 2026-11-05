"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from itertools import combinations

def _has_c6_in_color(colors, n, color):
    """Check if edges of given color form a graph containing a C6."""
    # Build adjacency for this color
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == color:
                adj[i][j] = adj[j][i] = True
            idx += 1
    
    # For each pair of vertices at distance 2, check if they have a common neighbor
    # A C6 exists if there's a path of length 5 between two vertices that are also at distance 1
    # More efficiently: check all 6-tuples? That's O(n^6) which is too slow for n=20.
    
    # Better: For each ordered pair (u,v) with u!=v, find common neighbors.
    # A C6: v0-v1-v2-v3-v4-v5-v0. 
    # Equivalent: there exist distinct v0..v5 such that consecutive pairs are edges.
    
    # Use a smarter check: for each edge (a,b) in the color graph, 
    # check if there's a path of length 4 from a to b using only color edges,
    # with all intermediate vertices distinct from a,b.
    
    # Actually, let's use: a C6 exists iff there exist two vertices at distance exactly 6 in the graph?
    # No, that's not right either.
    
    # Let's use: for each pair (u,v) that are NOT adjacent in this color,
    # check if there's a path of length 5 from u to v. But that's complex.
    
    # Simpler: enumerate all 6-cycles. For n<=20, number of 6-tuples is manageable with pruning.
    # Actually, let's use: a graph has a C6 iff there exist distinct vertices v0,v1,v2,v3,v4,v5
    # such that (vi, v(i+1 mod 6)) are all edges.
    
    # Efficient check: for each pair of edges (a,b) and (c,d) that are disjoint,
    # check if there's a path of length 2 from b to c and from d to a, etc.
    
    # Let me just do a BFS-based check for girth or use networkx-like approach manually.
    
    # For n<=20, we can check all 6-cycles by:
    # For each starting vertex v0, for each neighbor v1, for each neighbor v2 of v1 (v2!=v0),
    # for each neighbor v3 of v2 (v3!=v0,v1), for each neighbor v4 of v3 (v4!=v0,v1,v2),
    # check if v4 is adjacent to v5 where v5 is a neighbor of v4 and v5 adjacent to v0, v5 not in {v0..v4}.
    
    # This is O(n * d^5) where d is avg degree. For n=20, this should be fine.
    
    for v0 in range(n):
        for v1 in range(n):
            if v1 == v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 in (v0, v1) or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 in (v0, v1, v2) or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 in (v0, v1, v2, v3) or not adj[v3][v4]:
                            continue
                        for v5 in range(n):
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            if adj[v4][v5] and adj[v5][v0]:
                                return True
    return False

def _count_c6s_in_color(colors, n, color):
    """Count number of C6s in the subgraph of given color."""
    adj = [[False] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == color:
                adj[i][j] = adj[j][i] = True
            idx += 1
    
    count = 0
    for v0 in range(n):
        for v1 in range(n):
            if v1 == v0 or not adj[v0][v1]:
                continue
            for v2 in range(n):
                if v2 in (v0, v1) or not adj[v1][v2]:
                    continue
                for v3 in range(n):
                    if v3 in (v0, v1, v2) or not adj[v2][v3]:
                        continue
                    for v4 in range(n):
                        if v4 in (v0, v1, v2, v3) or not adj[v3][v4]:
                            continue
                        for v5 in range(n):
                            if v5 in (v0, v1, v2, v3, v4):
                                continue
                            if adj[v4][v5] and adj[v5][v0]:
                                count += 1
    return count

def _has_any_c6(colors, n):
    """Check if any color class contains a C6."""
    for c in range(4):
        if _has_c6_in_color(colors, n, c):
            return True
    return False

def _total_c6_count(colors, n):
    """Total number of monochromatic C6s across all colors."""
    total = 0
    for c in range(4):
        total += _count_c6s_in_color(colors, n, c)
    return total

def construct():
    """Find a 4-coloring of K_n with no monochromatic C6, maximizing n."""
    random.seed(42)
    
    # Try from high n down to low n
    for n in range(17, 1, -1):
        # Try multiple random initializations
        for attempt in range(50):
            # Initialize with random colors
            m = n * (n - 1) // 2
            colors = [random.randint(0, 3) for _ in range(m)]
            
            # Local search: try to eliminate C6s
            improved = True
            max_iter = 200
            it = 0
            while improved and it < max_iter:
                improved = False
                it += 1
                # Find a C6 in some color
                target_color = -1
                for c in range(4):
                    if _has_c6_in_color(colors, n, c):
                        target_color = c
                        break
                if target_color == -1:
                    # No C6s found!
                    return {'n': n, 'colors': colors}
                
                # Find a specific C6 to break
                adj = [[False] * n for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if colors[idx] == target_color:
                            adj[i][j] = adj[j][i] = True
                        idx += 1
                
                # Find one C6
                cycle = None
                for v0 in range(n):
                    if cycle:
                        break
                    for v1 in range(n):
                        if v1 == v0 or not adj[v0][v1]:
                            continue
                        for v2 in range(n):
                            if v2 in (v0, v1) or not adj[v1][v2]:
                                continue
                            for v3 in range(n):
                                if v3 in (v0, v1, v2) or not adj[v2][v3]:
                                    continue
                                for v4 in range(n):
                                    if v4 in (v0, v1, v2, v3) or not adj[v3][v4]:
                                        continue
                                    for v5 in range(n):
                                        if v5 in (v0, v1, v2, v3, v4):
                                            continue
                                        if adj[v4][v5] and adj[v5][v0]:
                                            cycle = (v0, v1, v2, v3, v4, v5)
                                            break
                                    if cycle:
                                        break
                                if cycle:
                                    break
                            if cycle:
                                break
                        if cycle:
                            break
                    if cycle:
                        break
                
                if not cycle:
                    break
                
                # Pick a random edge in the C6 to recolor
                edges_in_cycle = [(cycle[0], cycle[1]), (cycle[1], cycle[2]), 
                                  (cycle[2], cycle[3]), (cycle[3], cycle[4]),
                                  (cycle[4], cycle[5]), (cycle[5], cycle[0])]
                ei = random.randint(0, 5)
                a, b = edges_in_cycle[ei]
                if a > b:
                    a, b = b, a
                
                # Find index in colors array
                idx = 0
                found_idx = -1
                for i in range(n):
                    for j in range(i + 1, n):
                        if i == a and j == b:
                            found_idx = idx
                            break
                        idx += 1
                    if found_idx != -1:
                        break
                
                if found_idx == -1:
                    continue
                
                # Try recoloring to a different color
                old_color = colors[found_idx]
                for new_color in range(4):
                    if new_color == old_color:
                        continue
                    colors[found_idx] = new_color
                    # Check if this reduces total C6s
                    if not _has_any_c6(colors, n):
                        improved = True
                        break
                    colors[found_idx] = old_color
                
                if not improved:
                    # Try random recoloring of this edge
                    new_color = random.randint(0, 3)
                    if new_color != old_color:
                        colors[found_idx] = new_color
                        improved = True
        
        # If we couldn't solve for this n, try lower
    # Fallback: return a small valid coloring
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
# EVOLVE-BLOCK-END
