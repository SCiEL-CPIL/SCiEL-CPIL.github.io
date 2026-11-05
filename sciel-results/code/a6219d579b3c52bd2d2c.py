import time
import random
from itertools import combinations

def construct():
    # 1. Construct Grötzsch graph on vertices 0..10
    # Standard Grötzsch graph edges
    # Cycle 0-1-2-3-4-0
    # Cycle 5-6-7-8-9-5
    # 10 connected to 5,6,7,8,9
    # Edges between 0-4 and 5-9:
    # (0,5), (1,6), (2,7), (3,8), (4,9)
    # Wait, the current code uses a specific construction. Let's stick to it to ensure it's Grötzsch.
    # The current code:
    # cycle = [(i, (i + 1) % 5) for i in range(5)] -> 0-1, 1-2, 2-3, 3-4, 4-0
    # for u, w in cycle:
    #   add (u, 5+w) and (w, 5+u)
    #   This creates a "twisted" connection.
    #   u=0, w=1: (0,6), (1,5)
    #   u=1, w=2: (1,7), (2,6)
    #   u=2, w=3: (2,8), (3,7)
    #   u=3, w=4: (3,9), (4,8)
    #   u=4, w=0: (4,5), (0,9)
    #   And 10 connected to 5..9.
    # This is a valid Grötzsch graph representation.
    
    groetzs_edges = set()
    n_g = 11
    # Cycle 0-4
    for i in range(5):
        groetzs_edges.add((min(i, (i+1)%5), max(i, (i+1)%5)))
    # Cycle 5-9
    for i in range(5):
        u = i
        w = (i+1)%5
        groetzs_edges.add((min(5+u, 5+w), max(5+u, 5+w)))
    # Connections
    for i in range(5):
        u = i
        w = (i+1)%5
        groetzs_edges.add((min(u, 5+w), max(u, 5+w)))
        groetzs_edges.add((min(w, 5+u), max(w, 5+u)))
    # Vertex 10 to 5-9
    for i in range(5):
        groetzs_edges.add((min(10, 5+i), max(10, 5+i)))
        
    # 2. Find all independent sets in Grötzsch graph
    # Vertices 0..10
    # An independent set is a set of vertices with no edges between them.
    # We represent sets as frozensets or bitmasks. Bitmasks are faster.
    # Mask for 11 bits.
    
    # Build adjacency bitmask for Grötzsch
    adj_mask = [0] * 11
    for u, v in groetzs_edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    independent_sets = []
    # Iterate over all 2^11 subsets
    for mask in range(1 << 11):
        # Check if mask is independent
        is_indep = True
        for i in range(11):
            if mask & (1 << i):
                # Check if any neighbor of i is in mask
                if adj_mask[i] & mask:
                    is_indep = False
                    break
        if is_indep:
            independent_sets.append(mask)
            
    # 3. Iterate over all edge subsets for vertices 11..15
    # Vertices 11..15 are 5 vertices.
    # There are 10 possible edges between them.
    # We represent the edge set as a bitmask of 10 bits.
    # Map edge (i,j) to bit index.
    # i,j in 0..4 corresponding to 11..15.
    new_vertices = [11, 12, 13, 14, 15]
    edge_map = {}
    idx = 0
    for i in range(5):
        for j in range(i+1, 5):
            edge_map[(i, j)] = idx
            idx += 1
            
    best_total_edges = len(groetzs_edges)
    best_config = None
    
    # Precompute triangles in the 5-vertex subgraph
    # A triangle is 3 vertices with 3 edges.
    # We can just check the bitmask.
    
    for edge_mask in range(1 << 10):
        # Check if edge_mask forms a triangle-free graph on 5 vertices
        # Triangles in K5:
        # (0,1,2), (0,1,3), (0,1,4), (0,2,3), (0,2,4), (0,3,4), (1,2,3), (1,2,4), (1,3,4), (2,3,4)
        # We can precompute the bitmask for each triangle.
        # If (edge_mask & tri_mask) == tri_mask, then triangle exists.
        
        # Let's just check all 10 triangles
        has_triangle = False
        # Triangles are defined by 3 vertices
        # We can hardcode the triangle masks
        # Or iterate
        # For simplicity, check all 3-combinations of 5 vertices
        for c in combinations(range(5), 3):
            e1 = edge_map[(c[0], c[1])]
            e2 = edge_map[(c[1], c[2])]
            e3 = edge_map[(c[0], c[2])]
            if (edge_mask & (1 << e1)) and (edge_mask & (1 << e2)) and (edge_mask & (1 << e3)):
                has_triangle = True
                break
        if has_triangle:
            continue
            
        # Solve for independent sets S_11..S_15
        # Variables: 5
        # Domains: independent_sets (list of masks)
        # Constraints: If edge (i,j) in edge_mask, then S_i & S_j == 0
        
        # Backtracking
        best_sum = -1
        best_assignment = [0] * 5
        
        def backtrack(k, current_sum, assignment):
            nonlocal best_sum, best_assignment
            if k == 5:
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_assignment = assignment[:]
                return
            
            for s in independent_sets:
                # Check constraints with previous vertices
                valid = True
                for j in range(k):
                    if edge_mask & (1 << edge_map[(j, k)]):
                        if s & assignment[j]:
                            valid = False
                            break
                if valid:
                    assignment[k] = s
                    backtrack(k+1, current_sum + bin(s).count("1"), assignment)
                    assignment[k] = 0
                    
        backtrack(0, 0, [0]*5)
        
        if best_sum > -1:
            total_edges = len(groetzs_edges) + best_sum + bin(edge_mask).count("1")
            if total_edges > best_total_edges:
                best_total_edges = total_edges
                best_config = (edge_mask, best_assignment)
                
    # 4. Construct final edge list
    if best_config:
        edge_mask, assignment = best_config
        final_edges = list(groetzs_edges)
        
        # Add edges between 11-15
        for i in range(5):
            for j in range(i+1, 5):
                if edge_mask & (1 << edge_map[(i, j)]):
                    final_edges.append([11+i, 11+j])
                    
        # Add edges between 0-10 and 11-15
        for i in range(5):
            s = assignment[i]
            for j in range(11):
                if s & (1 << j):
                    final_edges.append([j, 11+i])
                    
        return final_edges
    else:
        # Fallback
        return [[u, v] for u, v in groetzs_edges]

