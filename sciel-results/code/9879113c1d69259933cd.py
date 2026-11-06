import itertools
import time

def construct():
    # Chvátal graph on 12 vertices (0..11)
    # Edges of the Chvátal graph
    chvatal_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (10, 0), (10, 1), (10, 2), (10, 3), (10, 4),
        (11, 5), (11, 6), (11, 7), (11, 8), (11, 9),
        (10, 11), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Verify Chvátal graph properties (optional, for safety)
    # It should be 24 edges.
    # Let's check unique edges
    chvatal_edges = list(set(map(tuple, chvatal_edges)))
    if len(chvatal_edges) != 24:
        # Fallback to a known valid graph if definition is wrong
        # Grötzsch graph + 5 isolated
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]
        
    # Build adjacency matrix for Chvátal graph (0..11)
    n_old = 12
    adj = [[False]*n_old for _ in range(n_old)]
    for u, v in chvatal_edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Find all independent sets in the Chvátal graph
    # An independent set is a set of vertices with no edges between them
    independent_sets = []
    # Use backtracking to find all independent sets
    # Since n=12, we can iterate over all 2^12 = 4096 subsets
    for mask in range(1 << n_old):
        is_independent = True
        vertices = []
        for i in range(n_old):
            if (mask >> i) & 1:
                vertices.append(i)
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if adj[vertices[i]][vertices[j]]:
                    is_independent = False
                    break
            if not is_independent:
                break
        if is_independent:
            independent_sets.append(frozenset(vertices))
            
    # We want to assign an independent set to each of the 4 new vertices (12, 13, 14, 15)
    # To maximize edges, we should focus on larger independent sets.
    # Sort independent sets by size descending
    independent_sets.sort(key=len, reverse=True)
    
    # To limit search space, we can restrict to sets of size >= 3
    # But let's try all first, it's small.
    # However, 4096^4 is too big.
    # We need to prune.
    # Observation: If we pick a set I for vertex 12, and J for vertex 13,
    # the number of edges between 12 and 13 is 1 if I and J are disjoint, else 0.
    # The total edges added is sum(|I_k|) + number of disjoint pairs.
    
    # We can use a branch and bound or just iterate over the top K independent sets.
    # Let's take the top 50 largest independent sets.
    top_sets = independent_sets[:50]
    
    best_edges = []
    best_count = len(chvatal_edges)
    
    start_time = time.time()
    deadline = start_time + 100 # 100 seconds budget
    
    # Iterate over combinations of 4 sets from top_sets
    # This is 50^4 = 6,250,000 iterations. Feasible.
    for i1 in range(len(top_sets)):
        if time.time() > deadline: break
        s1 = top_sets[i1]
        for i2 in range(len(top_sets)):
            if time.time() > deadline: break
            s2 = top_sets[i2]
            for i3 in range(len(top_sets)):
                if time.time() > deadline: break
                s3 = top_sets[i3]
                for i4 in range(len(top_sets)):
                    if time.time() > deadline: break
                    s4 = top_sets[i4]
                    
                    sets = [s1, s2, s3, s4]
                    sum_sizes = sum(len(s) for s in sets)
                    
                    # Calculate max edges between new vertices
                    # Edge (k, l) exists if sets[k] and sets[l] are disjoint
                    inter_edges = 0
                    for k in range(4):
                        for l in range(k+1, 4):
                            if not (sets[k] & sets[l]):
                                inter_edges += 1
                                
                    total_new_edges = sum_sizes + inter_edges
                    total_edges = len(chvatal_edges) + total_new_edges
                    
                    if total_edges > best_count:
                        best_count = total_edges
                        # Reconstruct edges
                        best_edges = list(chvatal_edges)
                        new_vertices = [12, 13, 14, 15]
                        for k in range(4):
                            v = new_vertices[k]
                            for u in sets[k]:
                                best_edges.append([v, u])
                        for k in range(4):
                            for l in range(k+1, 4):
                                if not (sets[k] & sets[l]):
                                    best_edges.append([new_vertices[k], new_vertices[l]])
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result

# EVOLVE-BLOCK-START
def construct():
    # Chvátal graph on 12 vertices (0..11)
    # Edges of the Chvátal graph
    chvatal_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (10, 0), (10, 1), (10, 2), (10, 3), (10, 4),
        (11, 5), (11, 6), (11, 7), (11, 8), (11, 9),
        (10, 11), (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Verify Chvátal graph properties (optional, for safety)
    # It should be 24 edges.
    # Let's check unique edges
    chvatal_edges = list(set(map(tuple, chvatal_edges)))
    if len(chvatal_edges) != 24:
        # Fallback to a known valid graph if definition is wrong
        # Grötzsch graph + 5 isolated
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        edges = [list(e) for e in cycle]
        for u, w in cycle:
            edges += [[u, 5 + w], [w, 5 + u]]
        return edges + [[5 + u, 10] for u in range(5)]
        
    # Build adjacency matrix for Chvátal graph (0..11)
    n_old = 12
    adj = [[False]*n_old for _ in range(n_old)]
    for u, v in chvatal_edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Find all independent sets in the Chvátal graph
    independent_sets = []
    for mask in range(1 << n_old):
        is_independent = True
        vertices = []
        for i in range(n_old):
            if (mask >> i) & 1:
                vertices.append(i)
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if adj[vertices[i]][vertices[j]]:
                    is_independent = False
                    break
            if not is_independent:
                break
        if is_independent:
            independent_sets.append(frozenset(vertices))
            
    # Sort independent sets by size descending
    independent_sets.sort(key=len, reverse=True)
    
    # To limit search space, we take the top 50 largest independent sets
    top_sets = independent_sets[:50]
    
    best_edges = []
    best_count = len(chvatal_edges)
    
    start_time = time.time()
    deadline = start_time + 100 # 100 seconds budget
    
    # Iterate over combinations of 4 sets from top_sets
    for i1 in range(len(top_sets)):
        if time.time() > deadline: break
        s1 = top_sets[i1]
        for i2 in range(len(top_sets)):
            if time.time() > deadline: break
            s2 = top_sets[i2]
            for i3 in range(len(top_sets)):
                if time.time() > deadline: break
                s3 = top_sets[i3]
                for i4 in range(len(top_sets)):
                    if time.time() > deadline: break
                    s4 = top_sets[i4]
                    
                    sets = [s1, s2, s3, s4]
                    sum_sizes = sum(len(s) for s in sets)
                    
                    # Calculate max edges between new vertices
                    inter_edges = 0
                    for k in range(4):
                        for l in range(k+1, 4):
                            if not (sets[k] & sets[l]):
                                inter_edges += 1
                                
                    total_new_edges = sum_sizes + inter_edges
                    total_edges = len(chvatal_edges) + total_new_edges
                    
                    if total_edges > best_count:
                        best_count = total_edges
                        # Reconstruct edges
                        best_edges = list(chvatal_edges)
                        new_vertices = [12, 13, 14, 15]
                        for k in range(4):
                            v = new_vertices[k]
                            for u in sets[k]:
                                best_edges.append([v, u])
                        for k in range(4):
                            for l in range(k+1, 4):
                                if not (sets[k] & sets[l]):
                                    best_edges.append([new_vertices[k], new_vertices[l]])
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    return result
# EVOLVE-BLOCK-END
