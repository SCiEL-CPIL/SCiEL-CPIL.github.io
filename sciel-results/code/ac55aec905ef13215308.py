import random

def has_q3(edges):
    # Build adjacency matrix
    n = 16
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Precompute all 8-subsets
    # This is too slow if done from scratch every time.
    # But we only add edges.
    # We can just check the new edge's impact?
    # Simpler: Just check the whole graph each time. 12870 subsets * 8^2 checks is small.
    
    from itertools import combinations
    
    # Q3 structure:
    # Vertices 0..7
    # Edges: (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    
    for subset in combinations(range(n), 8):
        # Map subset indices to 0..7
        # Check if all q3_edges are present in the induced subgraph
        # We need to check all permutations of mapping?
        # No, we can check if the induced subgraph contains Q3 as a subgraph.
        # This is isomorphism check.
        # Simpler: Just check if there exists a mapping.
        # Given small size, we can try to match.
        
        # Optimization: Only check if the induced subgraph has at least 12 edges.
        ind_edges = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[subset[i]][subset[j]]:
                    ind_edges += 1
        if ind_edges < 12:
            continue
            
        # Check isomorphism to Q3
        # Try all 8! mappings? 40320.
        # 12870 * 40320 is too slow.
        
        # Better: Backtracking search for Q3 in the whole graph.
        pass
        
    return False

def construct():
    # Seed: K3,13
    edges = []
    A = [0, 1, 2]
    B = list(range(3, 16))
    for a in A:
        for b in B:
            edges.append([a, b])
            
    # Potential edges to add
    potential = []
    # Within B
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            potential.append([B[i], B[j]])
    # Within A
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            potential.append([A[i], A[j]])
            
    random.shuffle(potential)
    
    for e in potential:
        new_edges = edges + [e]
        if not has_q3(new_edges):
            edges = new_edges
            
    return edges
