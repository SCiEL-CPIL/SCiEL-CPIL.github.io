import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Base solution: 8 disjoint K4s
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Try to find a 9-vertex C5-free graph with alpha <= 2
    # If found, replace two K4s (8 vertices) with this graph (9 vertices)
    # New n = 6*4 + 9 = 33
    # New alpha = 6*1 + 2 = 8
    
    def is_valid_9_graph(adj):
        # adj is a list of sets of neighbors
        # Check C5
        # We can check all 5-cycles
        # n=9, vertices 0..8
        # A cycle is a sequence v0..v4 such that vi-vi+1 is edge, v4-v0 is edge, all distinct
        # To avoid duplicates, we can fix v0=0 and v1 < v2 < v3 < v4? No, that's for cliques.
        # Just check all permutations? 9*8*7*6*5 = 15120. Fast.
        # Or use DFS/BFS to find cycles of length 5.
        
        # Check Alpha <= 2
        # Check all subsets of size 3
        # If any subset of 3 vertices has no edges between them, alpha >= 3
        for i in range(9):
            for j in range(i+1, 9):
                for k in range(j+1, 9):
                    if j not in adj[i] and k not in adj[i] and k not in adj[j]:
                        return False
        
        # Check C5
        # Simple check: for each vertex, look for paths of length 4 that close?
        # Or just enumerate all 5-cycles.
        # Since n is small, we can use a bitset or just loops.
        # Let's use a simple loop for 5-cycles.
        # v0, v1, v2, v3, v4
        # To avoid counting the same cycle multiple times, we can enforce v0 < v1 < v2 < v3 < v4?
        # No, that's not sufficient for cycles.
        # Standard way: find all cycles of length 5.
        # We can use networkx if available, but we should stick to stdlib/np.
        # Let's just do a DFS from each vertex to find cycles of length 5.
        # Or simpler:
        # For each pair (u, v), check if there is a path of length 3 from u to v that is vertex-disjoint from u, v?
        # Actually, a C5 exists if there are vertices u, v such that there are two internally disjoint paths of length 2 between them? No.
        
        # Let's just check all 5-tuples (a,b,c,d,e) with a<b<c<d<e? No.
        # We can check if the graph contains a C5 by checking if the complement contains a C5? No.
        
        # Efficient check for C5 in 9 vertices:
        # There are 9*8*7*6*5 / 10 = 252 unique cycles?
        # 9 vertices. Number of 5-cycles is at most C(9,5) * 5!/10 = 126 * 12 = 1512.
        # We can just iterate through all combinations of 5 vertices and check if they form a C5.
        # A set of 5 vertices forms a C5 if the induced subgraph is a C5.
        # Induced subgraph on 5 vertices has 5 edges and is 2-regular.
        
        for combo in itertools.combinations(range(9), 5):
            # Check if induced subgraph is C5
            # Count edges in induced subgraph
            edge_count = 0
            for i in range(5):
                for j in range(i+1, 5):
                    if combo[j] in adj[combo[i]]:
                        edge_count += 1
            if edge_count == 5:
                # Check if it's a single cycle (2-regular)
                # In a 5-vertex graph with 5 edges, if it's connected, it's a C5.
                # If it's not connected, it could be C3 + 2 isolated? No, 5 edges.
                # C3 has 3 edges. Remaining 2 vertices with 2 edges? Impossible (max 1 edge between 2 vertices).
                # So any 5-vertex graph with 5 edges is either C5 or C3 + something?
                # C3 (3 edges) + 2 vertices. Max edges between 2 vertices is 1. Total 4.
                # So 5 edges implies C5.
                return False
                
        return True

    found_9_graph = False
    best_9_edges = []
    
    # Random search for 9-vertex graph
    # Start with a random graph and optimize?
    # Or just random sampling.
    # We want high edge density to keep alpha low.
    # Let's try random graphs with p=0.6
    
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Generate random graph
        adj = [set() for _ in range(9)]
        for i in range(9):
            for j in range(i+1, 9):
                if random.random() < 0.6:
                    adj[i].add(j)
                    adj[j].add(i)
        
        if is_valid_9_graph(adj):
            found_9_graph = True
            best_9_edges = []
            for i in range(9):
                for j in adj[i]:
                    if i < j:
                        best_9_edges.append([i, j])
            break

    if found_9_graph:
        # Construct 33-vertex graph
        n = 33
        edges = []
        # 6 K4s
        for b in range(6):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        
        # 1 9-vertex graph
        base = 24
        for u, v in best_9_edges:
            edges.append([base + u, base + v])
            
        return {'n': n, 'edges': edges}
    
    # Return 32-vertex graph if 33 not found
    return {'n': n, 'edges': edges}
