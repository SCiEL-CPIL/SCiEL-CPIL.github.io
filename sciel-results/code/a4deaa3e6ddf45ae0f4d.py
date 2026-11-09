import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Known construction: The complement of the Schläfli graph (or similar) 
    # Actually, let's try a specific known graph: The Clebsch graph (16 vertices) has no regular induced subgraph on 6 vertices? 
    # Wait, the Clebsch graph is strongly regular (16, 5, 0, 2). 
    # Let's check if it has a regular induced subgraph on 6 vertices.
    # A regular induced subgraph on 6 vertices in a (16,5,0,2) graph.
    # If we pick 6 vertices, the induced subgraph must be regular.
    # Possible regular graphs on 6 vertices: empty, matching, C6, K3,3, K6, etc.
    # In Clebsch graph, max clique is 4. Max independent set is 5.
    # So no K6, no empty set of size 6 (since alpha=5).
    # What about 1-regular (matching)? 2-regular (cycles)? 3-regular?
    # The Clebsch graph is a known example for G(5) or G(6)?
    # Actually, the problem asks for G(6) > n. 
    # Known values: G(5)=12, G(6)=? 
    # Literature says G(6) >= 20. 
    # The Clebsch graph has 16 vertices. 
    # Let's check if Clebsch graph works.
    # If Clebsch graph has no regular induced subgraph on 6 vertices, then n=16 is a valid solution, which is > 5.
    
    # Let's generate the Clebsch graph.
    # Vertices are even-weight binary strings of length 5.
    # Two vertices are adjacent if their Hamming distance is 2.
    
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
            
    n = len(vertices) # 16
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            u = vertices[i]
            v = vertices[j]
            dist = bin(u ^ v).count('1')
            if dist == 2:
                edges.append([i, j])
                
    # Verify if this graph has no regular induced subgraph on 6 vertices.
    # We must check all subsets of size 6.
    # C(16,6) = 8008. This is fast enough to check directly.
    
    def is_valid_graph(n, edges):
        # Build adjacency matrix for fast subset checks
        adj = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            adj[u, v] = True
            adj[v, u] = True
            
        # Check all subsets of size 6
        for subset in combinations(range(n), 6):
            # Get induced subgraph
            sub_adj = adj[np.ix_(subset, subset)]
            # Check if regular: all degrees in subgraph are equal
            degrees = sub_adj.sum(axis=1)
            if len(set(degrees)) == 1:
                return False
        return True

    if is_valid_graph(n, edges):
        return {'n': n, 'edges': edges}
    
    # If Clebsch fails (it might have a 2-regular induced subgraph, e.g., a 6-cycle),
    # we need a better candidate.
    # Let's try the Hoffman-Singleton graph? No, that's 50 vertices.
    # What about the complement of the Clebsch graph?
    # Complement of Clebsch is (16, 10, 6, 6).
    # Let's check the complement.
    
    comp_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if [i, j] not in edges and [j, i] not in edges:
                 # Simple check: if not in edges list
                 pass
                 
    # Faster complement construction
    adj_clebsch = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj_clebsch[u, v] = True
        adj_clebsch[v, u] = True
        
    comp_adj = ~adj_clebsch
    np.fill_diagonal(comp_adj, False)
    
    comp_edges_list = []
    for i in range(n):
        for j in range(i+1, n):
            if comp_adj[i, j]:
                comp_edges_list.append([i, j])
                
    if is_valid_graph(n, comp_edges_list):
        return {'n': n, 'edges': comp_edges_list}
        
    # If both fail, we try a random search with a fast validator for n=16..20
    # But first, let's see if we can find a known 20-vertex graph.
    # The task says "Graphs with 20 vertices are known".
    # We should try to construct one or find one via search.
    
    # Let's implement a fast local search for n=16 to n=20.
    # Since we have time, we can do a few random restarts.
    
    best_n = 5
    best_edges = []
    
    for target_n in range(16, 21):
        if time.time() > deadline - 10:
            break
            
        # Try to find a valid graph on target_n vertices
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Start with a random graph
            adj = np.random.rand(target_n, target_n) < 0.5
            np.fill_diagonal(adj, False)
            adj = (adj + adj.T).astype(bool)
            
            # Local search: flip edges to remove regular induced subgraphs of size 6
            # This is complex. Instead, let's just random sample and check.
            # C(20,6) = 38760. Checking 38760 subsets is fast in numpy.
            
            for trial in range(100):
                if time.time() > deadline - 5:
                    break
                adj = np.random.rand(target_n, target_n) < 0.5
                np.fill_diagonal(adj, False)
                adj = (adj + adj.T).astype(bool)
                
                valid = True
                for subset in combinations(range(target_n), 6):
                    sub_adj = adj[np.ix_(subset, subset)]
                    degrees = sub_adj.sum(axis=1)
                    if len(set(degrees)) == 1:
                        valid = False
                        break
                        
                if valid:
                    # Extract edges
                    e_list = []
                    for i in range(target_n):
                        for j in range(i+1, target_n):
                            if adj[i, j]:
                                e_list.append([i, j])
                    return {'n': target_n, 'edges': e_list}
                    
        # If we didn't find one for this n, continue to next n (smaller n is easier)
        # Actually, we want MAX n. So we should try 20 first.
        # If 20 fails, try 19, etc.
        # But random search for 20 is hard.
        
    # Fallback: Return the Clebsch graph if it was valid, or the best found.
    # Wait, the code above returned immediately if valid.
    # If we reach here, it means Clebsch and Comp-Clebsch were invalid, and random search failed.
    
    # Let's re-verify Clebsch. 
    # Actually, the Clebsch graph *does* contain 6-cycles. A 6-cycle is a 2-regular graph on 6 vertices.
    # So Clebsch is NOT valid for G(6).
    
    # We need a graph with no regular induced subgraph on 6 vertices.
    # This is a very strong condition.
    
    # Let's try the (16, 6, 2, 2) graph? No, that's not SRG.
    
    # Let's try a specific known construction for G(6).
    # According to literature, the minimum number of vertices for a graph with no regular induced subgraph on 6 vertices is 20.
    # One such graph is derived from the Petersen graph or similar.
    
    # Since we might not find a 20-vertex graph quickly with random search, 
    # let's try to hardcode a known 16-vertex graph if one exists? 
    # No, G(6) > 16 is not known to be true for all graphs, but we need *one* graph.
    # If G(6) >= 20, then there exists a graph on 19 vertices? No, G(6) is the max n such that EVERY graph on n vertices has a regular induced subgraph on 6 vertices.
    # So if G(6) = 19, then there is a graph on 19 vertices with NO regular induced subgraph on 6 vertices.
    # And there is a graph on 20 vertices with NO regular induced subgraph on 6 vertices.
    # The task says "Graphs with 20 vertices are known".
    
    # Let's try to find a 16-vertex graph that works.
    # If we can't find a 20-vertex one, a 16-vertex one is still a big jump from 5.
    
    # Let's try the complement of the Shrikhande graph?
    # Or just do a more extensive random search for n=16.
    
    for target_n in [16, 17, 18, 19, 20]:
        if time.time() > deadline - 10:
            break
            
        # Random search with local improvement?
        # For now, just random restarts.
        for restart in range(200):
            if time.time() > deadline - 5:
                break
                
            adj = np.random.rand(target_n, target_n) < 0.5
            np.fill_diagonal(adj, False)
            adj = (adj + adj.T).astype(bool)
            
            valid = True
            for subset in combinations(range(target_n), 6):
                sub_adj = adj[np.ix_(subset, subset)]
                degrees = sub_adj.sum(axis=1)
                if len(set(degrees)) == 1:
                    valid = False
                    break
                    
            if valid:
                e_list = []
                for i in range(target_n):
                    for j in range(i+1, target_n):
                        if adj[i, j]:
                            e_list.append([i, j])
                return {'n': target_n, 'edges': e_list}
                
    # If nothing works, return the 5-vertex empty graph as fallback.
    return {'n': 5, 'edges': []}
