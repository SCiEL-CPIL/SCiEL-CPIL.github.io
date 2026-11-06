import time
import random
import numpy as np

def construct():
    """
    Constructs a graph with no K5 and no independent 5-set.
    Returns Exoo's 42-vertex graph if found/known, otherwise falls back to 37-vertex Paley graph.
    """
    start_time = time.time()
    timeout = 300  # Safety margin before 330s kill
    
    # Fallback: Paley graph on 37 vertices
    def paley_graph(q):
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    # Helper to check validity and return bad sets
    def find_bad_sets(adj, N):
        """
        adj: list of ints, adj[i] has bit j set if i~j.
        Returns a list of (type, set_of_5_vertices) for bad sets.
        type: 1 for Clique, 0 for Independent Set.
        """
        bads = []
        # Find K5s
        four_cliques = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    # Common neighbors of i and j, strictly greater than j
                    common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                    # Iterate k in common
                    k_list = [k for k in range(j + 1, N) if common & (1 << k)]
                    for k in k_list:
                        for l in range(k + 1, N):
                            if common & (1 << l) and adj[k] & (1 << l):
                                four_cliques.append((i, j, k, l))
        
        for (i, j, k, l) in four_cliques:
            # Common neighbors of i, j, k, l
            common = adj[i] & adj[j] & adj[k] & adj[l]
            # Remove i, j, k, l from common
            common &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
            # If any bit is set, it's a K5
            if common:
                # Find one such m
                m = (common & -common).bit_length() - 1
                bads.append((1, (i, j, k, l, m)))
                
        # Find Independent 5-sets (K5 in complement)
        mask = (1 << N) - 1
        comp_adj = [~adj[i] & mask for i in range(N)]
        for i in range(N):
            comp_adj[i] &= ~(1 << i)
            
        four_ind_cliques = []
        for i in range(N):
            for j in range(i + 1, N):
                if comp_adj[i] & (1 << j):
                    common = comp_adj[i] & comp_adj[j] & ~((1 << (j + 1)) - 1)
                    k_list = [k for k in range(j + 1, N) if common & (1 << k)]
                    for k in k_list:
                        for l in range(k + 1, N):
                            if common & (1 << l) and comp_adj[k] & (1 << l):
                                four_ind_cliques.append((i, j, k, l))
        
        for (i, j, k, l) in four_ind_cliques:
            common = comp_adj[i] & comp_adj[j] & comp_adj[k] & comp_adj[l]
            common &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
            if common:
                m = (common & -common).bit_length() - 1
                bads.append((0, (i, j, k, l, m)))
                
        return bads

    # Try to find a 42-vertex graph using local search
    N = 42
    
    # Local Search
    max_restarts = 100
    for restart in range(max_restarts):
        if time.time() - start_time > timeout - 20:
            break
            
        # Initialize random graph with slight bias towards 50% density
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Simulated Annealing / Random Walk
        for step in range(2000):
            if time.time() - start_time > timeout - 10:
                break
                
            bads = find_bad_sets(adj, N)
            if not bads:
                # Found a valid graph!
                edges = []
                for i in range(N):
                    for j in range(i + 1, N):
                        if adj[i] & (1 << j):
                            edges.append((i, j))
                return N, edges
            
            # Pick a random bad set
            is_clique, v_set = random.choice(bads)
            i, j, k, l, m = v_set
            
            # For a clique, remove an edge. For an independent set, add an edge.
            # Pick a random pair from the 5 vertices.
            vertices = list(v_set)
            random.shuffle(vertices)
            u, v = vertices[0], vertices[1]
            
            # Flip edge u-v
            if adj[u] & (1 << v):
                # Remove edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
    # Fallback
    return paley_graph(37)
