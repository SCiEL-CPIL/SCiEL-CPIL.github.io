import random
import numpy as np

def construct():
    n = 20
    # Use a fixed seed for reproducibility if needed, but random is fine
    # random.seed(42) 
    
    # Start with a random graph
    # Probability 0.4 gives ~ 190 * 0.4 = 76 edges
    # K3,3 free graphs on 20 vertices can have up to ~ 80-100 edges? 
    # ex(20, K3,3) is likely around 80-90.
    
    # Initialize adjacency matrix
    adj = np.zeros((n, n), dtype=int)
    
    # Generate random edges
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.4:
                adj[i, j] = 1
                adj[j, i] = 1
                
    def is_k33_free(adj):
        # Check all triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    # Common neighbors of i, j, k
                    common = np.logical_and(np.logical_and(adj[i], adj[j]), adj[k])
                    if np.sum(common) > 2:
                        return False, (i, j, k)
        return True, None

    # Remove edges to make it K3,3 free
    # We loop until it's free
    # To be efficient, we can just check and remove one edge from a violating triple
    while True:
        is_free, triple = is_k33_free(adj)
        if is_free:
            break
        i, j, k = triple
        # Find a common neighbor to break the connection
        common = np.where(np.logical_and(np.logical_and(adj[i], adj[j]), adj[k]) == 1)[0]
        if len(common) > 0:
            # Pick a random common neighbor and remove an edge involving it
            # Actually, removing any edge from the K3,3 formed by {i,j,k} + 3 common neighbors works.
            # Let's just remove an edge between one of i,j,k and a common neighbor
            target = common[random.randint(0, len(common)-1)]
            # Remove edge (i, target) or (j, target) or (k, target)
            # Removing (i, target) is sufficient to reduce common neighbors for this triple
            adj[i, target] = 0
            adj[target, i] = 0
        else:
            # Should not happen if is_k33_free returned False
            break

    # Now greedily add edges
    # Iterate over all non-edges
    # To avoid infinite loops or heavy computation, we do a few passes
    for _ in range(10):
        added = False
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] == 0:
                    # Temporarily add
                    adj[i, j] = 1
                    adj[j, i] = 1
                    is_free, _ = is_k33_free(adj)
                    if is_free:
                        added = True
                    else:
                        # Revert
                        adj[i, j] = 0
                        adj[j, i] = 0
        if not added:
            break

    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
                
    return edges
