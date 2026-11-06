import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n = 20
    # Precompute all triples of vertices
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    
    def is_valid(adj):
        """
        adj: list of integers representing adjacency bitmasks.
        Returns True if graph is K3,3-free.
        """
        for i, j, k in triples:
            # Common neighbors of i, j, k
            common = adj[i] & adj[j] & adj[k]
            # If there are at least 3 common neighbors, we can form K3,3
            # by picking any 3 from common.
            if common.bit_count() >= 3:
                return False
        return True

    def count_edges(adj):
        # Sum of bits in all adjacency matrices, divided by 2
        # But we can just maintain a counter or compute it.
        # Computing is fast enough for n=20.
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def get_violating_triple(adj):
        """
        Returns a triple (i, j, k) such that common neighbors >= 3, or None.
        """
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k, common)
        return None

    def repair(adj):
        """
        Removes edges until the graph is valid.
        Returns the number of edges removed.
        """
        removed = 0
        while True:
            v = get_violating_triple(adj)
            if v is None:
                break
            i, j, k, common = v
            # We need to remove an edge to break the K3,3.
            # The K3,3 is formed by {i, j, k} and any 3 nodes in 'common'.
            # Let's pick 3 specific nodes from 'common'.
            # To be effective, we should pick an edge that is part of many violations?
            # For now, pick a random edge from the complete bipartite graph between {i,j,k} and common.
            # Or just pick a random node m in common and remove edge (i, m).
            # Removing (i, m) breaks this specific K3,3.
            
            # Get list of nodes in common
            # Since n=20, we can iterate
            candidates = []
            for m in range(n):
                if common & (1 << m):
                    candidates.append(m)
            
            # Pick a random candidate
            m = random.choice(candidates)
            # Remove edge (i, m)
            adj[i] &= ~(1 << m)
            adj[m] &= ~(1 << i)
            removed += 1
            
        return removed

    def spectral_init(rng):
        """
        Create a dense graph using spectral method:
        1. Generate random symmetric matrix
        2. Compute top eigenvector
        3. Use outer product to create adjacency
        4. Threshold to create binary adjacency
        """
        # Generate random symmetric matrix
        A = rng.random((n, n))
        A = (A + A.T) / 2
        
        # Compute eigenvalues and eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(A)
            # Take the top eigenvector (largest eigenvalue)
            idx = np.argmax(eigvals)
            v = eigvecs[:, idx]
            
            # Make it non-negative (flip sign if needed)
            if v.sum() < 0:
                v = -v
            
            # Create adjacency matrix from outer product
            # Use threshold based on mean
            outer = np.outer(v, v)
            mean_val = np.mean(outer)
            
            # Create binary adjacency
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if outer[i, j] > mean_val:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            return adj
        except:
            # Fallback to random
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.7:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            return adj

    best_edges = 0
    best_adj = None
    
    # Try multiple random starts
    for seed in range(200):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(seed)
        rng = np.random.default_rng(seed)
        
        # Use spectral initialization
        adj = spectral_init(rng)
        
        # Repair to valid state
        repair(adj)
        
        current_edges = count_edges(adj)
        
        # Local Search: Try to add edges
        # We maintain a list of non-edges to pick from
        # Or just pick random pairs
        no_improve_count = 0
        max_iterations = 10000
        iter_count = 0
        
        while iter_count < max_iterations:
            if time.time() > deadline - 2.0:
                break
            
            iter_count += 1
            
            # Try to add a random non-edge
            # Generate random pair
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            # Check if edge exists
            if adj[u] & (1 << v):
                continue
            
            # Try adding it
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid(adj):
                current_edges += 1
                no_improve_count = 0
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                no_improve_count += 1
                
                # If stuck for a while, remove a random edge to allow progress
                if no_improve_count > 50:
                    # Find a random existing edge and remove it
                    # Pick random u, then random v in adj[u]
                    while True:
                        u_rm = random.randint(0, n - 1)
                        if adj[u_rm] == 0:
                            continue
                        # Pick random set bit
                        # Simple way: get list of neighbors
                        neighbors = []
                        for w in range(n):
                            if adj[u_rm] & (1 << w):
                                neighbors.append(w)
                        if neighbors:
                            v_rm = random.choice(neighbors)
                            adj[u_rm] &= ~(1 << v_rm)
                            adj[v_rm] &= ~(1 << u_rm)
                            current_edges -= 1
                            break
                    no_improve_count = 0
                    
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [x for x in adj]

    # Construct edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
