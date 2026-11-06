import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s kill

    n = 20
    # Precompute all triples of vertices
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triples.append((i, j, k))
    num_triples = len(triples)

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
            # Get list of nodes in common
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

    def random_dense_init(rng):
        """
        Create a dense random graph initialization.
        """
        adj = [0] * n
        # Start with a fairly dense random graph
        p = 0.6
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def spectral_init(rng):
        """
        Create a dense graph using spectral method.
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

    def block_design_init():
        """
        Try to construct a graph based on a 2-(v, k, lambda) design.
        Vertices are points, edges connect points in same block.
        For K3,3-free, we need any 3 points to have at most 2 common neighbors.
        
        A symmetric 2-(v, k, lambda) design has v points, v blocks, each block size k,
        each point in r blocks, any two points in lambda blocks.
        
        If we take the incidence graph, it's bipartite. But we want a graph on v vertices.
        Let's use the block intersection graph: vertices are blocks, two blocks are adjacent
        if they share a point. For a 2-(v,k,lambda) design, two blocks share lambda points.
        So the block graph has every pair of vertices with exactly lambda common neighbors?
        No, that's not right.
        
        Actually, let's think about this differently.
        A strongly regular graph (v, k, lambda, mu) has:
        - v vertices
        - each vertex has k neighbors
        - adjacent vertices have lambda common neighbors
        - non-adjacent vertices have mu common neighbors
        
        For K3,3-free, we need any 3 vertices to have at most 2 common neighbors.
        In an srg, any 3 vertices have at most max(lambda, mu) common neighbors.
        So we need max(lambda, mu) <= 2.
        
        Known srgs with max(lambda, mu) <= 2:
        - Petersen: (10, 3, 0, 1)
        - Clebsch: (16, 5, 0, 2)
        - Shrikhande: (16, 6, 2, 2)
        - Hoffman-Singleton: (50, 7, 0, 1)
        
        For n=20, we can try to find or construct an srg-like graph.
        Or we can use a known construction.
        
        Let's try to use the complement of a known graph or construct one.
        """
        # For now, return None to use other methods
        return None

    best_edges = 0
    best_adj = None
    
    # Try multiple random starts
    num_starts = 300
    for seed in range(num_starts):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(seed)
        rng = np.random.default_rng(seed)
        
        # Choose initialization method
        if seed % 3 == 0:
            adj = random_dense_init(rng)
        elif seed % 3 == 1:
            adj = spectral_init(rng)
        else:
            adj = random_dense_init(rng)
        
        # Repair to valid state
        repair(adj)
        
        current_edges = count_edges(adj)
        
        # Local Search: Try to add edges
        no_improve_count = 0
        max_iterations = 15000
        iter_count = 0
        
        while iter_count < max_iterations:
            if time.time() > deadline - 2.0:
                break
            
            iter_count += 1
            
            # Try to add a random non-edge
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
                if no_improve_count > 30:
                    # Find a random existing edge and remove it
                    while True:
                        u_rm = random.randint(0, n - 1)
                        if adj[u_rm] == 0:
                            continue
                        # Pick random set bit
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
