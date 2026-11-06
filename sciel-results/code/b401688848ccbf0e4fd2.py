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
        # Precompute pair intersections for speed
        pair_common = {}
        for i in range(n):
            for j in range(i + 1, n):
                pair_common[(i, j)] = adj[i] & adj[j]
        
        for i, j, k in triples:
            # Common neighbors of i, j, k
            common = pair_common[(i, j)] & adj[k]
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
        Uses a smarter strategy: count how many violations each edge participates in,
        and remove the edge with the highest count.
        Returns the number of edges removed.
        """
        removed = 0
        while True:
            v = get_violating_triple(adj)
            if v is None:
                break
            i, j, k, common = v
            # Count how many times each edge in the violating structure appears
            # The violating K3,3 involves vertices {i,j,k} and 3+ common neighbors
            # We want to remove an edge that breaks this and potentially other violations
            
            # Get all common neighbors
            common_nodes = []
            for m in range(n):
                if common & (1 << m):
                    common_nodes.append(m)
            
            # For each edge (x, m) where x in {i,j,k} and m in common_nodes,
            # count how many triples (a,b,c) have m as a common neighbor and include x
            # This is expensive, so use a heuristic: remove an edge that is in many potential K3,3s
            
            # Simple heuristic: pick the edge (x, m) where x has highest degree among {i,j,k}
            # and m has highest degree among common_nodes
            x = max([i, j, k], key=lambda v: adj[v].bit_count())
            m = max(common_nodes, key=lambda v: adj[v].bit_count())
            
            # Remove edge (x, m)
            adj[x] &= ~(1 << m)
            adj[m] &= ~(1 << x)
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
    
    # Try multiple random starts - reduced to allow more time for polish
    for seed in range(100):
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
                    
        # Final polish: systematically try all non-edges in order
        if time.time() < deadline - 3.0:
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > deadline - 1.0:
                        break
                    if adj[u] & (1 << v):
                        continue
                    # Try adding
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if is_valid(adj):
                        current_edges += 1
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [x for x in adj]

    # Final global polish on the best solution found
    if best_adj is not None and time.time() < deadline - 2.0:
        # Do one more systematic pass on the best solution
        for u in range(n):
            for v in range(u + 1, n):
                if time.time() > deadline - 0.5:
                    break
                if best_adj[u] & (1 << v):
                    continue
                # Try adding
                best_adj[u] |= (1 << v)
                best_adj[v] |= (1 << u)
                if is_valid(best_adj):
                    best_edges += 1
                else:
                    best_adj[u] &= ~(1 << v)
                    best_adj[v] &= ~(1 << u)
    
    # Construct edge list from best_adj
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
                
    return edges
