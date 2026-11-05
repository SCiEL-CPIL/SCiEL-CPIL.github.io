import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def has_time():
        return elapsed() < DEADLINE
    
    # Bitmask representation: adj[i] is a bitmask where bit j is set if edge (i,j) exists
    def make_graph(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj
    
    def edges_from_adj(n, adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    def degree_in_subset(adj, subset_mask, v, n):
        """Compute degree of vertex v within the subset defined by subset_mask."""
        # Count neighbors of v that are in subset_mask
        neighbors_in_subset = adj[v] & subset_mask
        return bin(neighbors_in_subset).count('1')
    
    def is_regular_induced(adj, subset_mask, n):
        """Check if the induced subgraph on subset_mask is regular."""
        # Get vertices in subset
        verts = []
        m = subset_mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            verts.append(v)
            m ^= lb
        
        k = len(verts)
        if k < 6:
            return False
        
        # Compute degrees of each vertex within the subset
        degs = []
        for v in verts:
            degs.append(degree_in_subset(adj, subset_mask, v, n))
        
        # Check if all degrees are the same
        return len(set(degs)) == 1
    
    def find_violation(adj, n):
        """Find any subset of size >= 6 that induces a regular subgraph."""
        # Check subsets of size 6 first (most likely to exist, cheapest to find)
        # Then size 7, 8, ...
        # Use combinations but with early termination
        from itertools import combinations
        
        for k in range(6, n+1):
            if not has_time():
                return None
            # For large k, the number of combinations is huge.
            # We'll sample or use a smarter approach for k > 8
            if k <= 8:
                for combo in combinations(range(n), k):
                    if not has_time():
                        return None
                    mask = 0
                    for v in combo:
                        mask |= (1 << v)
                    if is_regular_induced(adj, mask, n):
                        return list(combo)
            else:
                # For larger subsets, sampling
                # A regular induced subgraph on k vertices is rare in random graphs
                # but we need to be thorough. Let's sample some.
                samples = 200
                for _ in range(samples):
                    if not has_time():
                        return None
                    combo = random.sample(range(n), k)
                    mask = 0
                    for v in combo:
                        mask |= (1 << v)
                    if is_regular_induced(adj, mask, n):
                        return list(combo)
        return None
    
    def random_graph(n, p=0.5):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    edges.append([i, j])
        return edges
    
    def local_search(adj, n, max_iters=5000):
        """Try to fix violations by flipping edges."""
        for it in range(max_iters):
            if not has_time():
                break
            violation = find_violation(adj, n)
            if violation is None:
                return True, adj
            # Violation found: try to break it by flipping an edge
            # Pick a random edge within the violating subset or incident to it
            vset = set(violation)
            # Candidate edges to flip: edges within vset or edges from vset to outside
            candidates = []
            for i in range(n):
                for j in range(i+1, n):
                    if i in vset or j in vset:
                        candidates.append((i, j))
            if not candidates:
                break
            # Try a random candidate
            u, v = random.choice(candidates)
            # Flip the edge
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        return False, adj
    
    # Strategy: start small, grow n, use local search to fix violations
    best_n = 5
    best_edges = []
    
    # Start from n=6 and try to grow
    for target_n in range(6, 23):
        if not has_time():
            break
        success = False
        for attempt in range(20):
            if not has_time():
                break
            # Create a random graph on target_n vertices
            edges = random_graph(target_n, p=0.5)
            adj = make_graph(target_n, edges)
            
            # Run local search
            ok, adj = local_search(adj, target_n, max_iters=2000)
            if ok:
                # Verify thoroughly
                if find_violation(adj, target_n) is None:
                    best_n = target_n
                    best_edges = edges_from_adj(target_n, adj)
                    success = True
                    break
        
        if not success and target_n > best_n + 1:
            # Can't even get target_n, stop trying larger
            break
    
    # If we couldn't do better than 5, at least return 5
    if best_n < 6:
        return {'n': 5, 'edges': []}
    
    return {'n': best_n, 'edges': best_edges}
