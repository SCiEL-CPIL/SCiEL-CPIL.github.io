"""Rewritten program for R(C4,K11) construction."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a C4-free graph with no independent set of size 11.
    Uses a structured base graph and local search to minimize independence number.
    """
    
    def is_c4_free(edges, n):
        """Check if graph is C4-free using adjacency set."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return False
        return True
    
    def greedy_max_independent_set(adj, n, num_trials=100, seed=42):
        """Find maximum independent set size using greedy with multiple random orderings."""
        rng = random.Random(seed)
        best = 0
        for _ in range(num_trials):
            order = list(range(n))
            rng.shuffle(order)
            independent = set()
            for v in order:
                if all(u not in independent for u in adj[v]):
                    independent.add(v)
            if len(independent) > best:
                best = len(independent)
        return best
    
    def find_independent_set(adj, n, target_size, rng):
        """Try to find an independent set of at least target_size."""
        for _ in range(200):
            order = list(range(n))
            rng.shuffle(order)
            independent = set()
            for v in order:
                if all(u not in independent for u in adj[v]):
                    independent.add(v)
            if len(independent) >= target_size:
                return independent
        return None
    
    n = 39
    
    # Build a structured base graph: use a 3-partite-like structure
    # Partition vertices into 3 groups of 13
    # Within each group, add a cycle (13-cycle)
    # Between groups, add edges carefully to avoid C4
    
    adj = [set() for _ in range(n)]
    edges = []
    
    # Group sizes: 13, 13, 13
    groups = [list(range(0, 13)), list(range(13, 26)), list(range(26, 39))]
    
    # Add cycles within each group
    for g in groups:
        for i in range(len(g)):
            u = g[i]
            v = g[(i+1) % len(g)]
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    # Add edges between groups using a structured pattern
    # Between group 0 and group 1: connect g0[i] to g1[i] and g1[i+1 mod 13]
    # But this may create C4s. Let's be more careful.
    
    # Use a bipartite matching-like structure between groups
    # For each i in 0..12, connect groups[0][i] to groups[1][i]
    for i in range(13):
        u = groups[0][i]
        v = groups[1][i]
        common = adj[u] & adj[v]
        if len(common) < 2:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    # Connect groups[0][i] to groups[2][i]
    for i in range(13):
        u = groups[0][i]
        v = groups[2][i]
        common = adj[u] & adj[v]
        if len(common) < 2:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    # Connect groups[1][i] to groups[2][i+1 mod 13]
    for i in range(13):
        u = groups[1][i]
        v = groups[2][(i+1) % 13]
        common = adj[u] & adj[v]
        if len(common) < 2:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    # Now try to add more edges to reduce independence number
    rng = random.Random(42)
    
    # Greedy edge addition: try all non-adjacent pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if v in adj[u]:
            continue
        common = adj[u] & adj[v]
        if len(common) < 2:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    # Check current independence number
    alpha = greedy_max_independent_set(adj, n, 200, seed=42)
    
    # Local search: if alpha > 10, try to add edges within independent sets
    max_iterations = 2000
    iteration = 0
    while alpha > 10 and iteration < max_iterations:
        iteration += 1
        # Find an independent set
        indep = find_independent_set(adj, n, 11, rng)
        if indep is None:
            alpha = 10
            break
        
        # Try to add an edge within the independent set
        indep_list = list(indep)
        rng.shuffle(indep_list)
        added = False
        for i in range(len(indep_list)):
            for j in range(i+1, len(indep_list)):
                u, v = indep_list[i], indep_list[j]
                if v not in adj[u]:
                    common = adj[u] & adj[v]
                    if len(common) < 2:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
                        added = True
                        break
            if added:
                break
        
        if not added:
            # No more edges can be added within this independent set
            # Try a different approach: swap edges
            break
        
        alpha = greedy_max_independent_set(adj, n, 200, seed=iteration % 1000)
    
    # Final verification
    if not is_c4_free(edges, n):
        # Fallback to 30-vertex construction
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    alpha_final = greedy_max_independent_set(adj, n, 500, seed=12345)
    if alpha_final > 10:
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
