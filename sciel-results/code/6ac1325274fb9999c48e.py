import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    n = 40
    
    def build_k5_base():
        """Build 8 disjoint K5s."""
        edges = set()
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    edges.add((base + a, base + b))
        return edges
    
    def has_c6_with_edge(adj, u, v):
        """Check if adding edge (u,v) creates a C6.
        A C6 through (u,v) means there's a path u-a-b-c-d-v of length 5
        with all distinct vertices, none being u or v."""
        # Find all vertices at distance 1 from u (excluding v)
        neighbors_u = [x for x in adj[u] if x != v]
        if not neighbors_u:
            return False
        
        # BFS/DFS to find paths of length 5 from u to v
        # Path: u -> a -> b -> c -> d -> v
        # We need to check if there's a simple path of exactly length 5 from u to v
        # that doesn't use edge (u,v)
        
        # Use DFS with pruning
        # a must be neighbor of u (not v)
        # b must be neighbor of a (not u, not v)
        # c must be neighbor of b (not a, not u, not v)
        # d must be neighbor of c (not b, not u, not v)
        # v must be neighbor of d
        
        for a in neighbors_u:
            neighbors_a = [x for x in adj[a] if x not in (u, v)]
            for b in neighbors_a:
                neighbors_b = [x for x in adj[b] if x not in (u, v, a)]
                for c in neighbors_b:
                    neighbors_c = [x for x in adj[c] if x not in (u, v, a, b)]
                    for d in neighbors_c:
                        if v in adj[d]:
                            return True
        return False
    
    def try_greedy_addition(edges, adj, random_seed):
        """Try to add edges greedily, starting from the K5 base."""
        rng = random.Random(random_seed)
        edge_set = set(edges)
        
        # Build adjacency
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Candidate edges: all non-edges
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        non_edges = [(i, j) for i, j in all_pairs if (i, j) not in edge_set and (j, i) not in edge_set]
        
        rng.shuffle(non_edges)
        
        added = 0
        for u, v in non_edges:
            if time.time() > deadline - 5:
                break
            if u == v:
                continue
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            if has_c6_with_edge(adj, u, v):
                continue
            # Add the edge
            edge_set.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            added += 1
        
        return edge_set
    
    def verify_c6_free(edge_set):
        """Verify the graph is C6-free by checking all possible C6s."""
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check for C6: v0-v1-v2-v3-v4-v5-v0
        # For efficiency, check all 6-cycles
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 in (v0, v1):
                        continue
                    for v3 in adj[v2]:
                        if v3 in (v0, v1, v2):
                            continue
                        for v4 in adj[v3]:
                            if v4 in (v0, v1, v2, v3):
                                continue
                            for v5 in adj[v4]:
                                if v5 in (v0, v1, v2, v3, v4):
                                    continue
                                if v0 in adj[v5]:
                                    # Found a C6
                                    return False
        return True
    
    # Start with K5 base
    base_edges = build_k5_base()
    base_adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    best_edges = base_edges
    best_count = len(base_edges)
    
    # Try multiple random restarts with greedy addition
    for seed in range(200):
        if time.time() > deadline - 10:
            break
        
        # Work on a copy
        adj_copy = {i: set(base_adj[i]) for i in range(n)}
        result = try_greedy_addition(base_edges, adj_copy, seed)
        
        if len(result) > best_count:
            best_count = len(result)
            best_edges = result
    
    # Final verification
    if verify_c6_free(best_edges):
        return [[u, v] for u, v in best_edges]
    
    # If verification fails (shouldn't happen), return the base
    return [[u, v] for u, v in base_edges]
