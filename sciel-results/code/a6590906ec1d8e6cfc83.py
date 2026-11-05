import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    n = 40
    
    # Initialize with a good starting graph: use a known construction
    # Start with 8 copies of K5 (80 edges) and try to add more
    # Or start with a random dense graph and remove C6s
    
    def has_c6(adj):
        """Check if graph with adjacency matrix adj (n x n boolean) has a C6."""
        n_local = adj.shape[0]
        # For each pair (u, v), count length-3 paths
        # A length-3 path from u to v goes u-w-x-y-v where w in N(u), y in N(v), x in N(w) ∩ N(y)
        # So count = sum_{w in N(u)} sum_{y in N(v)} |N(w) ∩ N(y)|
        # But we need u not adjacent to v for the standard C6 (actually C6 can have u,v at distance 3)
        # For C6: v1-v2-v3-v4-v5-v6-v1, so v1 and v4 are at distance 3
        # So we check pairs at distance 3 (non-adjacent, not same)
        for u in range(n_local):
            for v in range(u + 1, n_local):
                if adj[u, v]:
                    continue
                # Count length-3 paths from u to v
                # Path: u - w - x - y - v
                # w in N(u), y in N(v), x in N(w) ∩ N(y)
                count = 0
                # Use bitsets for speed
                Nu = np.where(adj[u])[0]
                Nv = np.where(adj[v])[0]
                for w in Nu:
                    for y in Nv:
                        common = np.sum(adj[w] & adj[y])
                        count += common
                        if count >= 2:
                            # Check if two paths are internally disjoint
                            # For now, assume C6 exists (conservative)
                            return True
                # If count >= 2, need to verify internal disjointness
                # But for safety, if count >= 2, check more carefully
                if count >= 2:
                    # Collect all length-3 paths
                    paths = []
                    for w in Nu:
                        for y in Nv:
                            common = np.where(adj[w] & adj[y])[0]
                            for x in common:
                                if x != u and x != v:
                                    paths.append((w, x, y))
                    # Check if any two paths are internally disjoint
                    for i in range(len(paths)):
                        for j in range(i + 1, len(paths)):
                            p1 = set(paths[i])
                            p2 = set(paths[j])
                            if p1.isdisjoint(p2):
                                return True
                    # If no two are internally disjoint, no C6 through this pair
                    continue
        return False
    
    def edge_count(adj):
        return int(np.sum(adj) // 2)
    
    # Start with 8 copies of K5
    adj = np.zeros((n, n), dtype=bool)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u, v] = True
                adj[v, u] = True
    
    best_adj = adj.copy()
    best_edges = edge_count(adj)
    
    # Simulated annealing
    T = 10.0
    cooling = 0.9995
    min_T = 0.1
    
    random.seed(42)
    
    while time.time() < deadline and T > min_T:
        # Pick a random move: add or remove a random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        if adj[u, v]:
            # Remove edge
            adj[u, v] = False
            adj[v, u] = False
            new_edges = best_edges - 1
            # Removing an edge always keeps C6-free
            delta = -1
        else:
            # Add edge
            adj[u, v] = True
            adj[v, u] = True
            new_edges = best_edges + 1
            if has_c6(adj):
                # Revert
                adj[u, v] = False
                adj[v, u] = False
                T *= cooling
                continue
            delta = 1
        
        # Accept or reject
        if delta >= 0 or random.random() < np.exp(delta / T):
            best_adj = adj.copy()
            best_edges = new_edges
        else:
            # Revert
            if adj[u, v] == False and delta == 1:
                # We added it but rejected
                adj[u, v] = False
                adj[v, u] = False
            elif adj[u, v] == False and delta == -1:
                # We removed it but rejected (shouldn't happen since delta<0 only for removals... wait)
                # Actually delta=-1 is always accepted since it's a valid move
                pass
        
        T *= cooling
    
    # Return best edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return edges
