import random
import time
import sys
import numpy as np

def construct():
    n = 20
    start = time.time()
    deadline = start + 145.0  # Safety margin

    def verify(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if v in adj[u]:
                return None
            adj[u].add(v)
            adj[v].add(u)
        for u in range(n):
            for v in range(u + 1, n):
                if len(adj[u] & adj[v]) > 2:
                    return None
        return edges

    def fallback():
        e = []
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    e.append([base + a, base + b])
        return e

    best_edges = fallback()
    best_count = len(best_edges)

    def try_add_edge(adj, common, u, v):
        """
        Check if edge (u, v) can be added.
        Returns True if valid.
        """
        if adj[u, v]:
            return False
        
        # Check neighbors of v
        neighbors_v = np.where(adj[v] == 1)[0]
        for w in neighbors_v:
            if w == u:
                continue
            if common[u, w] >= 2:
                return False
                
        # Check neighbors of u
        neighbors_u = np.where(adj[u] == 1)[0]
        for w in neighbors_u:
            if w == v:
                continue
            if common[v, w] >= 2:
                return False
                
        return True

    def add_edge(adj, common, u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        
        neighbors_v = np.where(adj[v] == 1)[0]
        for w in neighbors_v:
            if w == u:
                continue
            common[u, w] += 1
            common[w, u] += 1
            
        neighbors_u = np.where(adj[u] == 1)[0]
        for w in neighbors_u:
            if w == v:
                continue
            common[v, w] += 1
            common[w, v] += 1

    def remove_edge(adj, common, u, v):
        neighbors_v = np.where(adj[v] == 1)[0]
        for w in neighbors_v:
            if w == u:
                continue
            common[u, w] -= 1
            common[w, u] -= 1
            
        neighbors_u = np.where(adj[u] == 1)[0]
        for w in neighbors_u:
            if w == v:
                continue
            common[v, w] -= 1
            common[w, v] -= 1
            
        adj[u, v] = 0
        adj[v, u] = 0

    def greedy_build(seed):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=np.uint8)
        common = np.zeros((n, n), dtype=np.int16)
        
        # Init with 5 K4s
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    u = base + a
                    v = base + b
                    add_edge(adj, common, u, v)
                    
        # List of all non-edges
        non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
        rng.shuffle(non_edges)
        
        added = 0
        for u, v in non_edges:
            if try_add_edge(adj, common, u, v):
                add_edge(adj, common, u, v)
                added += 1
                
        # Local search: try to add more by re-shuffling and retrying?
        # Or just do a pass of 1-for-2 swaps
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Collect current edges
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
            rng.shuffle(edges)
            
            for rem_u, rem_v in edges:
                if time.time() > deadline - 10.0:
                    break
                
                # Save state
                adj_save = adj.copy()
                common_save = common.copy()
                
                remove_edge(adj, common, rem_u, rem_v)
                
                # Try to add 2 edges
                added_count = 0
                # Get current non-edges
                # We can just iterate through all pairs
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v]:
                            continue
                        if u == rem_u and v == rem_v:
                            continue
                        if try_add_edge(adj, common, u, v):
                            add_edge(adj, common, u, v)
                            added_count += 1
                            if added_count == 2:
                                break
                    if added_count == 2:
                        break
                        
                if added_count == 2:
                    improved = True
                    # Accept the change
                    break
                else:
                    # Revert
                    adj[:] = adj_save
                    common[:] = common_save
                    
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    result.append([u, v])
        return result

    # Run multiple trials
    for seed in range(100):
        if time.time() > deadline - 20.0:
            break
        edges = greedy_build(seed)
        valid_edges = verify(edges)
        if valid_edges is not None:
            if len(valid_edges) > best_count:
                best_count = len(valid_edges)
                best_edges = valid_edges
                # Optional: polish best_edges further if time permits
                # But greedy_build already does local search.

    return best_edges

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
