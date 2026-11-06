import numpy as np
import time
import random

def construct():
    def has_c4(adj):
        """Returns (True, edge_to_remove) if C4 exists, else (False, None)."""
        n = adj.shape[0]
        counts = adj @ adj
        # Find pairs (u,v) with adj[u,v]==0 and counts[u,v]>=2
        # We can vectorize the search for candidates
        # But we need to check the "non-edge between common neighbors" condition.
        
        # Iterate over potential u, v
        # To speed up, we can precompute neighbors.
        
        # For n=47, simple loops are okay if not too deep.
        # Let's try to find one C4.
        
        # Get indices where adj is 0 and counts >= 2
        # Note: counts[u,u] is degree, we don't care about u==v.
        
        for u in range(n):
            for v in range(u+1, n):
                if adj[u,v] == 0:
                    if counts[u,v] >= 2:
                        # Check common neighbors
                        # common = np.where(adj[u] & adj[v])[0]
                        # If len(common) >= 2, check if subgraph is complete
                        common = np.nonzero(adj[u] & adj[v])[0]
                        if len(common) >= 2:
                            # Check if any pair in common is not connected
                            # We can just check the first few pairs
                            for i in range(len(common)):
                                for j in range(i+1, len(common)):
                                    a = common[i]
                                    b = common[j]
                                    if adj[a,b] == 0:
                                        return True, (u, a) # Remove (u,a)
        return False, None

    def can_add(adj, u, v):
        """Check if adding (u,v) creates a C4."""
        # C4: u-a-v-b-u
        # Requires a in N(u), b in N(v), (a,b) in E.
        N_u = np.nonzero(adj[u])[0]
        N_v = np.nonzero(adj[v])[0]
        # Check if any a in N_u is connected to any b in N_v
        # adj[N_u, :][:, N_v]
        if len(N_u) == 0 or len(N_v) == 0:
            return True
        # Efficient check:
        # sum of adj[N_u][:, N_v]
        if np.sum(adj[np.ix_(N_u, N_v)]) > 0:
            return False
        return True

    def repair(adj, n, target_deg):
        # Loop
        for _ in range(20):
            # Remove C4s
            while True:
                found, edge = has_c4(adj)
                if not found:
                    break
                u, v = edge
                adj[u,v] = 0
                adj[v,u] = 0
            
            # Add edges
            degs = adj.sum(axis=1)
            for u in range(n):
                if degs[u] < target_deg:
                    candidates = np.nonzero(adj[u] == 0)[0]
                    random.shuffle(candidates)
                    for v in candidates:
                        if degs[v] >= n-1: continue
                        if can_add(adj, u, v):
                            adj[u,v] = 1
                            adj[v,u] = 1
                            degs[u] += 1
                            degs[v] += 1
                            if degs[u] >= target_deg:
                                break

    def is_valid(adj, n):
        if np.min(adj.sum(axis=1)) < n - 39:
            return False
        if has_c4(adj)[0]:
            return False
        return True

    best = None
    start = time.time()
    
    for n in [47, 46]:
        target = n - 39
        for _ in range(20):
            if time.time() - start > 100:
                break
            # Random start
            adj = np.random.randint(0, 2, (n, n))
            adj = np.triu(adj, 1) + np.triu(adj, 1).T
            np.fill_diagonal(adj, 0)
            
            repair(adj, n, target)
            
            if is_valid(adj, n):
                edges = [[u, v] for u in range(n) for v in range(u+1, n) if adj[u,v]==1]
                if best is None or n > best['n']:
                    best = {'n': n, 'edges': edges}
                if n == 47:
                    return best

    if best:
        return best
    return {'n': 39, 'edges': []}
