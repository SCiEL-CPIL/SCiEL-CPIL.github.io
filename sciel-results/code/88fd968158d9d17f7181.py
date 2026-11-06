import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s for safety
    
    def has_triangle(adj, u, v):
        """Check if adding edge (u,v) creates a triangle using bitsets."""
        return (adj[u] & adj[v]) != 0
    
    def max_independent_set_size(adj, n):
        """
        Exact branch and bound for maximum independent set size.
        adj: list of int bitsets, adj[i] has bit j set if (i,j) is an edge.
        n: number of vertices.
        Returns the size of the maximum independent set.
        """
        best = [0]
        
        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Remap vertices: new index -> old index
        # We'll work in the original indexing but branch in order
        
        def search(candidates, current_size, depth):
            nonlocal best
            if not candidates:
                return
            if current_size + bin(candidates).count('1') <= best[0]:
                return
            
            # Pick vertex with highest degree in candidates
            # For speed, just pick the first in our order that's in candidates
            v = -1
            for u in order:
                if (candidates >> u) & 1:
                    v = u
                    break
            
            if v == -1:
                return
            
            # Branch 1: Include v
            # Remove v and its neighbors from candidates
            new_cands = candidates & ~(1 << v) & ~adj[v]
            search(new_cands, current_size + 1, depth + 1)
            if current_size + 1 > best[0]:
                best[0] = current_size + 1
            
            # Branch 2: Exclude v
            new_cands = candidates & ~(1 << v)
            search(new_cands, current_size, depth + 1)
        
        search((1 << n) - 1, 0, 0)
        return best[0]
    
    def find_large_independent_set(adj, n, target=10):
        """Find an independent set of size >= target, or return None."""
        # Use greedy with backtracking
        def search(candidates, current_set, needed):
            if needed == 0:
                return current_set
            if bin(candidates).count('1') < needed:
                return None
            
            # Pick vertex with min degree in candidates
            v = -1
            min_deg = n + 1
            for u in range(n):
                if (candidates >> u) & 1:
                    deg = bin(adj[u] & candidates).count('1')
                    if deg < min_deg:
                        min_deg = deg
                        v = u
            
            if v == -1:
                return None
            
            # Try including v
            new_cands = candidates & ~(1 << v) & ~adj[v]
            res = search(new_cands, current_set + [v], needed - 1)
            if res is not None:
                return res
            
            # Try excluding v
            new_cands = candidates & ~(1 << v)
            return search(new_cands, current_set, needed)
        
        return search((1 << n) - 1, [], target)
    
    def try_construct(n, seed):
        rng = random.Random(seed)
        
        # Initialize empty graph with bitsets
        adj = [0] * n
        
        # Phase 1: Greedy triangle-free edge addition in random order
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        
        for u, v in edges:
            if not has_triangle(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Phase 2: Local search to reduce alpha
        for iteration in range(50):
            alpha = max_independent_set_size(adj, n)
            if alpha < 10:
                return adj
            
            # Find a large independent set
            is_set = find_large_independent_set(adj, n, 10)
            if is_set is None:
                # Should not happen if alpha >= 10, but just in case
                break
            
            # Try to add edges within is_set to break it
            added = False
            for i in range(len(is_set)):
                for j in range(i + 1, len(is_set)):
                    u = is_set[i]
                    v = is_set[j]
                    if not ((adj[u] >> v) & 1):
                        if not has_triangle(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added = True
            
            if not added:
                # Try edge swaps: remove an edge, add two new ones
                # Find an edge (a,b) such that removing it allows adding (a,c) and (b,d)
                # This is complex; skip for now
                break
        
        return None
    
    # Try n from 40 down to 30
    for n in range(40, 29, -1):
        if time.time() > deadline:
            break
        
        max_restarts = 500
        for seed in range(max_restarts):
            if time.time() > deadline:
                break
            res = try_construct(n, seed)
            if res is not None:
                # Convert bitsets to edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (res[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Fallback: C5 blow-up with k=4, n=20, alpha=8
    n = 20
    adj = [0] * n
    for g1 in range(5):
        for g2 in range(g1 + 1, 5):
            if (g2 - g1) % 5 in [1, 4]:
                for u in range(g1 * 4, (g1 + 1) * 4):
                    for v in range(g2 * 4, (g2 + 1) * 4):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
