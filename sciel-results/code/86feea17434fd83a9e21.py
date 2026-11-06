import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s for safety
    
    def has_triangle(adj, u, v):
        # Check if adding edge (u, v) creates a triangle
        # We need N(u) and N(v) to have no common neighbors
        # Note: adj is a list of sets
        return bool(adj[u] & adj[v])
        
    def is_triangle_free(adj, n):
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def max_independent_set_size(adj, n):
        # Exact branch and bound for alpha
        # Returns the size of the maximum independent set
        # Optimized for n <= 40
        
        # Sort vertices by degree
        degrees = [len(adj[i]) for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x])
        
        # Use a recursive search
        # To speed up, we use a greedy lower bound and an upper bound (remaining vertices)
        
        best_alpha = 0
        
        # Recursive helper
        # avail: set of vertices that can still be added
        # current_size: size of current independent set
        def search(avail, current_size):
            nonlocal best_alpha
            if not avail:
                return
            if current_size + len(avail) <= best_alpha:
                return
            
            # Heuristic: pick vertex with min degree in avail?
            # Or just iterate
            # Optimization: if we can pick all remaining, update best
            # But they must be independent.
            
            # Pick a vertex v from avail
            # To prune, pick v with max degree in the induced subgraph on avail?
            # Actually, standard B&B for MIS:
            # 1. Branch on v: Include v -> remove v and its neighbors from avail.
            # 2. Branch on v: Exclude v -> remove v from avail.
            
            # Let's pick the vertex with the highest degree in the subgraph induced by avail
            # This usually prunes better.
            v = None
            max_deg = -1
            for u in avail:
                deg = len(adj[u] & avail)
                if deg > max_deg:
                    max_deg = deg
                    v = u
            
            if v is None:
                return

            # Option 1: Include v
            # New avail: avail - {v} - N(v)
            new_avail = avail - {v} - (adj[v] & avail)
            search(new_avail, current_size + 1)
            
            # Update best
            if current_size + 1 > best_alpha:
                best_alpha = current_size + 1

            # Option 2: Exclude v
            new_avail = avail - {v}
            search(new_avail, current_size)

        search(set(range(n)), 0)
        return best_alpha

    def try_construct(n, seed):
        random.seed(seed)
        
        # Initialize empty graph
        adj = [set() for _ in range(n)]
        
        # Phase 1: Fill with random edges to maximize density (random maximal triangle-free)
        # Order edges randomly
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            if not has_triangle(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                
        # Phase 2: Try to reduce alpha
        # Loop until alpha < 10 or no progress
        for _ in range(100): # Max iterations
            alpha = max_independent_set_size(adj, n)
            if alpha < 10:
                return adj
            
            # Find an independent set of size >= 10
            # We can modify the search to return the set itself
            # For now, let's just try to add edges between random pairs in a found IS
            
            # To find an IS of size 10, we can use a greedy approach or modify B&B
            # Let's use a simple greedy to find a large IS
            is_set = []
            avail = set(range(n))
            while avail:
                # Pick vertex with min degree in avail
                v = min(avail, key=lambda x: len(adj[x] & avail))
                is_set.append(v)
                avail = avail - {v} - (adj[v] & avail)
                
            if len(is_set) < 10:
                # Greedy found < 10, but exact said >= 10?
                # Greedy is lower bound. If greedy < 10, alpha could still be >= 10.
                # If exact alpha >= 10, we should be able to find one.
                # Let's assume the greedy IS is large enough or we need a better finder.
                # If greedy IS is small, the graph is likely dense, alpha is small.
                # If exact alpha >= 10, greedy usually finds something close.
                # If we can't find an IS of size 10, we might be stuck.
                # Let's break if we can't improve.
                break

            # Try to add edges within is_set
            added = False
            for i in range(len(is_set)):
                for j in range(i+1, len(is_set)):
                    u = is_set[i]
                    v = is_set[j]
                    if u not in adj[v]:
                        if not has_triangle(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            added = True
            if not added:
                break
                
        return None

    # Try n=40
    for seed in range(1000):
        if time.time() > deadline:
            break
        res = try_construct(40, seed)
        if res is not None:
            edges = [[i, j] for i in range(40) for j in res[i] if i < j]
            return {'n': 40, 'edges': edges}

    # Try n=39
    for seed in range(1000):
        if time.time() > deadline:
            break
        res = try_construct(39, seed)
        if res is not None:
            edges = [[i, j] for i in range(39) for j in res[i] if i < j]
            return {'n': 39, 'edges': edges}

    # Fallback: Return a small valid graph
    # C5 blow-up with m=4 gives n=20, alpha=8
    n = 20
    adj = [set() for _ in range(n)]
    # 5 groups of 4
    for g1 in range(5):
        for g2 in range(g1+1, 5):
            if (g2 - g1) % 5 in [1, 4]: # Adjacent in C5
                for u in range(g1*4, (g1+1)*4):
                    for v in range(g2*4, (g2+1)*4):
                        adj[u].add(v)
                        adj[v].add(u)
                        
    edges = [[i, j] for i in range(n) for j in adj[i] if i < j]
    return {'n': n, 'edges': edges}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
