"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def has_c4(adj, u, v):
        """Check if adding edge (u,v) creates a C4."""
        # A C4 is created if there exist a,b such that a~u, a~v, b~u, b~v
        # i.e., |N(u) ∩ N(v)| >= 2 before adding the edge
        nu = adj[u]
        nv = adj[v]
        common = 0
        for x in nu:
            if x in nv:
                common += 1
                if common >= 2:
                    return True
        return False
    
    def count_c4s(adj, n):
        """Count number of C4s in the graph."""
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if i in adj[j]:
                    common = 0
                    for x in adj[i]:
                        if x in adj[j]:
                            common += 1
                    if common >= 2:
                        count += common * (common - 1) // 2
        return count
    
    def min_degree(adj, n):
        return min(len(adj[i]) for i in range(n))
    
    def try_add_edge(adj, u, v, n):
        """Try to add edge (u,v) without creating C4."""
        if u == v or u in adj[v]:
            return False
        if has_c4(adj, u, v):
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True
    
    def greedy_fill(adj, n, target_min_deg):
        """Greedily add edges to raise minimum degree."""
        # Sort vertices by degree, repeatedly add edges for lowest-degree vertices
        for iteration in range(200):
            degs = [len(adj[i]) for i in range(n)]
            if min(degs) >= target_min_deg:
                return True
            # Find lowest degree vertex
            min_idx = min(range(n), key=lambda i: degs[i])
            # Try to add edges from min_idx to non-adjacent vertices
            candidates = [v for v in range(n) if v != min_idx and v not in adj[min_idx]]
            random.shuffle(candidates)
            added = False
            for v in candidates:
                if try_add_edge(adj, min_idx, v, n):
                    added = True
                    degs[min_idx] += 1
                    degs[v] += 1
                    if min(degs) >= target_min_deg:
                        return True
            if not added:
                # Try adding edges between other low-degree pairs
                low_verts = sorted(range(n), key=lambda i: degs[i])[:10]
                for a in range(len(low_verts)):
                    for b in range(a+1, len(low_verts)):
                        u, v = low_verts[a], low_verts[b]
                        if u not in adj[v] and try_add_edge(adj, u, v, n):
                            degs[u] += 1
                            degs[v] += 1
                            if min(degs) >= target_min_deg:
                                return True
        return min(degs) >= target_min_deg
    
    def local_search(adj, n, target_min_deg, max_iters=5000):
        """Local search: remove edges that create C4s, then re-add."""
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            degs = [len(adj[i]) for i in range(n)]
            if min(degs) >= target_min_deg:
                # Check no C4s
                c4_count = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if i in adj[j]:
                            common = len(adj[i] & adj[j])
                            if common >= 2:
                                c4_count += 1
                if c4_count == 0:
                    return True
                # Remove one problematic edge
                removed = False
                for i in range(n):
                    for j in range(i+1, n):
                        if i in adj[j]:
                            common = adj[i] & adj[j]
                            if len(common) >= 2:
                                adj[i].discard(j)
                                adj[j].discard(i)
                                removed = True
                                break
                    if removed:
                        break
                if not removed:
                    break
            else:
                # Need to add edges
                min_idx = min(range(n), key=lambda i: degs[i])
                candidates = [v for v in range(n) if v != min_idx and v not in adj[min_idx]]
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    if try_add_edge(adj, min_idx, v, n):
                        added = True
                        break
                if not added:
                    # Try random pair
                    for _ in range(20):
                        u = random.randint(0, n-1)
                        v = random.randint(0, n-1)
                        if u != v and u not in adj[v]:
                            if try_add_edge(adj, u, v, n):
                                added = True
                                break
        return min(degs) >= target_min_deg
    
    def build_graph(n, target_min_deg, seed=None):
        """Build a C4-free graph on n vertices with min degree >= target_min_deg."""
        if seed is not None:
            random.seed(seed)
        
        # Start with random sparse graph
        adj = [set() for _ in range(n)]
        
        # Initial random edges (sparse to avoid C4s)
        num_initial = min(n * 2, n * n // 4)
        attempts = 0
        while len([1 for i in range(n) for j in range(i+1,n) if j in adj[i]]) < num_initial and attempts < 10000:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and u not in adj[v]:
                try_add_edge(adj, u, v, n)
            attempts += 1
        
        # Greedy fill
        greedy_fill(adj, n, target_min_deg)
        
        # Local search to fix any issues
        local_search(adj, n, target_min_deg, max_iters=3000)
        
        # Verify
        degs = [len(adj[i]) for i in range(n)]
        if min(degs) < target_min_deg:
            return None
        
        # Check C4-free
        for i in range(n):
            for j in range(i+1, n):
                if i in adj[j]:
                    if len(adj[i] & adj[j]) >= 2:
                        return None
        
        # Convert to edge list
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        
        return {'n': n, 'edges': edges}
    
    def verify(result):
        """Verify the result is valid."""
        if result is None:
            return False
        n = result['n']
        edges = result['edges']
        if n < 1 or n > 47:
            return False
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v or u >= n or v >= n:
                return False
            adj[u].add(v)
            adj[v].add(u)
        # Check min degree
        for i in range(n):
            if len(adj[i]) < n - 39:
                return False
        # Check C4-free
        for i in range(n):
            for j in range(i+1, n):
                if i in adj[j]:
                    if len(adj[i] & adj[j]) >= 2:
                        return False
        return True
    
    best_result = None
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        if time.time() > deadline - 15:
            break
        target_min_deg = n - 39
        if target_min_deg < 0:
            target_min_deg = 0
        
        # Multiple attempts with different seeds
        found = False
        for attempt in range(50):
            if time.time() > deadline - 15:
                break
            result = build_graph(n, target_min_deg, seed=attempt * 12345 + n)
            if result is not None and verify(result):
                best_result = result
                found = True
                break
        
        if found:
            break
    
    # Fallback: return valid smaller graph
    if best_result is None:
        # Try simpler constructions
        for n in range(47, 38, -1):
            if time.time() > deadline - 10:
                break
            target_min_deg = max(0, n - 39)
            result = build_graph(n, target_min_deg, seed=999)
            if result is not None and verify(result):
                best_result = result
                break
    
    if best_result is None:
        # Absolute fallback: empty graph on 39 vertices
        best_result = {'n': 39, 'edges': []}
    
    return best_result
# EVOLVE-BLOCK-END

