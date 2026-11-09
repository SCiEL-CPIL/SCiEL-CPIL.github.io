"""Improved construction using structured C4-free graphs and local search."""
import random
import time
import numpy as np

def construct():
    start = time.time()
    deadline = 145  # safety margin before 150s kill
    
    def is_c4_free_fast(n, adj):
        """Check C4-freeness using bitmasks for speed."""
        adj_masks = []
        for i in range(n):
            mask = 0
            for j in adj[i]:
                mask |= (1 << j)
            adj_masks.append(mask)
        
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = adj_masks[u] & adj_masks[v]
                    if common & (common - 1):  # More than 1 bit set
                        return False
        return True
    
    def min_degree(adj):
        return min(len(adj[i]) for i in range(len(adj)))
    
    def valid(n, adj):
        return min_degree(adj) >= n - 39 and is_c4_free_fast(n, adj)
    
    def to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) preserves C4-freeness."""
        # Need |adj[u] ∩ adj[v]| <= 1 (since u and v are not adjacent yet)
        common = adj[u] & adj[v]
        return len(common) <= 1
    
    def find_c4(adj, n):
        """Find a C4 in the graph. Returns (u, v, w, x) or None."""
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = adj[u] & adj[v]
                    if len(common) > 1:
                        common_list = list(common)
                        w = common_list[0]
                        x = common_list[1]
                        return (u, v, w, x)
        return None
    
    def try_complete_adj(adj, n, min_deg):
        """Try to add edges to reach minimum degree while preserving C4-freeness."""
        n_local = len(adj)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Process vertices with lowest degree first
            vertices = sorted(range(n_local), key=lambda i: len(adj[i]))
            for u in vertices:
                if len(adj[u]) >= min_deg:
                    continue
                candidates = [v for v in range(n_local) if v != u and v not in adj[u]]
                random.shuffle(candidates)
                for v in candidates:
                    if time.time() > deadline:
                        break
                    if can_add_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        improved = True
                        if len(adj[u]) >= min_deg:
                            break
                if time.time() > deadline:
                    break
        return valid(n_local, adj)
    
    def try_remove_edges(adj, n):
        """Try removing edges to fix C4 violations."""
        n_local = len(adj)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            c4 = find_c4(adj, n_local)
            if c4:
                u, v, w, x = c4
                # Remove an edge from the C4 - try removing edge u-w or v-w
                if random.random() < 0.5:
                    adj[u].discard(w)
                    adj[w].discard(u)
                else:
                    adj[v].discard(w)
                    adj[w].discard(v)
                changed = True
        return is_c4_free_fast(n_local, adj)
    
    def random_c4_free_graph(n, min_deg, seed=None):
        """Generate a random C4-free graph with min degree >= min_deg."""
        if seed is not None:
            random.seed(seed)
        
        adj = [set() for _ in range(n)]
        
        # Start with a sparse random graph
        p = min_deg / (n - 1) if n > 1 else 0
        p = max(0.3, min(0.95, p))
        
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    if can_add_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
        
        # Try to complete
        success = try_complete_adj(adj, n, min_deg)
        if not success:
            # Try to fix C4 violations by removing edges
            if try_remove_edges(adj, n):
                if min_degree(adj) >= min_deg:
                    success = True
        
        return adj if success else None
    
    def complete_graph_remove_c4s(n, min_deg):
        """Start from complete graph and remove edges to break C4s."""
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                adj[u].add(v)
                adj[v].add(u)
        
        removed = [0] * n
        max_removals = 39
        
        # Iteratively find and break C4s
        for iteration in range(10000):
            if time.time() > deadline:
                break
            
            c4 = find_c4(adj, n)
            if not c4:
                break
            
            u, v, w, x = c4
            # Choose which edge to remove based on removal counts
            options = []
            if removed[u] < max_removals:
                options.append((u, v, removed[u]))
                options.append((u, w, removed[u]))
            if removed[v] < max_removals:
                options.append((v, w, removed[v]))
                options.append((v, x, removed[v]))
            if removed[w] < max_removals:
                options.append((w, x, removed[w]))
            
            if options:
                # Pick option with lowest removal count
                options.sort(key=lambda x: x[2])
                choice = options[0]
                a, b = choice[0], choice[1]
                adj[a].discard(b)
                adj[b].discard(a)
                removed[a] += 1
                removed[b] += 1
        
        if valid(n, adj):
            return adj
        return None
    
    best = {'n': 39, 'edges': []}
    
    # Try descending n from 47 to 40
    for n in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        min_deg = n - 39
        found = False
        
        # Strategy 1: Start from complete, remove C4s (often most effective)
        if time.time() < deadline:
            adj = complete_graph_remove_c4s(n, min_deg)
            if adj is not None:
                best = {'n': n, 'edges': to_edges(adj, n)}
                found = True
        
        # Strategy 2: Random + repair with multiple seeds
        if not found and time.time() < deadline:
            max_attempts = 500
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                seed = attempt * 7919 + 12345
                adj = random_c4_free_graph(n, min_deg, seed)
                if adj is not None:
                    best = {'n': n, 'edges': to_edges(adj, n)}
                    found = True
                    break
        
        if found:
            break
    
    return best

# EVOLVE-BLOCK-END
