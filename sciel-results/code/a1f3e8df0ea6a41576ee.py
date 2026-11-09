"""
Strategy: Use the polarity graph of PG(2,5) as a C4-free base (90 edges), then
perform local search over "almost-polarity" graphs by swapping edges to minimize C4s.

Key structural idea: The polarity graph of PG(2,5) with a symmetric bilinear form
gives a C4-free graph. We use this as our starting point and then perform
efficient local search using incremental C4 counting.

For speed, we maintain:
- adj: adjacency matrix (int8)
- common: common neighbor counts matrix (int32)
- c4_count: total number of C4s

When adding edge (a,b):
- For each x in N(a) \ {b}: common(x,b) += 1, common(b,x) += 1
- For each y in N(b) \ {a}: common(a,y) += 1, common(y,a) += 1
- Delta C4 = (1/2) * [sum_{x in N(a)\{b}} common(x,b) + sum_{y in N(b)\{a}} common(a,y)]

When removing edge (a,b):
- For each x in N(a) \ {b}: common(x,b) -= 1, common(b,x) -= 1
- For each y in N(b) \ {a}: common(a,y) -= 1, common(y,a) -= 1
- Delta C4 = -(1/2) * [sum_{x in N(a)\{b}} common(x,b) + sum_{y in N(b)\{a}} common(a,y)]
  (computed with the common values BEFORE removal)
"""

import time
import random
import numpy as np

def construct():
    n = 31
    m = 91
    deadline = time.time() + 130
    
    rng = random.Random(42)
    
    def count_c4(adj):
        """Count C4s in graph given by adjacency matrix."""
        A = adj.astype(np.int32)
        C = A @ A
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                c = C[u, v]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2
    
    def build_polarity_graph():
        """Build the polarity graph of PG(2,5) using a symmetric bilinear form."""
        points = []
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        inv = pow(x, -1, 5)
                        pts = (1, (y * inv) % 5, (z * inv) % 5)
                    elif y != 0:
                        inv = pow(y, -1, 5)
                        pts = (0, 1, (z * inv) % 5)
                    else:
                        pts = (0, 0, 1)
                    if pts not in points:
                        points.append(pts)
        
        assert len(points) == 31, f"Expected 31 points, got {len(points)}"
        
        def dot_product(p, q):
            return (p[0] * q[0] + p[1] * q[1] + p[2] * q[2]) % 5
        
        adj = np.zeros((n, n), dtype=np.int8)
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if dot_product(points[i], points[j]) == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    edge_count += 1
        
        return adj, edge_count
    
    def compute_common(adj):
        """Compute common neighbor counts from adjacency matrix."""
        A = adj.astype(np.int32)
        C = A @ A
        # Zero out diagonal (degree) entries
        np.fill_diagonal(C, 0)
        return C
    
    def add_edge(adj, common, c4_count, a, b):
        """Add edge (a,b) and update common counts and C4 count. Returns new c4_count."""
        # Get neighbors of a and b (excluding each other)
        na = np.where(adj[a] == 1)[0]
        na = na[na != b]
        nb = np.where(adj[b] == 1)[0]
        nb = nb[nb != a]
        
        # Compute delta C4 before modifying
        delta = 0
        for x in na:
            delta += common[x, b]
        for y in nb:
            delta += common[a, y]
        delta //= 2
        
        # Update adjacency
        adj[a, b] = 1
        adj[b, a] = 1
        
        # Update common counts
        for x in na:
            common[x, b] += 1
            common[b, x] += 1
        for y in nb:
            common[a, y] += 1
            common[y, a] += 1
        
        return c4_count + delta
    
    def remove_edge(adj, common, c4_count, a, b):
        """Remove edge (a,b) and update common counts and C4 count. Returns new c4_count."""
        # Get neighbors of a and b (excluding each other)
        na = np.where(adj[a] == 1)[0]
        na = na[na != b]
        nb = np.where(adj[b] == 1)[0]
        nb = nb[nb != a]
        
        # Compute delta C4 before modifying
        delta = 0
        for x in na:
            delta += common[x, b]
        for y in nb:
            delta += common[a, y]
        delta //= 2
        
        # Update adjacency
        adj[a, b] = 0
        adj[b, a] = 0
        
        # Update common counts
        for x in na:
            common[x, b] -= 1
            common[b, x] -= 1
        for y in nb:
            common[a, y] -= 1
            common[y, a] -= 1
        
        return c4_count - delta
    
    def local_search(adj, common, c4_count, time_limit):
        """Perform local search with edge swaps to minimize C4 count."""
        best_adj = adj.copy()
        best_common = common.copy()
        best_c4 = c4_count
        
        # Build list of current edges and non-edges
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        T = 50.0
        T_min = 1.0
        cooling = 0.9999
        no_improve = 0
        
        while T > T_min and time.time() < time_limit:
            # Pick random edge to remove
            u, v = rng.choice(edges)
            
            # Pick random non-edge to add
            c, d = rng.choice(non_edges)
            
            # Compute delta for removing (u,v)
            delta_remove = 0
            nu = np.where(adj[u] == 1)[0]
            nu = nu[nu != v]
            nv = np.where(adj[v] == 1)[0]
            nv = nv[nv != u]
            for x in nu:
                delta_remove += common[x, v]
            for y in nv:
                delta_remove += common[u, y]
            delta_remove //= 2
            
            # Compute delta for adding (c,d)
            delta_add = 0
            nc = np.where(adj[c] == 1)[0]
            nc = nc[nc != d]
            nd = np.where(adj[d] == 1)[0]
            nd = nd[nd != c]
            for x in nc:
                delta_add += common[x, d]
            for y in nd:
                delta_add += common[c, y]
            delta_add //= 2
            
            delta = -delta_remove + delta_add
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Perform the swap
                c4_count = remove_edge(adj, common, c4_count, u, v)
                c4_count = add_edge(adj, common, c4_count, c, d)
                
                # Update edges and non_edges lists
                edges.remove((u, v))
                edges.append((c, d))
                non_edges.remove((c, d))
                non_edges.append((u, v))
                
                no_improve = 0
                if c4_count < best_c4:
                    best_c4 = c4_count
                    best_adj = adj.copy()
                    best_common = common.copy()
            else:
                no_improve += 1
                if no_improve > 500:
                    break
            
            T *= cooling
        
        return best_adj, best_common, best_c4
    
    # Build the polarity graph
    base_adj, base_edges = build_polarity_graph()
    
    # Check if we have 90 edges (C4-free)
    if base_edges != 90:
        # Adjust to get 90 edges
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        existing = set()
        for i in range(n):
            for j in range(i + 1, n):
                if base_adj[i, j] == 1:
                    existing.add((i, j))
        
        if base_edges < 90:
            missing = [p for p in all_pairs if p not in existing]
            to_add = rng.sample(missing, 90 - base_edges)
            for u, v in to_add:
                base_adj[u, v] = 1
                base_adj[v, u] = 1
        else:
            existing_list = list(existing)
            to_remove = rng.sample(existing_list, base_edges - 90)
            for u, v in to_remove:
                base_adj[u, v] = 0
                base_adj[v, u] = 0
                existing.remove((u, v))
    
    # Verify base graph is C4-free
    base_c4 = count_c4(base_adj)
    
    # Add the 91st edge that creates the fewest C4s
    best_adj = None
    best_c4 = float('inf')
    
    # Try adding each possible non-edge and pick the one with minimum C4 increase
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if base_adj[i, j] == 0:
                non_edges.append((i, j))
    
    # For each candidate edge, compute the C4 count if added
    # Use incremental computation for speed
    common_base = compute_common(base_adj)
    
    for edge in non_edges:
        a, b = edge
        adj_temp = base_adj.copy()
        common_temp = common_base.copy()
        
        c4_temp = add_edge(adj_temp, common_temp, base_c4, a, b)
        
        if c4_temp < best_c4:
            best_c4 = c4_temp
            best_adj = adj_temp.copy()
    
    # Now perform local search to minimize C4 count
    common_best = compute_common(best_adj)
    best_adj, common_best, best_c4 = local_search(best_adj, common_best, best_c4, deadline)
    
    # Extract edges from best_adj
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                final_edges.append([i, j])
    
    # Ensure we have exactly 91 edges
    assert len(final_edges) == m, f"Expected {m} edges, got {len(final_edges)}"
    
    return final_edges

if __name__ == "__main__":
    edges = construct()
    adj = np.zeros((31, 31), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A = adj.astype(np.int32)
    C = A @ A
    total = 0
    for u in range(31):
        for v in range(u + 1, 31):
            c = C[u, v]
            if c >= 2:
                total += c * (c - 1) // 2
    print(f"C4: {total // 2}")
