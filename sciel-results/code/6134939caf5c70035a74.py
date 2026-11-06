import random
import time
import itertools

def construct():
    n = 20
    best_edges = 0
    best_graph = None
    start_time = time.time()
    timeout = 130  # Safety margin before 150s kill

    def has_k33(adj):
        # Check if graph has K3,3 using bitsets
        # adj is list of int bitmasks
        for u1 in range(n):
            for u2 in range(u1+1, n):
                for u3 in range(u2+1, n):
                    common = adj[u1] & adj[u2] & adj[u3]
                    if bin(common).count('1') >= 3:
                        return True
        return False

    def count_violations(adj):
        # Returns list of (u1, u2, u3, common_mask)
        violations = []
        for u1 in range(n):
            for u2 in range(u1+1, n):
                for u3 in range(u2+1, n):
                    common = adj[u1] & adj[u2] & adj[u3]
                    if bin(common).count('1') >= 3:
                        violations.append((u1, u2, u3, common))
        return violations

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def is_add_safe(adj, u, v):
        # Check if adding (u,v) creates a K3,3
        # For each pair (b1, b2) in N(u), check if |N(v) & N(b1) & N(b2) \ {u,v,b1,b2}| >= 2
        nu = adj[u]
        nv = adj[v]
        # Iterate over pairs in N(u)
        # Extract neighbors of u as list
        neighbors_u = [i for i in range(n) if (nu >> i) & 1]
        for i in range(len(neighbors_u)):
            b1 = neighbors_u[i]
            for j in range(i+1, len(neighbors_u)):
                b2 = neighbors_u[j]
                common = nv & adj[b1] & adj[b2]
                # Remove u, v, b1, b2 from common
                common &= ~(1 << u)
                common &= ~(1 << v)
                common &= ~(1 << b1)
                common &= ~(1 << b2)
                if bin(common).count('1') >= 2:
                    return False
        return True

    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def build_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def local_search(adj, max_time=None):
        # Phase 1: Remove edges to fix K3,3 violations
        while True:
            if max_time and time.time() - start_time > max_time:
                break
            violations = count_violations(adj)
            if not violations:
                break
            
            # Count edge frequencies in violations
            edge_counts = {}
            for u1, u2, u3, common in violations:
                # Extract common neighbors
                common_neighbors = [c for c in range(n) if (common >> c) & 1]
                for c in common_neighbors:
                    for u in (u1, u2, u3):
                        e = (min(u, c), max(u, c))
                        edge_counts[e] = edge_counts.get(e, 0) + 1
            
            if not edge_counts:
                break
                
            # Pick edge with max count
            max_count = -1
            best_edge = None
            for e, cnt in edge_counts.items():
                if cnt > max_count:
                    max_count = cnt
                    best_edge = e
            
            if best_edge:
                remove_edge(adj, best_edge[0], best_edge[1])
        
        # Phase 2: Add edges greedily
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if not ((adj[i] >> j) & 1):
                        if is_add_safe(adj, i, j):
                            add_edge(adj, i, j)
                            improved = True
        
        # Phase 3: 1-edge swap moves to escape local optima
        swap_improved = True
        while swap_improved:
            if max_time and time.time() - start_time > max_time:
                break
            swap_improved = False
            current_count = edge_count(adj)
            
            # Try removing one edge and adding another
            # First collect all edges
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        edges.append((i, j))
            
            # Try each removal
            for rem_edge in edges:
                if max_time and time.time() - start_time > max_time:
                    break
                u1, v1 = rem_edge
                # Save state
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                # Try to add edges to compensate
                added = False
                for i in range(n):
                    for j in range(i+1, n):
                        if not ((adj[i] >> j) & 1):
                            if is_add_safe(adj, i, j):
                                add_edge(adj, i, j)
                                added = True
                                break
                    if added:
                        break
                
                # Check if we improved
                new_count = edge_count(adj)
                if new_count > current_count:
                    # Keep the change
                    swap_improved = True
                else:
                    # Revert: remove added edge and restore removed edge
                    if added:
                        # Find the added edge and remove it
                        # We need to track which edge was added
                        # Simplify: just revert everything
                        # Actually, let's be more careful
                        pass
                
                # For simplicity, let's do a cleaner swap
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)
                
                # Try a proper swap: remove (u1,v1), add (i,j)
                for i in range(n):
                    for j in range(i+1, n):
                        if (i, j) == (u1, v1) or (j, i) == (u1, v1):
                            continue
                        if not ((adj[i] >> j) & 1):
                            # Check if swap is valid: remove (u1,v1), add (i,j)
                            # Temporarily remove
                            adj[u1] &= ~(1 << v1)
                            adj[v1] &= ~(1 << u1)
                            
                            if is_add_safe(adj, i, j):
                                add_edge(adj, i, j)
                                # Check no K3,3
                                if not has_k33(adj):
                                    swap_improved = True
                                else:
                                    # Revert
                                    remove_edge(adj, i, j)
                                    adj[u1] |= (1 << v1)
                                    adj[v1] |= (1 << u1)
                            else:
                                adj[u1] |= (1 << v1)
                                adj[v1] |= (1 << u1)
                        
                        if swap_improved:
                            break
                    if swap_improved:
                        break
                
                if swap_improved:
                    break
        
        return adj, edge_count(adj)

    def greedy_fill(adj):
        # Add all possible edges
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if not ((adj[i] >> j) & 1):
                        if is_add_safe(adj, i, j):
                            add_edge(adj, i, j)
                            improved = True
        return adj

    # Deterministic structural starting point
    # Use a 5x4 grid-like construction
    # Vertices 0-19
    # Try a few structured initial graphs
    
    def make_structured_graph():
        adj = [0] * n
        # Create a bipartite-like structure
        # Left: 0-9, Right: 10-19
        # Connect with a pattern that avoids K3,3
        # Use a 5x4 incidence matrix pattern
        for i in range(10):
            for j in range(10, 20):
                # Pattern: connect if (i//2 + (j-10)) % 5 == 0 or similar
                # Let's use a simpler pattern
                if (i + (j - 10)) % 4 == 0:
                    add_edge(adj, i, j)
        # Add some edges within left and right
        for i in range(10):
            for j in range(i+1, 10):
                if (i + j) % 3 == 0:
                    add_edge(adj, i, j)
        for i in range(10, 20):
            for j in range(i+1, 20):
                if (i + j) % 3 == 0:
                    add_edge(adj, i, j)
        return adj

    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Random with different densities
    for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for _ in range(5):
            if time.time() - start_time > timeout:
                break
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        add_edge(adj, i, j)
            strategies.append(adj)
    
    # Strategy 2: Structured
    strategies.append(make_structured_graph())
    
    # Run local search on each strategy
    for adj in strategies:
        if time.time() - start_time > timeout:
            break
        adj, count = local_search(adj, max_time=timeout)
        if count > best_edges:
            best_edges = count
            best_graph = build_edges_from_adj(adj)
    
    if best_graph is None:
        # Fallback
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    return best_graph
