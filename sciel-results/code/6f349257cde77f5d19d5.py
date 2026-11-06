import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 100.0

    def find_violation(adj):
        """Find a K3,3 violation. Returns (u, v, w) triple with >=3 common neighbors, or None."""
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    common = adj[u] & adj[v] & adj[w]
                    if common.bit_count() >= 3:
                        return (u, v, w)
        return None

    def repair(adj):
        """Remove edges to eliminate all K3,3 violations."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            viol = find_violation(adj)
            if viol is None:
                break
            u, v, w = viol
            # Get common neighbors
            common = adj[u] & adj[v] & adj[w]
            # Remove one edge from the K3,3: pick a random common neighbor and remove an edge
            # Strategy: remove edge (u, common_neighbor) or (v, common_neighbor) or (w, common_neighbor)
            # or remove one of the triangle edges (u,v), (u,w), (v,w)
            # Better: remove an edge incident to one of the triple vertices to a common neighbor
            cn_list = []
            c = common
            while c:
                lb = c & (-c)
                cn_list.append(lb.bit_length() - 1)
                c ^= lb
            
            if cn_list:
                # Pick a random common neighbor and remove a random edge from {u,v,w} to that neighbor
                cn = random.choice(cn_list)
                # Try removing edges (u,cn), (v,cn), (w,cn) - pick randomly
                options = [(u, cn), (v, cn), (w, cn)]
                # Filter to only existing edges
                valid_options = [(a, b) for a, b in options if a != b and adj[a] & (1 << b)]
                if valid_options:
                    a, b = random.choice(valid_options)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    changed = True
                else:
                    # Fallback: remove one of the triangle edges
                    tri_edges = [(u, v), (u, w), (v, w)]
                    valid_tri = [(a, b) for a, b in tri_edges if adj[a] & (1 << b)]
                    if valid_tri:
                        a, b = random.choice(valid_tri)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        changed = True
            else:
                # Should not happen if violation found
                break
        return adj

    def is_valid_after_add(adj, u, v):
        """
        Checks if adding edge (u,v) to adj creates a K3,3.
        """
        for a in range(n):
            if a == u:
                continue
            for b in range(a + 1, n):
                if b == u:
                    continue
                common = adj[u] & adj[a] & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        for a in range(n):
            if a == u or a == v:
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                common = adj[v] & adj[a] & adj[b]
                if common.bit_count() >= 3:
                    return False
        
        return True

    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2

    def greedy_fill(adj):
        """Greedily adds edges in random order."""
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for u, v in pairs:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid_after_add(adj, u, v):
                pass
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return count_edges(adj)

    def random_dense_graph(edge_prob):
        """Generate a random graph with given edge probability."""
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < edge_prob:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_adj = [0] * n
    best_count = 0

    # Phase 1: Start from dense random graphs, repair, then greedy fill
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Try different densities
        if trial < 15:
            p = 0.7
        elif trial < 30:
            p = 0.6
        elif trial < 40:
            p = 0.5
        else:
            p = 0.4

        current_adj = random_dense_graph(p)
        
        # Repair violations
        current_adj = repair(current_adj)
        
        # Greedy fill to add more edges
        current_count = greedy_fill(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]

    # Phase 2: Local search from best - remove random edges, then greedy fill
    while time.time() < deadline:
        current_adj = best_adj[:]
        
        # Remove 1 to 3 random edges
        num_removals = random.randint(1, 3)
        edges_list = []
        for u in range(n):
            for v in range(u + 1, n):
                if current_adj[u] & (1 << v):
                    edges_list.append((u, v))
        
        if not edges_list:
            break
            
        random.shuffle(edges_list)
        for u, v in edges_list[:num_removals]:
            current_adj[u] &= ~(1 << v)
            current_adj[v] &= ~(1 << u)
        
        current_count = greedy_fill(current_adj)
        
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]

    # Convert best_adj to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result
