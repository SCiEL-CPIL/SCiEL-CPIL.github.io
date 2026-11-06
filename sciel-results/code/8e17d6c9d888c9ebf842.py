import time
import random
import sys

def construct():
    n = 40
    # Start with 8 disjoint K5s
    adj = [0] * n
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    def has_c6_with_edge(u, v):
        # Check if adding edge (u, v) creates a C6.
        # This is equivalent to checking if there is a path of length 5 between u and v in G \ {(u, v)}.
        # Since G is C6-free, any C6 created must use (u, v).
        # We do a BFS/DFS from u to v with depth 5, avoiding u and v as intermediate nodes.
        # Actually, path is u -> a -> b -> c -> d -> v.
        # a, b, c, d must be distinct and not u, v.
        
        # Optimization: Check if there is a path of length 4 between u and v in G \ {u, v}?
        # No, path length 5 means 4 intermediate nodes.
        
        # Let's just check all paths of length 5.
        # Since n=40, we can use bitsets.
        # Let S0 = {u}
        # S1 = N(u) \ {v}
        # S2 = (union of N(x) for x in S1) \ {u, v}
        # ...
        # S5 = (union of N(x) for x in S4)
        # If v is in S5, then there is a path of length 5.
        
        # However, we must ensure the path is simple.
        # With bitsets, we don't track simplicity easily.
        # But for small n, we can just do a DFS with pruning.
        
        # DFS from u, target v, depth 5.
        # We can use a recursive function or iterative.
        # Given the small size, a simple recursive DFS might be okay if we prune well.
        
        def dfs(current, depth, visited):
            if depth == 5:
                return (adj[current] & (1 << v)) != 0
            nexts = adj[current] & ~visited
            # Exclude v if depth < 4? No, v can only be at depth 5.
            if depth < 4:
                nexts &= ~(1 << v)
            while nexts:
                lsb = nexts & -nexts
                nexts ^= lsb
                nxt = lsb.bit_length() - 1
                if dfs(nxt, depth + 1, visited | lsb):
                    return True
            return False
        
        visited = (1 << u)
        return dfs(u, 0, visited)

    def is_c6_free():
        # Check if the whole graph is C6-free.
        # This is expensive. We should only call this if we suspect we might have created a C6.
        # But in our incremental approach, we only add edges if they don't create a C6.
        # So the graph remains C6-free.
        return True

    best_edges = sum(bin(x).count('1') for x in adj) // 2
    best_adj = adj[:]
    
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    random.seed(42)
    
    # List of all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
    random.shuffle(all_edges)
    
    # Greedy addition
    for u, v in all_edges:
        if time.time() > deadline:
            break
        if adj[u] & (1 << v):
            continue
        if not has_c6_with_edge(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges = sum(bin(x).count('1') for x in adj) // 2
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
    
    # Local search: try to swap edges
    # If we are stuck, we might try to remove an edge and see if we can add more.
    # This is complex. Let's just do more random additions if we have time.
    
    # Re-shuffle and try again
    while time.time() < deadline:
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] & (1 << j))]
        if not all_edges:
            break
        random.shuffle(all_edges)
        added_any = False
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not has_c6_with_edge(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added_any = True
                current_edges = sum(bin(x).count('1') for x in adj) // 2
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
        if not added_any:
            # Try removing a random edge and see if we can add two
            # This is a heuristic improvement
            if random.random() < 0.5:
                # Pick a random edge to remove
                edges_to_remove = [(i, j) for i in range(n) for j in range(i + 1, n) if (adj[i] & (1 << j))]
                if edges_to_remove:
                    ru, rv = random.choice(edges_to_remove)
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
                    # Now try to add edges
                    # This is getting complicated. Let's just restart the greedy phase with a different seed/order.
                    # Or just break and hope the greedy phase was good.
                    pass
    
    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return edges
