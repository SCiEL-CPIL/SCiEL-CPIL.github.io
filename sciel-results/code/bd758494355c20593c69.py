"""Construct a K3,3-free graph on 20 vertices maximizing edges."""
import time
import random

def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill

    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    N = len(all_edges)  # 190

    def popcount(x):
        return bin(x).count('1')

    def is_valid(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    common = adj[i] & adj[j] & adj[k]
                    valid = common & ~((1 << i) | (1 << j) | (1 << k))
                    if popcount(valid) >= 3:
                        return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def fast_check(adj, u, v):
        """Check if adding edge (u,v) creates a K3,3.
        A new K3,3 must include the new edge, so it must have u and v on opposite sides,
        with 2 other vertices on u's side and 2 other on v's side, all 4 cross-adjacent.
        Side A = {u, a, b}, Side B = {v, c, d}
        Need: a,b in N(v), c,d in N(u), and a~c, a~d, b~c, b~d, with all 6 distinct.
        """
        Nv = adj[v]
        Nu = adj[u]
        excl = (1 << u) | (1 << v)
        cand_a = Nv & ~excl
        cand_c = Nu & ~excl
        
        # Collect vertices in cand_a
        alist = []
        x = cand_a
        while x:
            lb = x & (-x)
            alist.append(lb.bit_length() - 1)
            x -= lb
        
        # For each pair (a,b) in cand_a, check if there are >=2 common neighbors in cand_c
        # that are also neighbors of both a and b, and not a or b themselves
        for ii in range(len(alist)):
            a = alist[ii]
            adj_a = adj[a]
            for jj in range(ii + 1, len(alist)):
                b = alist[jj]
                # c,d must be in cand_c, neighbors of a, neighbors of b, not a, not b
                common_c = cand_c & adj_a & adj[b] & ~((1 << a) | (1 << b))
                # Check if at least 2 bits set
                if common_c & (common_c - 1):  # at least 2 bits
                    return False
        return True

    def count_edges(adj, n):
        total = 0
        for i in range(n):
            total += popcount(adj[i])
        return total // 2

    # Greedy construction with random edge order
    def greedy_build(order, adj0=None):
        adj = adj0[:] if adj0 else [0] * n
        edges = count_edges(adj, n)
        for (u, v) in order:
            if adj[u] & (1 << v):
                continue
            add_edge(adj, u, v)
            if fast_check(adj, u, v):
                edges += 1
            else:
                remove_edge(adj, u, v)
        return adj, edges

    best_adj = [0] * n
    best_edges = 0

    random.seed(42)

    # Multiple greedy trials with different random orderings
    for trial in range(80):
        if time.time() > deadline - 15:
            break
        order = all_edges[:]
        random.shuffle(order)
        adj, edges = greedy_build(order)

        # Local search: try adding any missing edge
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for (u, v) in all_edges:
                if time.time() > deadline - 10:
                    break
                if adj[u] & (1 << v):
                    continue
                add_edge(adj, u, v)
                if fast_check(adj, u, v):
                    edges += 1
                    improved = True
                else:
                    remove_edge(adj, u, v)

        # Improved local search: try targeted edge swaps
        if time.time() < deadline - 10:
            # First, try to find edges whose removal allows adding more edges
            # For each missing edge, find which existing edges block it, then try removing one
            for _ in range(300):
                if time.time() > deadline - 10:
                    break
                # Pick a random missing edge
                missing = [(u, v) for (u, v) in all_edges if not (adj[u] & (1 << v))]
                if not missing:
                    break
                (u2, v2) = random.choice(missing)
                
                # Try removing random existing edges and see if we can add (u2,v2)
                existing = [(u, v) for (u, v) in all_edges if adj[u] & (1 << v)]
                random.shuffle(existing)
                
                found_swap = False
                for (u1, v1) in existing[:20]:  # try up to 20 removals
                    if time.time() > deadline - 10:
                        break
                    # Skip if removing (u1,v1) doesn't help
                    # Try: remove (u1,v1), add (u2,v2)
                    remove_edge(adj, u1, v1)
                    add_edge(adj, u2, v2)
                    if fast_check(adj, u2, v2):
                        # Valid! Check if net gain
                        new_edges = count_edges(adj, n)
                        if new_edges >= edges:
                            edges = new_edges
                            found_swap = True
                            break
                        else:
                            # Revert
                            remove_edge(adj, u2, v2)
                            add_edge(adj, u1, v1)
                    else:
                        # Revert
                        remove_edge(adj, u2, v2)
                        add_edge(adj, u1, v1)
                
                if found_swap:
                    # Continue searching after a successful swap
                    continue
                else:
                    # No swap found for this missing edge, try another
                    pass

        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]

    # Final local search on best with multiple passes
    for polish_pass in range(5):
        if time.time() > deadline - 2:
            break
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for (u, v) in all_edges:
                if time.time() > deadline - 2:
                    break
                if best_adj[u] & (1 << v):
                    continue
                add_edge(best_adj, u, v)
                if fast_check(best_adj, u, v):
                    best_edges += 1
                    improved = True
                else:
                    remove_edge(best_adj, u, v)
        
        # Try edge swaps in polish pass
        if time.time() > deadline - 2:
            break
        for _ in range(200):
            if time.time() > deadline - 2:
                break
            missing = [(u, v) for (u, v) in all_edges if not (best_adj[u] & (1 << v))]
            if not missing:
                break
            (u2, v2) = random.choice(missing)
            existing = [(u, v) for (u, v) in all_edges if best_adj[u] & (1 << v)]
            random.shuffle(existing)
            for (u1, v1) in existing[:15]:
                if time.time() > deadline - 2:
                    break
                remove_edge(best_adj, u1, v1)
                add_edge(best_adj, u2, v2)
                if fast_check(best_adj, u2, v2):
                    new_edges = count_edges(best_adj, n)
                    if new_edges >= best_edges:
                        best_edges = new_edges
                        break
                    else:
                        remove_edge(best_adj, u2, v2)
                        add_edge(best_adj, u1, v1)
                else:
                    remove_edge(best_adj, u2, v2)
                    add_edge(best_adj, u1, v1)

    # Verify
    if not is_valid(best_adj, n):
        # fallback: rebuild conservatively
        best_adj = [0] * n
        best_edges = 0
        for (u, v) in all_edges:
            add_edge(best_adj, u, v)
            if fast_check(best_adj, u, v):
                best_edges += 1
            else:
                remove_edge(best_adj, u, v)

    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                edge_list.append([i, j])

    return edge_list
