import time
import random
import sys

def construct():
    n = 40
    deadline = time.time() + 115.0  # Safety margin before 120s limit

    # Start with 8 disjoint K5s
    adj = [0] * n  # bitmask of neighbors
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    def would_create_c6(adj, u, v):
        """Check if adding edge (u,v) would create a C6.
        Adding (u,v) creates a C6 iff there's a path of length 4 from u to v.
        Path: u - a - b - c - v (4 edges)
        """
        nu = adj[u]
        nv = adj[v]
        for a in range(n):
            if a == v or not ((nu >> a) & 1):
                continue
            na = adj[a] & ~(1 << u) & ~(1 << v)
            for b in range(n):
                if not ((na >> b) & 1):
                    continue
                # Check if exists c in N(b) ∩ N(v) \ {u, a, b, v}
                common = adj[b] & nv
                common &= ~(1 << u) & ~(1 << a) & ~(1 << b) & ~(1 << v)
                if common:
                    return True
        return False

    def try_add_edge(adj, u, v):
        if u == v or ((adj[u] >> v) & 1):
            return False
        if would_create_c6(adj, u, v):
            return False
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return True

    def count_edges(adj):
        total = 0
        for u in range(n):
            total += bin(adj[u]).count('1')
        return total // 2

    def random_non_edge(adj):
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not ((adj[u] >> v) & 1):
                return u, v

    best_adj = adj[:]
    best_count = count_edges(adj)

    # Phase 1: Greedy add edges - iterate until no progress
    while time.time() < deadline - 10.0:
        progress = False
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    if try_add_edge(adj, u, v):
                        progress = True
        if not progress:
            break

    curr_count = count_edges(adj)
    if curr_count > best_count:
        best_count = curr_count
        best_adj = adj[:]

    # Phase 2: Local search with random additions and 1-for-1 swaps
    while time.time() < deadline:
        # Try random non-edge additions
        for _ in range(150):
            if time.time() >= deadline:
                break
            u, v = random_non_edge(adj)
            try_add_edge(adj, u, v)

        # Try 1-for-1 swap moves: remove edge (a,b), add edge (c,d)
        for _ in range(150):
            if time.time() >= deadline:
                break
            # Pick a random edge to remove
            u = random.randint(0, n - 1)
            neighbors = [v for v in range(n) if v != u and (adj[u] >> v) & 1]
            if not neighbors:
                continue
            a = random.choice(neighbors)

            # Remove edge (u, a)
            adj[u] &= ~(1 << a)
            adj[a] &= ~(1 << u)

            # Try to add a random non-edge
            c, d = random_non_edge(adj)
            added = try_add_edge(adj, c, d)

            if added:
                curr_count = count_edges(adj)
                if curr_count > best_count:
                    best_count = curr_count
                    best_adj = adj[:]
                # Keep the swap (same edge count, different structure)
            else:
                # Restore the removed edge
                adj[u] |= (1 << a)
                adj[a] |= (1 << u)

    # Return the best edges found
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                result.append([u, v])

    return result
