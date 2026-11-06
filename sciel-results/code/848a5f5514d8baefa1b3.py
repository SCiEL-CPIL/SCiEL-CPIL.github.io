import time
import random

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def count_common(adj, u, v):
        """Count common neighbors of u and v using bitmask."""
        return (adj[u] & adj[v]).bit_count()

    def is_k23_free(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free.
        After adding (u,v), for every pair (a,b) where a,b are both in N(u)∪N(v)∪{u,v},
        we need common neighbors <= 2.
        Actually simpler: after adding (u,v), the only new common-neighbor pairs that change
        are those involving u or v. For any vertex w != u,v:
          - common neighbors of (u,w) = N(u) ∩ N(w) ∩ {v} (v is now neighbor of u)
          - common neighbors of (v,w) = N(v) ∩ N(w) ∩ {u} (u is now neighbor of v)
        For pairs not involving u or v, common neighbors don't change.
        So we just need to check:
          For all w != u, v: |N(u) ∩ N(w) ∩ {v}| + (existing common of u,w) <= 2
          i.e., if w is neighbor of v, then common(u,w) before adding must be <= 1 (since v becomes common)
          Similarly for v,w.
        """
        # After adding (u,v):
        # For any w != u, v:
        #   If w is in N(v) (i.e., adj[v] has bit w), then v becomes a common neighbor of u and w.
        #   So common(u,w) after = common(u,w) before + 1 (if w in N(v))
        #   We need this <= 2, so common(u,w) before must be <= 1 if w in N(v)
        #   Similarly, if w in N(u), then u becomes common neighbor of v and w.
        #   common(v,w) after = common(v,w) before + 1 (if w in N(u)), need <= 2
        # Also need common(u,v) after = |N(u) ∩ N(v)| <= 2 (but u,v are not in each other's N before adding)
        
        # Check common(u, v) after adding:
        cu = (adj[u] & adj[v]).bit_count()
        if cu > 2:
            return False
        
        # For each w != u, v:
        for w in range(n):
            if w == u or w == v:
                continue
            w_in_v = (adj[v] >> w) & 1
            w_in_u = (adj[u] >> w) & 1
            if w_in_v:
                # v becomes common neighbor of u and w
                common_uw = (adj[u] & adj[w]).bit_count()
                if common_uw > 1:  # after adding, it becomes common_uw + 1 > 2
                    return False
            if w_in_u:
                # u becomes common neighbor of v and w
                common_vw = (adj[v] & adj[w]).bit_count()
                if common_vw > 1:
                    return False
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            v = u + 1
            while v < n:
                if (mask >> v) & 1:
                    edges.append([u, v])
                v += 1
        return edges

    def greedy_build():
        """Greedy random construction."""
        adj = [0] * n
        # Random order of all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        for (u, v) in all_edges:
            if is_k23_free(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, max_iters=5000):
        """Local search: try adding edges, then try swaps."""
        adj = list(adj)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            # Try adding any missing edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_k23_free(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # Try edge swaps: remove (a,b), add (c,d)
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    # Remove (a, b)
                    adj[a] &= ~(1 << b)
                    adj[b] &= ~(1 << a)
                    # Try adding a missing edge
                    for c in range(n):
                        for d in range(c + 1, n):
                            if (c, d) == (a, b) or (c, d) == (b, a):
                                continue
                            if not ((adj[c] >> d) & 1):
                                if is_k23_free(adj, c, d):
                                    adj[c] |= (1 << d)
                                    adj[d] |= (1 << c)
                                    improved = True
                                    break
                        if improved:
                            break
                    if not improved:
                        # Restore (a, b)
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                    else:
                        break
                if improved:
                    break
        return adj

    best_adj = None
    best_count = 0

    # Multiple random restarts
    while time.time() < deadline:
        adj = greedy_build()
        cnt = edge_count(adj)
        if cnt > best_count:
            best_adj = list(adj)
            best_count = cnt
        
        # Local search
        if time.time() < deadline - 5.0:
            adj_ls = local_search(adj, max_iters=3000)
            cnt_ls = edge_count(adj_ls)
            if cnt_ls > best_count:
                best_adj = list(adj_ls)
                best_count = cnt_ls

    if best_adj is None:
        # Fallback: simple construction
        best_adj = greedy_build()

    return adj_to_edges(best_adj)
