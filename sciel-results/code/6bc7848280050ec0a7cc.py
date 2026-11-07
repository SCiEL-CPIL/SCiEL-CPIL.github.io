import time
import random

def construct():
    n = 15
    start = time.time()
    deadline = start + 82  # safety margin before 90s kill

    # Precompute all potential edges
    edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(edge_list)

    # Bitmask adjacency representation
    # adj_mask[v] is a bitmask of neighbors of v

    def check_valid(adj_mask, n):
        """Check if no simple 3-edge path has endpoints with equal degree."""
        deg = [bin(adj_mask[v]).count('1') for v in range(n)]
        
        # Precompute degree group masks: for each degree d, bitmask of vertices with that degree
        max_deg = max(deg) if deg else 0
        deg_group = [0] * (max_deg + 1)
        for v in range(n):
            deg_group[deg[v]] |= (1 << v)
        
        # For each edge (u, v) with u < v:
        #   For each s in N(u) \ {v}:
        #     For each t in N(v) \ {u, s}:
        #       if s != t and deg[s] == deg[t]: return False
        for u in range(n):
            mask_u = adj_mask[u]
            for v in range(u + 1, n):
                if not (mask_u >> v) & 1:
                    continue
                mask_v = adj_mask[v]
                # s in N(u), s != v
                s_mask = mask_u & ~(1 << v)
                # t in N(v), t != u
                t_mask_base = mask_v & ~(1 << u)
                # For each s in s_mask:
                s = 0
                while s_mask:
                    # Get lowest set bit
                    lb = s_mask & (-s_mask)
                    s = lb.bit_length() - 1
                    s_mask ^= lb
                    # t in t_mask_base, t != s
                    t_mask = t_mask_base & ~(1 << s)
                    # Check if any t in t_mask has deg[t] == deg[s]
                    # This is: t_mask & deg_group[deg[s]] != 0
                    if deg[s] <= max_deg and (t_mask & deg_group[deg[s]]):
                        return False
        return True

    def count_edges(adj_mask, n):
        total = 0
        for v in range(n):
            total += bin(adj_mask[v]).count('1')
        return total // 2

    # Start with K_{7,8} bipartite graph
    def make_bipartite(adj_mask, n, split):
        for v in range(n):
            adj_mask[v] = 0
        for i in range(split):
            for j in range(split, n):
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)

    def try_add_edges_greedy(adj_mask, n, deadline, order=None):
        """Greedy: try to add each missing edge if it maintains validity."""
        if order is None:
            order = list(range(m))
            random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline:
                break
            i, j = edge_list[idx]
            if (adj_mask[i] >> j) & 1:
                continue
            # Try adding edge (i, j)
            adj_mask[i] |= (1 << j)
            adj_mask[j] |= (1 << i)
            if check_valid(adj_mask, n):
                pass  # keep it
            else:
                adj_mask[i] ^= (1 << j)
                adj_mask[j] ^= (1 << i)
        return adj_mask

    # Initial: try different bipartite splits and randomized greedy
    best_adj = [0] * n
    best_count = 0

    for split in range(3, 13):
        if time.time() > deadline - 15:
            break
        adj_mask = [0] * n
        make_bipartite(adj_mask, n, split)
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Also try randomized greedy from empty
    for trial in range(20):
        if time.time() > deadline - 10:
            break
        adj_mask = [0] * n
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Local search: edge swaps (remove one edge, add another)
    def local_search(adj_mask, n, deadline, max_rounds=100):
        """Try edge swaps to improve edge count."""
        improved = True
        rounds = 0
        while improved and time.time() < deadline - 5 and rounds < max_rounds:
            rounds += 1
            improved = False
            # Get current edge set
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))

            random.shuffle(edges)
            random.shuffle(non_edges)

            # Try adding a single edge first (cheapest improvement)
            for e2 in non_edges:
                if time.time() > deadline - 5:
                    break
                if improved:
                    break
                i2, j2 = e2
                adj_mask[i2] |= (1 << j2)
                adj_mask[j2] |= (1 << i2)
                if check_valid(adj_mask, n):
                    improved = True
                    break
                else:
                    adj_mask[i2] ^= (1 << j2)
                    adj_mask[j2] ^= (1 << i2)

            if improved:
                continue

            # Try swaps: remove one edge, add another
            for e1 in edges:
                if time.time() > deadline - 5:
                    break
                if improved:
                    break
                i1, j1 = e1
                adj_mask[i1] ^= (1 << j1)
                adj_mask[j1] ^= (1 << i1)
                for e2 in non_edges:
                    if time.time() > deadline - 5:
                        break
                    if improved:
                        break
                    i2, j2 = e2
                    adj_mask[i2] |= (1 << j2)
                    adj_mask[j2] |= (1 << i2)
                    if check_valid(adj_mask, n):
                        improved = True
                        break
                    else:
                        adj_mask[i2] ^= (1 << j2)
                        adj_mask[j2] ^= (1 << i2)
                # Restore removed edge if no swap found
                if not improved:
                    adj_mask[i1] |= (1 << j1)
                    adj_mask[j1] |= (1 << i1)

        return adj_mask

    # Run local search from best
    best_adj = local_search(best_adj, n, deadline)
    best_count = count_edges(best_adj, n)

    # Try 2-edge swaps (remove 2, add 2) for more exploration
    if time.time() < deadline - 15:
        adj_mask = list(best_adj)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            edges = []
            non_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        edges.append((i, j))
                    else:
                        non_edges.append((i, j))

            random.shuffle(edges)
            random.shuffle(non_edges)

            # Try removing 2 edges and adding 2 edges
            max_e = min(len(edges), 40)
            max_ne = min(len(non_edges), 40)
            for idx1 in range(max_e):
                if time.time() > deadline - 5:
                    break
                if improved:
                    break
                for idx2 in range(idx1 + 1, max_e):
                    if time.time() > deadline - 5:
                        break
                    if improved:
                        break
                    e1 = edges[idx1]
                    e2 = edges[idx2]
                    # Try adding 2 non-edges
                    for idx3 in range(max_ne):
                        if time.time() > deadline - 5:
                            break
                        if improved:
                            break
                        for idx4 in range(idx3 + 1, max_ne):
                            if time.time() > deadline - 5:
                                break
                            if improved:
                                break
                            f1 = non_edges[idx3]
                            f2 = non_edges[idx4]
                            # Remove e1, e2; add f1, f2
                            for (a, b) in [e1, e2]:
                                adj_mask[a] ^= (1 << b)
                                adj_mask[b] ^= (1 << a)
                            for (a, b) in [f1, f2]:
                                adj_mask[a] |= (1 << b)
                                adj_mask[b] |= (1 << a)
                            if check_valid(adj_mask, n):
                                improved = True
                                break
                            else:
                                # Revert
                                for (a, b) in [f1, f2]:
                                    adj_mask[a] ^= (1 << b)
                                    adj_mask[b] ^= (1 << a)
                                for (a, b) in [e1, e2]:
                                    adj_mask[a] |= (1 << b)
                                    adj_mask[b] |= (1 << a)

        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Final: try to add any remaining edges greedily
    if time.time() < deadline - 2:
        adj_mask = list(best_adj)
        adj_mask = try_add_edges_greedy(adj_mask, n, deadline)
        cnt = count_edges(adj_mask, n)
        if cnt > best_count:
            best_count = cnt
            best_adj = list(adj_mask)

    # Convert to edge list
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([int(i), int(j)])

    return result
