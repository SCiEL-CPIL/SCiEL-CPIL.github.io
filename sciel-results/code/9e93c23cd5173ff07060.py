"""
K2,3-free graph construction on 20 vertices.
Uses deterministic greedy with minimum-common-neighbors heuristic,
combined with multiple structured initializations and local search.
"""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(all_pairs)

    def init_adj():
        return [0] * n

    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2

    def common_count(adj, u, v):
        return bin(adj[u] & adj[v]).count('1')

    def is_valid_add(adj, u, v):
        if common_count(adj, u, v) > 2:
            return False
        nv = adj[v]
        while nv:
            low = nv & (-nv)
            w = low.bit_length() - 1
            if w != u and common_count(adj, u, w) > 1:
                return False
            nv ^= low
        nu = adj[u]
        while nu:
            low = nu & (-nu)
            w = low.bit_length() - 1
            if w != v and common_count(adj, v, w) > 1:
                return False
            nu ^= low
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_min_common(adj):
        """
        Greedy construction: repeatedly pick the pair (u,v) with minimum
        common neighbors that can be added, until no more edges can be added.
        """
        remaining = all_pairs[:]
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            best_pair = None
            best_cn = n + 1
            for u, v in remaining:
                if (adj[u] >> v) & 1:
                    continue
                cn = common_count(adj, u, v)
                if cn < best_cn:
                    best_cn = cn
                    best_pair = (u, v)
                if cn > 2:
                    # Can't add if common neighbors already > 2
                    pass
            if best_pair is not None and best_cn <= 2:
                u, v = best_pair
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    remaining.remove(best_pair)
                    changed = True
        # Final pass: try all remaining pairs
        for u, v in remaining:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)

    def greedy_order(adj, order):
        """Add edges in the given order if valid."""
        for u, v in order:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)

    def local_search(adj, max_iters=500):
        """
        Local search: try removing one edge and greedily re-filling.
        Also try edge swaps.
        """
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        current_count = best_count
        no_improve = 0

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            # Get current edges
            cur_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (current_adj[i] >> j) & 1:
                        cur_edges.append((i, j))

            if not cur_edges:
                break

            # Random edge removal + refill
            idx = random.randint(0, len(cur_edges) - 1)
            u_rem, v_rem = cur_edges[idx]
            backup = current_adj[:]
            remove_edge(current_adj, u_rem, v_rem)

            # Greedy refill with shuffled missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n)
                       if not ((current_adj[i] >> j) & 1)]
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(current_adj, u, v):
                    add_edge(current_adj, u, v)

            new_count = count_edges(current_adj)
            if new_count > best_count:
                best_adj = current_adj[:]
                best_count = new_count
                current_adj = best_adj[:]
                no_improve = 0
            else:
                current_adj = backup[:]
                no_improve += 1
                if no_improve > 50:
                    break

        return best_adj, best_count

    def local_search_swap(adj, max_iters=300):
        """
        Local search via edge swaps: remove edge e1, try to add edge e2.
        """
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        no_improve = 0

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            cur_edges = []
            missing_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (current_adj[i] >> j) & 1:
                        cur_edges.append((i, j))
                    else:
                        missing_edges.append((i, j))

            if not cur_edges or not missing_edges:
                break

            u1, v1 = cur_edges[random.randint(0, len(cur_edges) - 1)]
            u2, v2 = missing_edges[random.randint(0, len(missing_edges) - 1)]

            backup = current_adj[:]
            remove_edge(current_adj, u1, v1)
            if is_valid_add(current_adj, u2, v2):
                add_edge(current_adj, u2, v2)

            new_count = count_edges(current_adj)
            if new_count > best_count:
                best_adj = current_adj[:]
                best_count = new_count
                current_adj = best_adj[:]
                no_improve = 0
            else:
                current_adj = backup[:]
                no_improve += 1
                if no_improve > 40:
                    break

        return best_adj, best_count

    # Main strategy
    best_adj_overall = None
    best_count_overall = 0

    # Strategy 1: Deterministic greedy with minimum common neighbors
    adj = init_adj()
    greedy_min_common(adj)
    adj, cnt = local_search(adj)
    adj, cnt = local_search_swap(adj)
    if cnt > best_count_overall:
        best_count_overall = cnt
        best_adj_overall = adj[:]

    # Strategy 2: Multiple random orderings with greedy + local search
    max_restarts = 30
    for r in range(max_restarts):
        if time.time() > deadline - 10:
            break

        adj = init_adj()
        order = all_pairs[:]
        random.shuffle(order)
        greedy_order(adj, order)
        adj, cnt = local_search(adj)
        adj, cnt = local_search_swap(adj)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]

    # Strategy 3: Structured initializations
    # Circulant-like
    for steps in [2, 3, 4, 5]:
        if time.time() > deadline - 5:
            break
        adj = init_adj()
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % n
                if diff <= steps or diff >= n - steps:
                    if is_valid_add(adj, i, j):
                        add_edge(adj, i, j)
        adj, cnt = local_search(adj)
        adj, cnt = local_search_swap(adj)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]

    # Strategy 4: Turan-like construction (split into parts)
    for parts in [4, 5]:
        if time.time() > deadline - 5:
            break
        adj = init_adj()
        part_size = n // parts
        for i in range(n):
            for j in range(i + 1, n):
                pi = i // part_size
                pj = j // part_size
                if pi != pj:
                    if is_valid_add(adj, i, j):
                        add_edge(adj, i, j)
        adj, cnt = local_search(adj)
        adj, cnt = local_search_swap(adj)
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]

    # Final: if we still have time, do more local search on best
    if best_adj_overall is not None and time.time() < deadline - 3:
        best_adj_overall, best_count_overall = local_search(best_adj_overall)
        best_adj_overall, best_count_overall = local_search_swap(best_adj_overall)

    # Convert to edge list
    if best_adj_overall is None:
        return []

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj_overall[i] >> j) & 1:
                edges.append([i, j])

    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
