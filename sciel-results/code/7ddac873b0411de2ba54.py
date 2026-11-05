"""
Longest simple path in G(100, 1.8/100) pinned graph.
Strategy: Multi-phase search with bitset operations, forced moves, and
connectivity-based upper bounds.
Phase 1: Greedy heuristics for good initial lower bound.
Phase 2: Branch and Bound with forced moves + connectivity upper bound.
"""

import time
from collections import deque

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    adj = _build_adjacency()

    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited_comp[nb]:
                        visited_comp[nb] = True
                        stack.append(nb)
            components.append(comp)

    largest_comp = max(components, key=len)
    comp_size = len(largest_comp)

    # Map vertices in largest component to indices 0..comp_size-1
    idx_map = {v: i for i, v in enumerate(largest_comp)}
    full_mask = (1 << comp_size) - 1

    adj_bits = [0] * comp_size
    for u in largest_comp:
        ui = idx_map[u]
        for v in adj[u]:
            if v in idx_map:
                adj_bits[ui] |= (1 << idx_map[v])

    adj_list = [[] for _ in range(comp_size)]
    for i in range(comp_size):
        bits = adj_bits[i]
        j = 0
        while bits:
            if bits & 1:
                adj_list[i].append(j)
            bits >>= 1
            j += 1

    best_path = []
    best_len = 0

    def time_up():
        return time.time() > deadline

    # --- Phase 1: Greedy heuristics ---
    def greedy_from(start_idx):
        path = [start_idx]
        visited = 1 << start_idx
        while True:
            cur = path[-1]
            nbrs = [x for x in adj_list[cur] if not (visited >> x) & 1]
            if not nbrs:
                break
            nbrs.sort(key=lambda x: bin(adj_bits[x] & ~visited).count('1'))
            nxt = nbrs[0]
            path.append(nxt)
            visited |= (1 << nxt)
        return path

    def greedy_both_ends(start_idx):
        path = [start_idx]
        visited = 1 << start_idx
        extend_front = False
        while True:
            if extend_front:
                cur = path[0]
                nbrs = [x for x in adj_list[cur] if not (visited >> x) & 1]
                if not nbrs:
                    extend_front = not extend_front
                    continue
                nbrs.sort(key=lambda x: bin(adj_bits[x] & ~visited).count('1'))
                nxt = nbrs[0]
                path.insert(0, nxt)
                visited |= (1 << nxt)
            else:
                cur = path[-1]
                nbrs = [x for x in adj_list[cur] if not (visited >> x) & 1]
                if not nbrs:
                    break
                nbrs.sort(key=lambda x: bin(adj_bits[x] & ~visited).count('1'))
                nxt = nbrs[0]
                path.append(nxt)
                visited |= (1 << nxt)
            extend_front = not extend_front
        return path

    def update_best(path):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)

    for start in range(comp_size):
        if time_up():
            break
        update_best(greedy_from(start))
        update_best(greedy_both_ends(start))

    if best_len == comp_size:
        return [largest_comp[i] for i in best_path]

    # --- Phase 2: Branch and Bound ---

    def find_reachable_components(unvisited_mask, e1, e2):
        """Find connected components in unvisited subgraph, return sizes of
        components adjacent to e1 or e2."""
        remaining = unvisited_mask
        reachable_sizes = []

        while remaining:
            start_bit = remaining & (-remaining)
            start_idx = start_bit.bit_length() - 1

            comp_mask = 1 << start_idx
            stack = [start_idx]

            while stack:
                node = stack.pop()
                nbrs = adj_bits[node] & unvisited_mask & ~comp_mask
                while nbrs:
                    bit = nbrs & (-nbrs)
                    idx = bit.bit_length() - 1
                    comp_mask |= bit
                    stack.append(idx)
                    nbrs &= nbrs - 1

            remaining &= ~comp_mask
            size = comp_mask.bit_count()

            # Check if component is adjacent to e1 or e2
            if (adj_bits[e1] & comp_mask) or (adj_bits[e2] & comp_mask):
                reachable_sizes.append(size)

        return reachable_sizes

    def upper_bound(path_indices, visited_mask):
        unvisited_mask = full_mask & ~visited_mask
        if unvisited_mask == 0:
            return len(path_indices)
        e1 = path_indices[0]
        e2 = path_indices[-1]
        reachable = find_reachable_components(unvisited_mask, e1, e2)
        if not reachable:
            return len(path_indices)
        reachable.sort(reverse=True)
        if len(reachable) >= 2:
            return len(path_indices) + reachable[0] + reachable[1]
        return len(path_indices) + reachable[0]

    def find_forced_moves(path_indices, visited_mask):
        """Find unvisited vertices that must be endpoints of the final path.
        A vertex v is forced if its degree in (unvisited + endpoints) is 1.
        Returns list of forced vertices (indices into comp_size)."""
        e1 = path_indices[0]
        e2 = path_indices[-1]
        unvisited = full_mask & ~visited_mask
        forced = []
        if unvisited == 0:
            return forced

        # For each unvisited vertex, count neighbors in unvisited + {e1, e2}
        # that are themselves "active" (i.e., can be in the final path)
        # Simplified: degree in subgraph induced by (unvisited ∪ {e1,e2})
        active = unvisited | (1 << e1) | (1 << e2)

        # Build degree counts for unvisited vertices
        # For speed, iterate over unvisited vertices
        bits = unvisited
        while bits:
            bit = bits & (-bits)
            v = bit.bit_length() - 1
            # Count neighbors of v that are in active
            nbrs_active = adj_bits[v] & active
            deg = nbrs_active.bit_count()
            if deg == 1:
                forced.append(v)
            bits &= bits - 1

        return forced

    # B&B DFS
    start_order = sorted(range(comp_size), key=lambda x: len(adj_list[x]), reverse=True)

    for start in start_order:
        if time_up():
            break

        path = [start]
        visited = 1 << start

        dfs_stack = [(path, visited)]

        while dfs_stack:
            if time_up():
                return [largest_comp[i] for i in best_path]

            curr_path, curr_visited = dfs_stack.pop()
            curr_len = len(curr_path)

            if curr_len > best_len:
                best_len = curr_len
                best_path = list(curr_path)
                if best_len == comp_size:
                    return [largest_comp[i] for i in best_path]

            # Upper bound check
            ub = upper_bound(curr_path, curr_visited)
            if ub <= best_len:
                continue

            # Forced move check: if >2 forced endpoints exist, prune
            forced = find_forced_moves(curr_path, curr_visited)
            if len(forced) > 2:
                continue

            # If exactly 2 forced moves and neither is an endpoint, the path
            # cannot be extended to include both (they'd need to be endpoints,
            # but we already have 2 endpoints). Prune.
            if len(forced) == 2:
                e1 = curr_path[0]
                e2 = curr_path[-1]
                if e1 not in forced and e2 not in forced:
                    # Both forced vertices would need to be endpoints, but
                    # the current endpoints are already fixed. Prune.
                    continue

            # Try extending the tail
            tail = curr_path[-1]
            nbrs = adj_list[tail]

            def nbr_key(x):
                if (curr_visited >> x) & 1:
                    return 999
                return bin(adj_bits[x] & ~curr_visited).count('1')

            nbrs_sorted = sorted(nbrs, key=nbr_key)

            for nxt in nbrs_sorted:
                if (curr_visited >> nxt) & 1:
                    continue

                new_visited = curr_visited | (1 << nxt)
                new_path = curr_path + [nxt]

                # Quick weak bound
                if len(new_path) + (comp_size - new_visited.bit_count()) <= best_len:
                    continue

                dfs_stack.append((new_path, new_visited))

    return [largest_comp[i] for i in best_path]
