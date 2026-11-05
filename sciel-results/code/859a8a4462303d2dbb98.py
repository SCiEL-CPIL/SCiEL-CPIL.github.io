import time
import sys

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
    adj_bits = [0] * comp_size
    for u in largest_comp:
        ui = idx_map[u]
        for v in adj[u]:
            if v in idx_map:
                vi = idx_map[v]
                adj_bits[ui] |= (1 << vi)

    # Precompute adjacency as lists
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

    # Phase 1: Greedy heuristics
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

    # Try multiple greedy strategies
    for start in range(comp_size):
        if time_up():
            break
        p1 = greedy_from(start)
        if len(p1) > best_len:
            best_len = len(p1)
            best_path = list(p1)
        p2 = greedy_both_ends(start)
        if len(p2) > best_len:
            best_len = len(p2)
            best_path = list(p2)

    if best_len == comp_size:
        return [largest_comp[i] for i in best_path]

    # Phase 2: Branch and Bound with better upper bounds
    # Use articulation point decomposition for tighter bounds

    # Precompute biconnected components and articulation points
    def compute_ap_decomposition():
        """Compute articulation points and biconnected components."""
        disc = [-1] * comp_size
        low = [0] * comp_size
        parent = [-1] * comp_size
        ap = [False] * comp_size
        biconnected = []
        stack = []
        time_counter = [0]

        def dfs(u):
            time_counter[0] += 1
            disc[u] = low[u] = time_counter[0]
            children = 0
            for v in adj_list[u]:
                if disc[v] == -1:
                    parent[v] = u
                    children += 1
                    stack.append(u)
                    stack.append(v)
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # u is an articulation point if:
                    # 1) u is root of DFS tree and has two or more children
                    # 2) u is not root and low[v] >= disc[u]
                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                        # Extract biconnected component
                        comp = []
                        while True:
                            x = stack.pop()
                            comp.append(x)
                            if x == v:
                                break
                        biconnected.append(comp)
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
                    stack.append(u)
                    stack.append(v)

        for i in range(comp_size):
            if disc[i] == -1:
                stack.append(i)
                dfs(i)
                if stack:
                    comp = []
                    while stack:
                        comp.append(stack.pop())
                    if comp:
                        biconnected.append(comp)

        return ap, biconnected

    ap, biconnected = compute_ap_decomposition()

    def upper_bound_fast(path_indices, visited_mask):
        """Compute upper bound using component analysis."""
        unvisited_mask = ((1 << comp_size) - 1) & ~visited_mask
        if unvisited_mask == 0:
            return len(path_indices)

        e1 = path_indices[0]
        e2 = path_indices[-1]

        # Find components reachable from endpoints
        def find_reachable_components(unvisited_mask, endpoint1_idx, endpoint2_idx):
            remaining = unvisited_mask
            reachable_sizes = []

            while remaining:
                start_bit = remaining & (-remaining)
                start_idx = start_bit.bit_length() - 1

                comp_mask = 0
                stack = [start_idx]
                comp_mask |= (1 << start_idx)

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

                e1_nbrs = adj_bits[endpoint1_idx] & comp_mask
                e2_nbrs = adj_bits[endpoint2_idx] & comp_mask
                if e1_nbrs or e2_nbrs:
                    reachable_sizes.append(size)

            return reachable_sizes

        reachable_sizes = find_reachable_components(unvisited_mask, e1, e2)

        if not reachable_sizes:
            return len(path_indices)

        reachable_sizes.sort(reverse=True)
        if len(reachable_sizes) >= 2:
            return len(path_indices) + reachable_sizes[0] + reachable_sizes[1]
        else:
            return len(path_indices) + reachable_sizes[0]

    # B&B DFS with better ordering
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

            ub = upper_bound_fast(curr_path, curr_visited)
            if ub <= best_len:
                continue

            tail = curr_path[-1]
            nbrs = adj_list[tail]

            # Sort neighbors by degree in unvisited subgraph (most constrained first)
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

                if len(new_path) + (comp_size - new_visited.bit_count()) <= best_len:
                    continue

                dfs_stack.append((new_path, new_visited))

    return [largest_comp[i] for i in best_path]
