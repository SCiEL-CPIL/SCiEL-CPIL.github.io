"""Combined packing program: large-first backtracking with bitmask edge tracking and multi-strategy restarts."""
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    trees = instance_trees(20, 611)
    n = 20

    # Precompute tree adjacencies
    tree_adj = [None] * n
    for i in range(n):
        k = i + 1
        adj = [[] for _ in range(k)]
        for (a, b) in trees[i]:
            adj[a].append(b)
            adj[b].append(a)
        tree_adj[i] = adj

    # Precompute edge index mapping for bitmask representation
    edge_to_idx = {}
    edge_list = []
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = idx
            edge_list.append((u, v))
            idx += 1
    total_edges = 190  # C(20,2)

    # Precompute for each host vertex, the bitmask of edges incident to it
    vertex_edge_mask = [0] * n
    for (u, v) in edge_list:
        ei = edge_to_idx[(u, v)]
        vertex_edge_mask[u] |= (1 << ei)
        vertex_edge_mask[v] |= (1 << ei)

    def get_bfs_order(adj, root=0):
        """Get BFS order from root."""
        k = len(adj)
        parent = [-1] * k
        order = [root]
        visited = [False] * k
        visited[root] = True
        queue = [root]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    order.append(v)
                    queue.append(v)
        return order, parent

    def get_dfs_order(adj, root=0):
        """Get DFS order from root."""
        k = len(adj)
        parent = [-1] * k
        order = []
        visited = [False] * k

        def dfs(u):
            visited[u] = True
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    parent[v] = u
                    dfs(v)

        dfs(root)
        return order, parent

    def find_embedding(tree_idx, used_mask, rng, order_type='bfs', root=0):
        """
        Finds an injective embedding of tree tree_idx into K20
        using edges not in used_mask (bitmask).
        Returns a list of length k where result[i] is host vertex for tree vertex i, or None.
        """
        k = tree_idx + 1
        if k == 1:
            return [0]

        adj = tree_adj[tree_idx]

        if order_type == 'bfs':
            order, parent = get_bfs_order(adj, root)
        else:
            order, parent = get_dfs_order(adj, root)

        host = [-1] * k
        used_hosts_mask = 0  # bitmask of used host vertices

        # Precompute: for each host vertex, count of available (unused) edges to other vertices
        # avail_count[hv] = number of edges (hv, nh) where nh != hv and edge not in used_mask
        avail_count = [0] * n
        for hv in range(n):
            mask = vertex_edge_mask[hv] & ~used_mask
            avail_count[hv] = bin(mask).count('1')

        def backtrack(idx):
            nonlocal used_hosts_mask
            if time.time() > deadline:
                return False

            if idx == k:
                return True

            tv = order[idx]
            p_tv = parent[tv]
            p_hv = host[p_tv] if p_tv != -1 else -1

            # Get candidates: unused host vertices that can connect to parent
            cands = []
            for hv in range(n):
                if (used_hosts_mask >> hv) & 1:
                    continue
                if p_tv != -1:
                    # Check if edge (hv, p_hv) is available
                    e = (min(hv, p_hv), max(hv, p_hv))
                    ei = edge_to_idx[e]
                    if (used_mask >> ei) & 1:
                        continue
                cands.append(hv)

            if not cands:
                return False

            # Score candidates: prefer those with more available connections
            # Use avail_count which is precomputed
            if len(cands) > 1:
                cands.sort(key=lambda hv: avail_count[hv], reverse=True)

            # Add some randomness for diversity
            if rng.random() < 0.3:
                rng.shuffle(cands)

            for hv in cands:
                host[tv] = hv
                used_hosts_mask |= (1 << hv)

                # Temporarily mark edges from hv as used for scoring purposes
                # (we don't actually add them to used_mask, but we can adjust avail_count)
                # For simplicity, we just recurse; the avail_count is approximate

                if backtrack(idx + 1):
                    return True

                used_hosts_mask &= ~(1 << hv)
                host[tv] = -1

            return False

        if backtrack(0):
            # Verify all edges are valid and unused
            for (a, b) in trees[tree_idx]:
                ha, hb = host[a], host[b]
                e = (min(ha, hb), max(ha, hb))
                ei = edge_to_idx[e]
                if (used_mask >> ei) & 1:
                    return None
            return host[:]
        else:
            return None

    def try_pack(seed, tree_order, order_types=None):
        """
        Try to pack all trees in tree_order using the given strategy.
        """
        rng = Random(seed)
        used_mask = 0
        res = [None] * n
        score = 0

        if order_types is None:
            order_types = []

        ot_map = {}
        for (ti, ot, rt) in order_types:
            ot_map[ti] = (ot, rt)

        for idx in tree_order:
            if time.time() > deadline:
                break

            k = idx + 1
            if k == 1:
                res[idx] = [0]
                continue

            ot, rt = ot_map.get(idx, ('bfs', 0))
            embedding = find_embedding(idx, used_mask, rng, order_type=ot, root=rt)

            if embedding is not None:
                res[idx] = embedding
                score += len(trees[idx])
                for u, v in trees[idx]:
                    hu = embedding[u]
                    hv = embedding[v]
                    e = (min(hu, hv), max(hu, hv))
                    ei = edge_to_idx[e]
                    used_mask |= (1 << ei)

        return res, score

    # Generate orderings
    tree_indices_desc = list(range(n))
    tree_indices_desc.sort(key=lambda i: len(trees[i]), reverse=True)

    tree_indices_asc = list(range(n))
    tree_indices_asc.sort(key=lambda i: len(trees[i]))

    orderings = [tree_indices_desc, tree_indices_asc]

    rng_order = Random(42)
    for _ in range(8):
        idx = list(range(n))
        rng_order.shuffle(idx)
        orderings.append(idx)

    # Also try: largest first, but with some randomization in the middle
    for _ in range(6):
        idx = list(range(n))
        idx.sort(key=lambda i: len(trees[i]), reverse=True)
        for _ in range(3):
            a = rng_order.randint(2, n - 3)
            b = rng_order.randint(2, n - 3)
            idx[a], idx[b] = idx[b], idx[a]
        orderings.append(idx)

    best_res = None
    best_score = 0

    seed_counter = 0
    for tree_order in orderings:
        for seed in range(300):
            if time.time() > deadline:
                break
            seed_counter += 1

            # Vary order types for some trees
            ot_list = []
            rng_ot = Random(seed_counter * 7 + 13)
            for idx in tree_order:
                if rng_ot.random() < 0.3:
                    ot = 'dfs' if rng_ot.random() < 0.5 else 'bfs'
                    rt = rng_ot.randint(0, idx)
                    ot_list.append((idx, ot, rt))
                else:
                    ot_list.append((idx, 'bfs', 0))

            res, score = try_pack(seed_counter, tree_order, ot_list)

            if score > best_score:
                best_score = score
                best_res = res

            if best_score == 190:
                break
        if best_score == 190:
            break

    # If we haven't achieved 190, try a second phase with different strategies
    if best_score < 190:
        remaining = [i for i in range(n) if best_res[i] is None]
        if remaining:
            # Rebuild used mask from best_res
            used_mask = 0
            for i in range(n):
                if best_res[i] is not None:
                    for u, v in trees[i]:
                        hu = best_res[i][u]
                        hv = best_res[i][v]
                        e = (min(hu, hv), max(hu, hv))
                        ei = edge_to_idx[e]
                        used_mask |= (1 << ei)

            for idx in sorted(remaining, key=lambda i: len(trees[i]), reverse=True):
                if time.time() > deadline:
                    break
                found = False
                for attempt in range(80):
                    rng = Random(seed_counter * 1000 + attempt)
                    ot = 'bfs' if attempt % 2 == 0 else 'dfs'
                    rt = attempt % (idx + 1)
                    embedding = find_embedding(idx, used_mask, rng, order_type=ot, root=rt)
                    if embedding is not None:
                        best_res[idx] = embedding
                        best_score += len(trees[idx])
                        for u, v in trees[idx]:
                            hu = embedding[u]
                            hv = embedding[v]
                            e = (min(hu, hv), max(hu, hv))
                            ei = edge_to_idx[e]
                            used_mask |= (1 << ei)
                        found = True
                        break
                if not found:
                    for root in range(idx + 1):
                        if time.time() > deadline:
                            break
                        for ot in ['bfs', 'dfs']:
                            if time.time() > deadline:
                                break
                            rng = Random(seed_counter * 10000 + root * 100 + (0 if ot == 'bfs' else 1))
                            embedding = find_embedding(idx, used_mask, rng, order_type=ot, root=root)
                            if embedding is not None:
                                best_res[idx] = embedding
                                best_score += len(trees[idx])
                                for u, v in trees[idx]:
                                    hu = embedding[u]
                                    hv = embedding[v]
                                    e = (min(hu, hv), max(hu, hv))
                                    ei = edge_to_idx[e]
                                    used_mask |= (1 << ei)
                                found = True
                                break
                        if found:
                            break

    if best_res is None:
        fallback = [None] * n
        fallback[0] = [0]
        fallback[1] = [0, 1]
        return fallback

    # Ensure types are correct
    final_result = []
    for i in range(n):
        if best_res[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_res[i]])

    return final_result
