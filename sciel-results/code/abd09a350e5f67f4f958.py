"""Packing trees T1..T20 into K20 using optimized backtracking with O(1) edge checks."""
import time
from random import Random
from collections import deque


def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
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
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def _embed_tree(tree_edges, n=20, used_edges=None, host_used_count=None, rng=None, deadline=None):
    """
    Try to embed tree_edges into K_n with edge-disjointness.
    Returns mapping dict from tree vertex -> host vertex, or None.
    host_used_count is a list of length n with counts of used edges per host.
    """
    if not tree_edges:
        return {}

    tree_verts = set()
    for u, v in tree_edges:
        tree_verts.add(u)
        tree_verts.add(v)
    tree_verts = sorted(tree_verts)
    k = len(tree_verts)

    adj = {v: [] for v in tree_verts}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    tree_degree = {v: len(adj[v]) for v in tree_verts}

    # Try multiple root choices and random restarts
    max_attempts = 200
    for attempt in range(max_attempts):
        if deadline and time.time() > deadline:
            return None

        # Choose root: alternate between high-degree and random
        if attempt < 5:
            # First few attempts: highest degree roots
            root = max(tree_verts, key=lambda v: (tree_degree[v], -v))
        elif attempt < 15:
            # Next: second-highest degree
            sorted_verts = sorted(tree_verts, key=lambda v: (-tree_degree[v], v))
            root = sorted_verts[attempt % len(sorted_verts)]
        else:
            # Random root
            root = rng.choice(tree_verts)

        # BFS from root to get parent relationships and depth
        parent = {root: -1}
        depth = {root: 0}
        queue = deque([root])
        order = [root]
        while queue:
            u = queue.popleft()
            for w in adj[u]:
                if w not in parent:
                    parent[w] = u
                    depth[w] = depth[u] + 1
                    queue.append(w)
                    order.append(w)

        # Create ordering: sort by (depth, -degree, v) to place high-degree vertices earlier
        # within each depth level, while maintaining parent-before-child
        sorted_order = sorted(order, key=lambda v: (depth[v], -tree_degree[v], v))

        # Verify parent comes before child
        pos_in_order = {v: i for i, v in enumerate(sorted_order)}
        valid = all(pos_in_order[parent[v]] < pos_in_order[v] for v in order[1:])
        if not valid:
            sorted_order = order

        mapping = {}
        host_used = set()

        # Assign root: try all available hosts in random order
        available = [h for h in range(n)]
        rng.shuffle(available)

        # For each root assignment, try to complete the embedding
        for host_root in available:
            if deadline and time.time() > deadline:
                return None

            mapping[root] = host_root
            host_used.add(host_root)

            # Backtracking assignment for remaining vertices
            def backtrack(idx):
                if idx == len(sorted_order):
                    return True
                if deadline and time.time() > deadline:
                    return False

                v = sorted_order[idx]
                p = parent[v]
                if p == -1:
                    return backtrack(idx + 1)

                host_p = mapping[p]

                # Candidates: hosts not yet used
                # Score: prefer hosts with fewer used edges (more free capacity)
                # Use host_used_count for O(1) access
                candidates = [h for h in range(n) if h not in host_used]

                # Sort by (used_count[h], random) to prefer less-used hosts with randomness
                candidates.sort(key=lambda h: (host_used_count[h], rng.random()))

                for h in candidates:
                    e = (min(host_p, h), max(host_p, h))
                    if e not in used_edges:
                        mapping[v] = h
                        host_used.add(h)
                        if backtrack(idx + 1):
                            return True
                        del mapping[v]
                        host_used.remove(h)

                return False

            if backtrack(0):
                return mapping

            # Clean up for next root attempt
            del mapping[root]
            host_used.discard(host_root)

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit)

    trees = instance_trees(20, 611)
    n = 20
    used_edges = set()
    host_used_count = [0] * n  # Track number of used edges incident to each host
    result = [None] * 20

    # Embed trees in order of decreasing size (more edges first)
    order = list(range(20))
    order.sort(key=lambda i: len(trees[i]), reverse=True)

    rng = Random(12345)

    for idx in order:
        if time.time() > deadline - 5.0:
            break
        k = idx + 1
        tree_edges = trees[idx]

        if not tree_edges:
            # Tree with 1 vertex (k=1)
            result[idx] = [0]
            continue

        # Try to embed
        mapping = _embed_tree(tree_edges, n, used_edges, host_used_count, rng, deadline)
        if mapping is not None:
            result_list = []
            for v in range(k):
                result_list.append(int(mapping[v]))
            result[idx] = result_list

            # Update used edges and host counts
            for u, v in tree_edges:
                hu, hv = mapping[u], mapping[v]
                e = (min(hu, hv), max(hu, hv))
                used_edges.add(e)
                host_used_count[hu] += 1
                host_used_count[hv] += 1

    # Second pass: try to embed any failed trees
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        for idx in order:
            if result[idx] is None and trees[idx]:
                if time.time() > deadline - 3.0:
                    break
                tree_edges = trees[idx]
                mapping = _embed_tree(tree_edges, n, used_edges, host_used_count, rng, deadline)
                if mapping is not None:
                    k = idx + 1
                    result_list = []
                    for v in range(k):
                        result_list.append(int(mapping[v]))
                    result[idx] = result_list
                    for u, v in tree_edges:
                        hu, hv = mapping[u], mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        used_edges.add(e)
                        host_used_count[hu] += 1
                        host_used_count[hv] += 1

    # Ensure types are correct
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]

    return result
