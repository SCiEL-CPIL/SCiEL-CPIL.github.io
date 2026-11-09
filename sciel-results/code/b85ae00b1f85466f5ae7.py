"""Packing trees T1..T20 into K20 using systematic backtracking with forward checking."""
import time
from random import Random


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


def _build_tree_info(tree_edges):
    """Build adjacency and degree info for a tree."""
    if not tree_edges:
        return {}, {}, []
    verts = set()
    for u, v in tree_edges:
        verts.add(u)
        verts.add(v)
    verts = sorted(verts)
    adj = {v: [] for v in verts}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    degree = {v: len(adj[v]) for v in verts}
    return adj, degree, verts


def _embed_with_backtracking(tree_edges, n, used_edges, rng, deadline, max_nodes=200000):
    """
    Embed tree_edges into K_n using backtracking with LCV heuristic and forward checking.
    Returns mapping dict {tree_vertex: host_vertex} or None.
    """
    if not tree_edges:
        return {}

    adj, degree, verts = _build_tree_info(tree_edges)
    k = len(verts)

    # Pick root as a leaf for better embedding properties
    leaves = [v for v in verts if degree[v] == 1]
    if leaves:
        root = rng.choice(leaves)
    else:
        root = verts[0]

    # BFS order from root
    order = [root]
    parent = {root: -1}
    visited = {root}
    queue = [root]
    idx = 0
    while idx < len(queue):
        u = queue[idx]
        idx += 1
        for w in adj[u]:
            if w not in visited:
                visited.add(w)
                parent[w] = u
                queue.append(w)
                order.append(w)

    # Precompute: for each host vertex h, which other vertices are NOT connected by a used edge
    # This allows fast candidate filtering
    # available_neighbors[h] = set of x such that (h,x) not in used_edges
    available_neighbors = [set(range(n)) for _ in range(n)]
    for h in range(n):
        available_neighbors[h].discard(h)
    for (a, b) in used_edges:
        if a < b:
            available_neighbors[a].discard(b)
            available_neighbors[b].discard(a)

    mapping = {}
    host_used = set()
    node_count = [0]

    def backtrack(pos):
        if node_count[0] > max_nodes:
            return False
        if time.time() > deadline:
            return False
        node_count[0] += 1

        if pos == len(order):
            return True

        v = order[pos]
        p = parent[v]

        if p == -1:
            # Root: any unused host vertex
            candidates = [h for h in range(n) if h not in host_used]
        else:
            hp = mapping[p]
            # Candidate must be unused AND edge (hp, h) not in used_edges
            candidates = [h for h in available_neighbors[hp] if h not in host_used]

        if not candidates:
            return False

        # LCV heuristic: sort candidates by how many options they leave for children
        # For each candidate h, count how many of v's unassigned children can connect to h
        def count_options(h):
            count = 0
            for w in adj[v]:
                if w not in mapping:
                    # Check if edge (h, some_unused) is possible for w
                    # w will need to connect to h, so (h, x) must not be in used_edges for some x
                    # Actually we just check if there exists any unused vertex x != h with (h,x) not in used_edges
                    # But simpler: just count how many unused vertices are in available_neighbors[h]
                    pass
            # Simpler heuristic: prefer candidates with more available neighbors
            return len(available_neighbors[h] - host_used)

        candidates.sort(key=count_options, reverse=True)
        rng.shuffle(candidates)  # Add some randomness to break ties

        for h in candidates:
            mapping[v] = h
            host_used.add(h)
            if backtrack(pos + 1):
                return True
            del mapping[v]
            host_used.discard(h)

        return False

    # Try multiple times with different random choices
    for attempt in range(50):
        if time.time() > deadline:
            return None
        mapping.clear()
        host_used.clear()
        node_count[0] = 0
        if backtrack(0):
            return dict(mapping)

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    trees = instance_trees(20, 611)
    n = 20
    used_edges = set()
    result = [None] * 20

    # Process trees in order of decreasing number of edges (most constrained first)
    # But we need to return in original order
    order = list(range(20))
    order.sort(key=lambda i: len(trees[i]), reverse=True)

    # Try multiple global random seeds to find the best packing
    best_total = 0
    best_result = [None] * 20

    for global_seed in range(100):
        if time.time() > deadline - 5.0:
            break

        rng = Random(global_seed)
        used_edges = set()
        result = [None] * 20
        success_count = 0

        for idx in order:
            if time.time() > deadline - 3.0:
                break

            k = idx + 1
            tree_edges = trees[idx]

            if not tree_edges:
                # Tree with 1 vertex (k=1)
                result[idx] = [0]
                success_count += 1
                continue

            mapping = _embed_with_backtracking(tree_edges, n, used_edges, rng, deadline)
            if mapping is not None:
                result_list = []
                for v in range(k):
                    result_list.append(int(mapping[v]))
                result[idx] = result_list
                success_count += 1

                for u, v in tree_edges:
                    hu, hv = mapping[u], mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    used_edges.add(e)

        # Calculate total edges embedded
        total_edges = sum(len(trees[i]) for i in range(20) if result[i] is not None)

        if total_edges > best_total:
            best_total = total_edges
            best_result = [list(r) if r is not None else None for r in result]

        # If we achieved full packing, we can stop early
        if best_total == 190:
            break

    return best_result
