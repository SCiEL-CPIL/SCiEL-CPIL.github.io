"""Packing trees T1..T20 into K20 via backtracking with leaf-first ordering."""
from random import Random
import time
import sys
sys.setrecursionlimit(10000)


def _generate_trees(n=20, seed=611):
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


def _build_adjacency(tree_edges, k):
    """Build adjacency list for a tree with k vertices."""
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _find_embedding(tree_edges, k, used_edges, deadline):
    """
    Find an embedding of the tree into K20 such that no edge conflicts with used_edges.
    Returns a list of k integers (the image of each vertex) or None.
    Uses backtracking with leaf-first BFS ordering.
    """
    if k == 1:
        return [0]
    if k == 2:
        # Need an edge (u,v) not in used_edges
        for u in range(20):
            for v in range(u + 1, 20):
                if (u, v) not in used_edges and (v, u) not in used_edges:
                    return [u, v]
        return None

    adj = _build_adjacency(tree_edges, k)

    # Find a leaf to start from, then do BFS ordering
    # Pick a leaf as root
    leaves = [v for v in range(k) if len(adj[v]) == 1]
    if not leaves:
        # Shouldn't happen for a tree with k >= 2
        return None
    root = leaves[0]

    # BFS from root to get ordering
    from collections import deque
    order = [root]
    parent = [-1] * k
    parent[root] = -2
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                order.append(v)
                queue.append(v)

    # Now backtrack: assign vertices in order
    # mapping[i] = host vertex for tree vertex i
    mapping = [-1] * k
    mapping[root] = 0  # We can fix root to 0 by symmetry

    # Track which host vertices are used
    host_used = [False] * 20
    host_used[0] = True

    # Edges in the embedding so far
    emb_edges = set()

    # For backtracking, we need to try different assignments
    # Let's do iterative backtracking

    # Build the tree structure: for each vertex in order (after root), its parent
    # children_in_order[i] = list of tree-vertices that are children of order[i]
    # Actually, let's just process in order and for each vertex, its parent is known

    # Better approach: recursive backtracking with the BFS order
    # State: index in order, current mapping, host_used, emb_edges

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == k:
            return True

        v = order[idx]
        if idx == 0:
            # Root already assigned
            return backtrack(idx + 1)

        p = parent[v]
        ph = mapping[p]

        # Try all host vertices for v
        # Constraint: edge (ph, h) must not be in used_edges or emb_edges
        # Also h must not be used yet (since it's a tree, no two tree vertices map to same host)
        for h in range(20):
            if host_used[h]:
                continue
            # Check edge (ph, h)
            if (ph, h) in used_edges or (h, ph) in used_edges:
                continue
            if (ph, h) in emb_edges or (h, ph) in emb_edges:
                continue

            # Assign
            mapping[v] = h
            host_used[h] = True
            emb_edges.add((ph, h))

            if backtrack(idx + 1):
                return True

            # Undo
            mapping[v] = -1
            host_used[h] = False
            emb_edges.discard((ph, h))

        return False

    if backtrack(0):
        return mapping
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe before 150s kill

    trees = _generate_trees(20, 611)

    # Sort trees by size (largest first) for better constraint propagation
    # Actually, let's try largest first - they have the most edges and constrain the most
    order = sorted(range(20), key=lambda i: len(trees[i]), reverse=True)

    used_edges = set()
    results = [None] * 20
    total_edges = 0

    for idx in order:
        k = idx + 1
        tree_edges = trees[idx]
        if len(tree_edges) == 0:
            results[idx] = None
            continue

        emb = _find_embedding(tree_edges, k, used_edges, deadline)
        if emb is None:
            # Try a different root or restart
            # For now, return what we have
            break

        results[idx] = emb
        # Add edges to used set
        adj = _build_adjacency(tree_edges, k)
        for u, v in tree_edges:
            hu, hv = emb[u], emb[v]
            used_edges.add((hu, hv))
            used_edges.add((hv, hu))
        total_edges += len(tree_edges)

    # Verify all are int type
    for i in range(20):
        if results[i] is not None:
            results[i] = [int(x) for x in results[i]]

    return results
