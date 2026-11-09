import time
import random
from random import Random


def _instance_trees(n=20, seed=611):
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


def _embed_tree(edges, used_edges, rng):
    """Embed a tree into K20 avoiding used edges. Returns list of 20 ints or None."""
    n = 20
    if not edges:
        return list(range(n))
    # Build adjacency of the tree
    adj = [[] for _ in range(len(edges) + 1)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    # Try to find an embedding via backtracking with forward checking
    # Map tree vertex -> host vertex
    assignment = [None] * len(adj)
    used_hosts = set()

    # Order: root at 0, do a DFS
    order = []
    visited = [False] * len(adj)
    stack = [0]
    visited[0] = True
    while stack:
        node = stack.pop()
        order.append(node)
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                stack.append(nb)

    # Reverse so we embed in reverse DFS order (leaves first) to reduce backtracking
    embed_order = list(reversed(order))

    def backtrack(idx):
        if idx == len(embed_order):
            return True
        v = embed_order[idx]
        # Determine constraints: if any neighbor already assigned, host must differ and edge not used
        candidates = []
        for h in range(n):
            if h in used_hosts:
                continue
            # Check edge constraints with already-assigned neighbors
            ok = True
            for nb in adj[v]:
                if assignment[nb] is not None:
                    e = frozenset((h, assignment[nb]))
                    if e in used_edges:
                        ok = False
                        break
            if ok:
                candidates.append(h)
        rng.shuffle(candidates)
        for h in candidates:
            assignment[v] = h
            used_hosts.add(h)
            if backtrack(idx + 1):
                return True
            used_hosts.remove(h)
            assignment[v] = None
        return False

    if backtrack(0):
        return assignment
    return None


def construct():
    deadline = time.time() + 130  # safety margin before 150s
    trees = _instance_trees(20, 611)
    # tree k has k-1 edges; tree 0 (k=1) has 0 edges
    # Order trees by descending edge count (hardest first) for better forward checking
    order = list(range(20))
    order.sort(key=lambda i: len(trees[i]), reverse=True)

    best_result = [None] * 20
    best_edges = 0

    # Try several random strategies
    for attempt in range(2000):
        if time.time() > deadline - 5:
            break
        rng = Random(attempt)
        # Shuffle the embedding order among same-size groups occasionally
        cur_order = order[:]
        if attempt % 3 == 0:
            rng.shuffle(cur_order)

        used_edges = set()
        result = [None] * 20
        success = True
        for idx in cur_order:
            if time.time() > deadline - 5:
                success = False
                break
            edges = trees[idx]
            emb = _embed_tree(edges, used_edges, rng)
            if emb is None:
                success = False
                break
            result[idx] = emb
            # Mark edges as used
            for u, v in edges:
                e = frozenset((emb[u], emb[v]))
                used_edges.add(e)

        if success:
            total = sum(len(trees[i]) for i in range(20))
            if total > best_edges:
                best_edges = total
                best_result = result[:]

        # Early exit if we got all trees (max possible)
        if best_edges == sum(len(t) for t in trees):
            break

    # Validate and return
    if best_edges == 0:
        # Fallback: identity embedding of T20
        return [None] * 19 + [list(range(20))]

    # Ensure types are Python int and list
    final = []
    for r in best_result:
        if r is None:
            final.append(None)
        else:
            final.append([int(x) for x in r])
    return final
