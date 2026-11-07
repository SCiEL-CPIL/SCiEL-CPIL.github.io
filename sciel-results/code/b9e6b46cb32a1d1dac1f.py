"""Rewritten: randomized greedy packing with per-tree backtracking."""
# EVOLVE-BLOCK-START
import time
import random
from random import Random


def _instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees


def _embed_tree(edges, n, used, rng):
    """Embed a tree (list of (a,b) edges) into K_n avoiding used edges.
    Returns a dict tree_vertex -> host_vertex, or None on failure."""
    m = len(edges)
    if m == 0:
        return {}
    # Build adjacency of tree (multigraph-safe, but trees are simple)
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    verts = list(adj.keys())
    if len(verts) == 0:
        return {}
    # Pick a root
    root = verts[0]
    # BFS order to get a spanning tree ordering (tree is connected)
    order = [root]
    seen = {root}
    i = 0
    while i < len(order):
        v = order[i]; i += 1
        for w in adj.get(v, []):
            if w not in seen:
                seen.add(w); order.append(w)
    # order is a valid topological-ish order where each (except root) has a parent
    # Determine parent for each vertex
    parent = {root: None}
    i = 0
    while i < len(order):
        v = order[i]; i += 1
        for w in adj.get(v, []):
            if w not in parent:
                parent[w] = v
    # Assign root first, then children in order
    assignment = {}
    host_of = {}  # host -> tree vertex

    def backtrack(idx):
        if idx == len(order):
            return True
        v = order[idx]
        if v == root:
            candidates = list(range(n))
        else:
            pv = parent[v]
            if pv not in assignment:
                return False
            ph = assignment[pv]
            candidates = [h for h in range(n) if h not in host_of and (ph, h) not in used and (h, ph) not in used]
        # Shuffle for randomization
        rng.shuffle(candidates)
        for h in candidates:
            # Check all edges from v to already-assigned neighbors
            ok = True
            for w in adj.get(v, []):
                if w in assignment:
                    wh = assignment[w]
                    if (h, wh) in used or (wh, h) in used:
                        ok = False; break
            if not ok:
                continue
            # Also ensure h not already used by another tree vertex
            if h in host_of:
                continue
            assignment[v] = h
            host_of[h] = v
            if backtrack(idx + 1):
                return True
            del assignment[v]
            del host_of[h]
        return False

    if not backtrack(0):
        return None
    return assignment


def _run_once(trees, n, rng):
    """One restart: pack trees largest-first, greedily with backtracking.
    Returns (result_list, edge_count)."""
    used = set()
    result = [None] * n
    edge_count = 0
    order = sorted(range(n), key=lambda k: -(len(trees[k]) if trees[k] else 0))
    for k in order:
        edges = trees[k]
        if not edges:
            if k == 0:
                result[k] = None
            else:
                # tree with no edges but k>1 shouldn't happen except k=1
                result[k] = None
            continue
        # try to embed
        for attempt in range(20):
            assignment = _embed_tree(edges, n, used, rng)
            if assignment is not None:
                # record
                lst = [0] * (k + 1)
                for tv, h in assignment.items():
                    lst[tv] = h
                # verify distinct
                if len(set(lst)) != len(lst):
                    continue
                result[k] = lst
                for a, b in edges:
                    ha = lst[a]; hb = lst[b]
                    used.add((ha, hb))
                    used.add((hb, ha))
                edge_count += len(edges)
                break
        # if failed, result[k] stays None
    return result, edge_count


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s
    n = 20
    trees = _instance_trees(n, 611)
    total_target = sum(len(t) for t in trees)
    best_result = [None] * n
    best_count = 0
    # Seed the best with the identity T20 baseline
    best_result[19] = list(range(n))
    best_count = len(trees[19]) if trees[19] else 0

    # Precompute: we want to try many restarts
    base_seed = 611
    restart = 0
    while True:
        if time.time() > deadline:
            break
        rng = Random(base_seed + restart)
        res, cnt = _run_once(trees, n, rng)
        if cnt > best_count:
            best_count = cnt
            best_result = res
            if best_count >= total_target:
                break
        restart += 1
        # periodic check
        if restart % 25 == 0 and time.time() > deadline:
            break
    # ensure types
    out = []
    for k in range(n):
        v = best_result[k]
        if v is None:
            out.append(None)
        else:
            out.append([int(x) for x in v])
    return out
# EVOLVE-BLOCK-END
