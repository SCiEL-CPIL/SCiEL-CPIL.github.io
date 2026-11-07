"""Packing trees T1..T20 into K20 - direct backtracking approach."""
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
    deadline = time.time() + 140  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    n = 20

    # Build all edges of K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Process trees from largest to smallest
    tree_order = list(range(20))
    tree_order.sort(key=lambda idx: -(idx + 1))

    # Precompute tree adjacency
    tree_info = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k <= 2:
            tree_info.append(None)
            continue
        tree_adj = {v: [] for v in range(k)}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        # Order by degree descending, then by vertex index
        ordered_verts = sorted(range(k), key=lambda v: (-len(tree_adj[v]), v))
        tree_info.append((tree_edges, tree_adj, ordered_verts))

    def try_embed(t_idx, avail, rng):
        """Try to embed tree t_idx into available edges using backtracking."""
        k = t_idx + 1
        if k == 1:
            return [rng.randint(0, n - 1)]
        if k == 2:
            for u, v in avail:
                return [u, v]
            return None

        tree_edges, tree_adj, ordered_verts = tree_info[t_idx]
        host_map = {}
        used_hosts = set()

        # Precompute which tree edges connect to each vertex
        # For each tree vertex, list of (neighbor, edge_index_in_tree_edges)
        vert_edges = {v: [] for v in range(k)}
        for u, v in tree_edges:
            vert_edges[u].append(v)
            vert_edges[v].append(u)

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                return True

            tv = ordered_verts[pos]
            candidates = []

            for h in range(n):
                if h in used_hosts:
                    continue
                # Check against already-assigned neighbors
                ok = True
                for nb in tree_adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in avail:
                            ok = False
                            break
                if ok:
                    # Score candidate: prefer hosts with fewer remaining available edges
                    # to leave more flexibility for other trees
                    remaining = sum(1 for j in range(n) if j != h and (min(h, j), max(h, j)) in avail)
                    candidates.append((remaining, h))

            if not candidates:
                return False

            # Sort by fewest remaining edges first (most constrained first)
            candidates.sort(key=lambda x: x[0])
            # Add some randomness among top candidates
            if len(candidates) > 1:
                top_n = min(len(candidates), 5)
                top = candidates[:top_n]
                rng.shuffle(top)
                rest = candidates[top_n:]
                candidates = top + rest

            for _, h in candidates:
                host_map[tv] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_hosts.remove(h)

            return False

        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    # Main search with multiple random restarts
    best_result = None

    for attempt in range(2000):
        if time.time() > deadline:
            break

        rng = Random(attempt * 31 + 7)
        avail = set(all_edges)
        res = [None] * 20
        success = True

        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            k = t_idx + 1

            if k == 1:
                res[t_idx] = [rng.randint(0, n - 1)]
                continue

            embedded = None
            for trial in range(5):
                if time.time() > deadline:
                    success = False
                    break
                embedded = try_embed(t_idx, avail, rng)
                if embedded is not None:
                    break

            if not success:
                break
            if embedded is None:
                success = False
                break

            res[t_idx] = embedded
            if k >= 2:
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    avail.discard(edge)

        if success:
            if len(avail) == 0:
                best_result = res
                break

    if best_result is None:
        return [None] * 19 + [list(range(20))]

    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final
