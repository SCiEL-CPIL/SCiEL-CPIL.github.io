import time
import random
from random import Random

def construct():
    deadline = time.time() + 120.0  # safety margin before 150s kill
    n = 20

    def instance_trees(n=20, seed=611):
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

    trees = instance_trees(n)
    # trees[k-1] is list of edges for T_k; T1 has 0 edges

    def try_embed(edges, used_edges):
        """Try to embed a tree (list of (a,b) edges on 0..k-1) into K_n edge-disjointly.
        Returns a list of length k (image vertices) or None."""
        k = len(edges) + 1 if edges else (1 if len(edges)==0 else 0)
        # Determine k from the vertex labels in edges
        if not edges:
            # k=1 tree
            for v in range(n):
                return [v]
            return None
        maxv = max(max(a,b) for a,b in edges)
        k = maxv + 1
        # Try to find a mapping of 0..k-1 to distinct vertices in 0..n-1
        # such that no edge (map[a],map[b]) is in used_edges.
        # Use backtracking with heuristics.
        # Order vertices by degree in the tree (high degree first) to constrain early.
        deg = [0]*k
        for a,b in edges:
            deg[a]+=1; deg[b]+=1
        order = sorted(range(k), key=lambda v: -deg[v])
        # We need to assign each tree-vertex to a host vertex.
        # Start by fixing the highest-degree vertex to some host, then branch.
        # This is a CSP; use DFS.
        assign = {}
        host_to_tree = {}
        used_local = set()  # edges used in this embedding (host edges)
        all_host_edges = set()
        # Precompute host adjacency? We just need edge-disjointness with used_edges.
        # used_edges is a set of frozenset({u,v}) for global.

        def dfs(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                return True
            tv = order[idx]
            # candidate host vertices not yet used
            for hv in range(n):
                if hv in host_to_tree:
                    continue
                # Check edges from tv to already-assigned tree-neighbors
                ok = True
                for (a,b) in edges:
                    if a == tv:
                        nb = b
                    elif b == tv:
                        nb = a
                    else:
                        continue
                    if nb in assign:
                        hnb = assign[nb]
                        e = frozenset((hv, hnb))
                        if e in used_edges or e in used_local:
                            ok = False; break
                if not ok:
                    continue
                # tentatively assign
                assign[tv] = hv
                host_to_tree[hv] = tv
                # mark local edges to assigned neighbors
                new_local = []
                for (a,b) in edges:
                    if a == tv:
                        nb = b
                    elif b == tv:
                        nb = a
                    else:
                        continue
                    if nb in assign:
                        e = frozenset((hv, assign[nb]))
                        new_local.append(e)
                used_local.update(new_local)
                if dfs(idx+1):
                    return True
                # undo
                for e in new_local:
                    used_local.discard(e)
                del assign[tv]
                del host_to_tree[hv]
            return False

        if dfs(0):
            return [assign[v] for v in range(k)]
        return None

    # Greedy: embed trees one by one, trying multiple random orders / random restarts
    # We want to maximize total embedded edges. Prefer embedding all.
    # Strategy: try to embed in order of decreasing difficulty (larger trees first)
    # because they are more constrained. But also try natural order.

    def run(order):
        used = set()
        result = [None]*n
        total = 0
        for k in order:
            if time.time() > deadline:
                break
            edges = trees[k-1]
            if not edges:
                # T1: single vertex
                # pick any free vertex (no edges to conflict)
                for v in range(n):
                    result[k-1] = [v]
                    total += 0
                    break
                continue
            # try multiple random permutations of host vertices to break symmetry
            # We'll try a few random seedings.
            found = None
            # Try with different randomization of host order inside dfs by shuffling candidates
            # Simpler: call try_embed a few times with different random host permutations.
            for attempt in range(60):
                if time.time() > deadline:
                    break
                # randomize by permuting host labels? We can shuffle the candidate order.
                # Re-implement try_embed with a shuffled candidate order.
                found = try_embed_shuffled(edges, used, n, attempt, deadline)
                if found is not None:
                    break
            if found is not None:
                result[k-1] = found
                for a,b in edges:
                    used.add(frozenset((found[a], found[b])))
                total += len(edges)
        return result, total

    def try_embed_shuffled(edges, used_edges, n, seed, deadline):
        k = max(max(a,b) for a,b in edges) + 1
        rng = Random(seed)
        deg = [0]*k
        for a,b in edges:
            deg[a]+=1; deg[b]+=1
        order = sorted(range(k), key=lambda v: -deg[v])
        assign = {}
        host_to_tree = {}
        used_local = set()

        def dfs(idx):
            if time.time() > deadline:
                return False
            if idx == len(order):
                return True
            tv = order[idx]
            candidates = [hv for hv in range(n) if hv not in host_to_tree]
            rng.shuffle(candidates)
            for hv in candidates:
                ok = True
                for (a,b) in edges:
                    if a == tv:
                        nb = b
                    elif b == tv:
                        nb = a
                    else:
                        continue
                    if nb in assign:
                        hnb = assign[nb]
                        e = frozenset((hv, hnb))
                        if e in used_edges or e in used_local:
                            ok = False; break
                if not ok:
                    continue
                assign[tv] = hv
                host_to_tree[hv] = tv
                new_local = []
                for (a,b) in edges:
                    if a == tv:
                        nb = b
                    elif b == tv:
                        nb = a
                    else:
                        continue
                    if nb in assign:
                        e = frozenset((hv, assign[nb]))
                        new_local.append(e)
                used_local.update(new_local)
                if dfs(idx+1):
                    return True
                for e in new_local:
                    used_local.discard(e)
                del assign[tv]
                del host_to_tree[hv]
            return False

        if dfs(0):
            return [assign[v] for v in range(k)]
        return None

    best_result = [None]*n
    best_total = 0

    # Try several orderings: natural, reverse, by size, random
    orders = []
    natural = list(range(1, n+1))
    orders.append(natural)
    orders.append(list(reversed(natural)))
    # by decreasing size (larger first)
    bysize = sorted(range(1, n+1), key=lambda k: -(k-1))
    orders.append(bysize)
    # a few random
    rng2 = Random(123)
    for _ in range(8):
        o = list(natural)
        rng2.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline - 5:
            break
        result, total = run(order)
        if total > best_total:
            best_total = total
            best_result = result

    # Convert to proper types
    out = []
    for k in range(1, n+1):
        e = best_result[k-1]
        if e is None:
            out.append(None)
        else:
            out.append([int(x) for x in e])
    return out
