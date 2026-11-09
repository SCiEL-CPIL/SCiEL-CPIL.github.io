import time
import random
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
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

N = 20

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s / 150s
    trees = instance_trees(N)
    # trees[k-1] is list of edges for tree with k vertices (vertices 0..k-1)
    # Build adjacency for each tree
    tree_adj = []
    for k in range(1, N+1):
        adj = [set() for _ in range(k)]
        for (a,b) in trees[k-1]:
            adj[a].add(b); adj[b].add(a)
        tree_adj.append(adj)

    # Order of processing trees. Largest first tends to be more constrained.
    # We'll try a few orders.
    # Host vertices 0..19.
    # used: set of frozenset edges used so far.

    best_result = None

    def check_time():
        return time.time() >= deadline

    def try_packing(order, rng):
        # order: list of tree indices (0-based, tree k has index k-1)
        used = set()
        assignment = [None]*N  # assignment[k-1] = list of host vertices for tree k vertices
        # We'll do backtracking over the order.
        # For speed, embed trees sequentially; if a tree can't be embedded, backtrack to previous.
        # To keep it tractable, do a full backtracking search with a node limit per restart.

        # Precompute for each tree the list of edges as (u,v) with u<v for host check.
        def embed_tree(tree_idx, depth, node_limit):
            # tree_idx is 0-based; k = tree_idx+1
            k = tree_idx + 1
            adj = tree_adj[tree_idx]
            # Choose an ordering of tree vertices to assign.
            # Use a fixed order: 0,1,...,k-1 but maybe better to start from a high-degree vertex.
            # We'll pick order: sort by degree desc.
            deg = [len(adj[v]) for v in range(k)]
            verts = list(range(k))
            verts.sort(key=lambda v: (-deg[v], v))
            # map: assigned[tree_v] = host_v
            assigned = [None]*k
            # For pruning, we need to check edges as we assign.
            # We'll do recursion.
            # Precompute for each tree vertex, its neighbors.
            # host candidates: 0..19 not yet used by this tree.
            # We'll maintain used_hosts set for this tree.
            host_used = [False]*N
            # Also we need to ensure edges to already-assigned tree vertices are unused in global used.
            # Let's do DFS with backtracking, with node limit.
            nodes = [0]
            def dfs(pos):
                if check_time():
                    return None
                if nodes[0] > node_limit:
                    return None
                nodes[0] += 1
                if pos == k:
                    # all assigned; build the list
                    return list(assigned)
                tv = verts[pos]
                # candidate host vertices
                for h in range(N):
                    if host_used[h]:
                        continue
                    # Check edges from tv to already-assigned neighbors
                    ok = True
                    for nb in adj[tv]:
                        if assigned[nb] is not None:
                            a = assigned[nb]
                            e = (min(a,h), max(a,h))
                            if e in used:
                                ok = False
                                break
                    if not ok:
                        continue
                    assigned[tv] = h
                    host_used[h] = True
                    res = dfs(pos+1)
                    if res is not None:
                        return res
                    assigned[tv] = None
                    host_used[h] = False
                return None
            return dfs(0)

        # Sequential with backtracking across trees is complex; instead do a simple approach:
        # embed trees in order; if one fails, we don't easily backtrack. 
        # Instead, do a proper backtracking over tree order.
        # Let's implement a recursive search over the order with the embed_tree as a subroutine
        # that returns one embedding; but we need multiple embeddings to backtrack.
        # Simpler: iterate over trees, for each try to find an embedding; if fail, restart with new rng.
        # This is a heuristic (greedy per tree) with restarts.

        for tree_idx in order:
            if check_time():
                return None
            k = tree_idx + 1
            if k == 1:
                # single vertex, no edges; any host vertex works, pick first free
                # but it must be distinct? Trees are edge-disjoint; vertex can repeat? 
                # "list of k distinct int host vertices" - within a tree distinct.
                # Across trees, vertices can repeat (only edges must be disjoint).
                # T1 has no edges, so it doesn't consume edges. Assign any vertex.
                # We'll assign vertex 0 (or any). But must be distinct within tree (trivially).
                assignment[tree_idx] = [0]
                continue
            res = embed_tree(tree_idx, 0, 200000)
            if res is None:
                return None
            assignment[tree_idx] = res
            # mark edges used
            for (a,b) in trees[tree_idx]:
                ha, hb = res[a], res[b]
                used.add((min(ha,hb), max(ha,hb)))
        return assignment

    # Try multiple orders and restarts.
    base_order = list(range(N))
    # largest first
    largest_first = sorted(base_order, key=lambda i: -(i+1))
    # Try several orderings.
    orders = [largest_first, base_order, sorted(base_order, key=lambda i: (i+1))]
    # Also random permutations.
    rng = Random(0)
    attempts = 0
    for oi in range(200):
        if check_time():
            break
        if oi < len(orders):
            order = orders[oi]
        else:
            order = list(range(N))
            rng.shuffle(order)
        # node limit for embed
        node_limit = 300000
        res = try_packing(order, rng)
        if res is not None:
            # Validate
            used = set()
            ok = True
            for tree_idx in range(N):
                k = tree_idx+1
                a = res[tree_idx]
                if a is None:
                    ok = False; break
                if len(a) != k:
                    ok = False; break
                if len(set(a)) != k:
                    ok = False; break
                for v in a:
                    if not (0 <= v <= 19):
                        ok = False; break
                for (u,v) in trees[tree_idx]:
                    e = (min(a[u],a[v]), max(a[u],a[v]))
                    if e in used:
                        ok = False; break
                    used.add(e)
                if not ok: break
            if ok:
                # Build result: entry k-1 is None or list
                result = [None]*N
                for tree_idx in range(N):
                    result[tree_idx] = [int(x) for x in res[tree_idx]]
                # Count edges
                # return
                return result
        attempts += 1

    # If we ran out of time without success, return best effort (only T20) 
    # Actually we should return something valid. Return the seed solution.
    return [None]*19 + [list(range(20))]
