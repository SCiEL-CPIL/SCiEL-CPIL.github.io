import time
from collections import deque
from random import Random


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


def construct():
    start_time = time.time()
    deadline = start_time + 145

    trees = _generate_trees(20, 611)
    n = 20

    # Precompute all edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Precompute tree adjacency
    tree_adj = []
    tree_leaves = []
    tree_max_degree = []
    for k in range(1, n + 1):
        edges = trees[k - 1]
        adj = [[] for _ in range(k)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        if not leaves:
            leaves = [0]
        tree_leaves.append(leaves)
        max_deg = max(len(adj[v]) for v in range(k)) if k > 1 else 0
        tree_max_degree.append(max_deg)

    def try_embed(tree_idx, available, rng, time_limit):
        """Try to embed tree tree_idx into available edges. Returns embedding list or None."""
        k = tree_idx + 1
        if k == 1:
            return [0]
        if k == 2:
            for (a, b) in available:
                return [a, b]
            return None

        tree_edges = trees[tree_idx]
        adj = tree_adj[tree_idx]
        leaves = tree_leaves[tree_idx]

        # Build available adjacency as sets for fast lookup
        avail_adj = [set() for _ in range(n)]
        for (a, b) in available:
            avail_adj[a].add(b)
            avail_adj[b].add(a)

        # Try multiple roots: prefer high-degree vertices first, then leaves
        # Actually, for backtracking, starting from a leaf is often better
        # because it has only one constraint (parent edge)
        # Let's try both: first high-degree, then leaves
        all_vertices = list(range(k))
        # Sort by degree descending for potential root candidates
        by_degree = sorted(all_vertices, key=lambda v: len(adj[v]), reverse=True)
        # Combine: try leaves first (most constrained in tree), then high-degree
        root_candidates = list(leaves) + [v for v in by_degree if v not in leaves]
        rng.shuffle(root_candidates)

        for root in root_candidates:
            if time.time() - start_time > time_limit:
                continue

            # Build BFS tree from root
            parent = [-1] * k
            children = [[] for _ in range(k)]
            visited = [False] * k
            visited[root] = True
            order_list = [root]
            queue = deque([root])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        children[u].append(v)
                        queue.append(v)
                        order_list.append(v)

            # Shuffle children for diversity
            for v in range(k):
                rng.shuffle(children[v])

            host = [-1] * k
            used_hosts = set()

            def backtrack(idx):
                if time.time() - start_time > time_limit:
                    return None

                if idx == len(order_list):
                    # Verify all edges are in available
                    for (u, v) in tree_edges:
                        hu, hv = host[u], host[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e not in available:
                            return None
                    return list(host)

                v = order_list[idx]

                if parent[v] == -1:
                    # Root: any unused host
                    candidates = [h for h in range(n) if h not in used_hosts]
                else:
                    pv = parent[v]
                    ph = host[pv]
                    if ph == -1:
                        return None
                    candidates = [h for h in avail_adj[ph] if h not in used_hosts]

                if not candidates:
                    return None

                # Forward checking: for each candidate, check if children can be placed
                # A child c of v needs at least one available neighbor of host[v]
                # that is not already used and not host[v] itself
                # We'll score candidates by how many available neighbors they have
                # after accounting for used hosts

                def score_candidate(h):
                    # Count available neighbors of h not in used_hosts
                    avail_count = len(avail_adj[h] - used_hosts)
                    # Penalize if this vertex has many children that need placement
                    # Each child needs a distinct available neighbor
                    num_children = len(children[v])
                    # If avail_count < num_children, this might be a problem
                    # but children can also use other paths... actually no,
                    # each child must connect to h via an edge
                    # So we need at least num_children available neighbors
                    # But some of those neighbors might be needed by other branches
                    # For now, use avail_count as primary score
                    # Prefer hosts with more available neighbors (more flexibility)
                    return -avail_count

                candidates.sort(key=score_candidate)

                for h in candidates:
                    host[v] = h
                    used_hosts.add(h)

                    # Forward check: ensure each child of v has at least one
                    # available neighbor of h that isn't used
                    feasible = True
                    for c in children[v]:
                        # c needs to connect to h, so needs an available neighbor of h
                        # that is not in used_hosts and not h itself
                        # Actually, c will be placed at some host nc such that (h, nc) in available
                        # So we need at least one available neighbor of h not in used_hosts
                        # But we also need distinct hosts for each child
                        # For a simple check: count available neighbors of h not in used_hosts
                        # If less than number of unplaced children, it's infeasible
                        pass  # Simplified: just try

                    res = backtrack(idx + 1)
                    if res is not None:
                        return res

                    host[v] = -1
                    used_hosts.discard(h)

                return None

            result = backtrack(0)
            if result is not None:
                return result

        return None

    # Process trees in decreasing size order
    order = list(range(19, -1, -1))  # T20 first

    # Try deterministic pass first
    rng_det = Random(42)
    result = [None] * 20
    available = set(all_edges)
    success = True

    for t_idx in order:
        if time.time() - start_time > deadline - 5:
            success = False
            break

        k = t_idx + 1
        if k == 1:
            result[t_idx] = [0]
            continue
        elif k == 2:
            edge_list = list(available)
            if not edge_list:
                success = False
                break
            (a, b) = rng_det.choice(edge_list)
            result[t_idx] = [a, b]
            available.discard((min(a, b), max(a, b)))
            continue

        # Adaptive time limit
        remaining_trees = 20 - t_idx
        time_limit = min(3.0, max(0.5, (deadline - 5 - (time.time() - start_time)) / max(1, remaining_trees)))
        emb = try_embed(t_idx, available, rng_det, time_limit)
        if emb is None:
            success = False
            break

        result[t_idx] = emb
        for (u, v) in trees[t_idx]:
            hu, hv = emb[u], emb[v]
            e = (min(hu, hv), max(hu, hv))
            if e in available:
                available.discard(e)
            else:
                success = False
                break
        if not success:
            break

    if success:
        # Verify
        used_edges = set()
        valid = True
        for t_idx in range(20):
            emb = result[t_idx]
            if emb is None:
                valid = False
                break
            if len(emb) != t_idx + 1:
                valid = False
                break
            if len(set(emb)) != t_idx + 1:
                valid = False
                break
            for x in emb:
                if not isinstance(x, int) or isinstance(x, bool) or x < 0 or x >= 20:
                    valid = False
                    break
            if not valid:
                break
            for (u, v) in trees[t_idx]:
                hu, hv = emb[u], emb[v]
                e = (min(hu, hv), max(hu, hv))
                if e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            if not valid:
                break

        if valid:
            for i in range(20):
                if result[i] is not None:
                    result[i] = [int(x) for x in result[i]]
            return result

    # Randomized restarts
    for restart in range(200):
        if time.time() - start_time > deadline - 5:
            break

        rng = Random(restart * 7919 + 13)
        result = [None] * 20
        available = set(all_edges)
        success = True

        for t_idx in order:
            if time.time() - start_time > deadline - 3:
                success = False
                break

            k = t_idx + 1
            if k == 1:
                result[t_idx] = [0]
                continue
            elif k == 2:
                edge_list = list(available)
                if not edge_list:
                    success = False
                    break
                (a, b) = rng.choice(edge_list)
                result[t_idx] = [a, b]
                available.discard((min(a, b), max(a, b)))
                continue

            remaining_trees = 20 - t_idx
            time_limit = min(3.0, max(0.3, (deadline - 3 - (time.time() - start_time)) / max(1, remaining_trees)))
            emb = try_embed(t_idx, available, rng, time_limit)
            if emb is None:
                success = False
                break

            result[t_idx] = emb
            for (u, v) in trees[t_idx]:
                hu, hv = emb[u], emb[v]
                e = (min(hu, hv), max(hu, hv))
                if e in available:
                    available.discard(e)
                else:
                    success = False
                    break
            if not success:
                break

        if success:
            # Verify
            used_edges = set()
            valid = True
            for t_idx in range(20):
                emb = result[t_idx]
                if emb is None:
                    valid = False
                    break
                if len(emb) != t_idx + 1:
                    valid = False
                    break
                if len(set(emb)) != t_idx + 1:
                    valid = False
                    break
                for x in emb:
                    if not isinstance(x, int) or isinstance(x, bool) or x < 0 or x >= 20:
                        valid = False
                        break
                if not valid:
                    break
                for (u, v) in trees[t_idx]:
                    hu, hv = emb[u], emb[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in used_edges:
                        valid = False
                        break
                    used_edges.add(e)
                if not valid:
                    break

            if valid:
                for i in range(20):
                    if result[i] is not None:
                        result[i] = [int(x) for x in result[i]]
                return result

    # Final fallback: return what we have
    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]
    return result
