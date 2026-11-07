"""Combined: largest-first backtracking with verification and fallback."""
import time
from collections import deque
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


def _verify(result, trees, N=20):
    """Verify the packing is valid and return (is_valid, total_edges)."""
    seen = set()
    total = 0
    for k in range(1, 21):
        e = result[k - 1]
        if e is None:
            continue
        if len(e) != k:
            return False, 0
        for x in e:
            if type(x) is not int or x < 0 or x >= N:
                return False, 0
        if len(set(e)) != k:
            return False, 0
        for u, v in trees[k - 1]:
            a, b = e[u], e[v]
            key = (a, b) if a < b else (b, a)
            if key in seen:
                return False, 0
            seen.add(key)
            total += 1
    return True, total


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    N = 20
    trees = instance_trees(20, 611)

    # Process trees from largest to smallest (most constrained first)
    # T20 has 19 edges, T19 has 18, etc.
    order = list(range(19, -1, -1))  # indices 19 down to 0 (T20 first)

    # Track used edges in K20
    used_edges = set()
    result = [None] * 20

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    for idx in order:
        if time.time() > deadline - 5:
            break

        k = idx + 1  # tree size
        tree_edges = trees[idx]

        if k == 1:
            # T1: single vertex, no edges. Pick any vertex.
            for v in range(N):
                result[idx] = [v]
                break
            continue

        if k == 2:
            # T2: single edge (0,1). Find any unused edge.
            found = False
            for u in range(N):
                for w in range(u + 1, N):
                    ek = edge_key(u, w)
                    if ek not in used_edges:
                        result[idx] = [u, w]
                        used_edges.add(ek)
                        found = True
                        break
                if found:
                    break
            if not found:
                result[idx] = None
            continue

        # For k >= 3, use backtracking to embed the tree
        tree_n = k
        tree_edge_list = tree_edges

        # Build adjacency for the tree
        tree_adj = [[] for _ in range(tree_n)]
        for u, v in tree_edge_list:
            tree_adj[u].append(v)
            tree_adj[v].append(u)

        # Get BFS order of tree vertices starting from 0
        visited = [False] * tree_n
        bfs_order = [0]
        visited[0] = True
        q = deque([0])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    bfs_order.append(v)
                    q.append(v)

        # Backtracking: assign host vertices to tree vertices in BFS order
        assignment = [None] * tree_n

        def backtrack(pos):
            if time.time() > deadline - 2:
                return False
            if pos == tree_n:
                return True

            tree_v = bfs_order[pos]
            # Try each host vertex
            for host_v in range(N):
                # Check if host_v is already assigned to another tree vertex
                if host_v in assignment:
                    continue

                # Check if all edges from already-assigned neighbors are valid
                valid = True
                for nb in tree_adj[tree_v]:
                    if assignment[nb] is not None:
                        ek = edge_key(assignment[nb], host_v)
                        if ek in used_edges:
                            valid = False
                            break

                if not valid:
                    continue

                # Tentatively assign
                assignment[tree_v] = host_v
                # Mark edges as temporarily used
                temp_edges = set()
                for nb in tree_adj[tree_v]:
                    if assignment[nb] is not None:
                        ek = edge_key(assignment[nb], host_v)
                        temp_edges.add(ek)

                if backtrack(pos + 1):
                    # Commit the edges
                    for ek in temp_edges:
                        used_edges.add(ek)
                    return True

                # Undo
                assignment[tree_v] = None

        if backtrack(0):
            result[idx] = assignment[:]
        else:
            result[idx] = None

    # Verify the result
    ok, total = _verify(result, trees)
    if ok and total == 190:
        return result

    # Fallback: if backtracking failed for some trees, try a greedy approach
    # to at least get a valid partial packing
    if not ok:
        # Reset and try greedy in order from smallest to largest
        used_edges2 = set()
        result2 = [None] * 20

        for idx in range(20):  # T1 to T20
            k = idx + 1
            tree_edges = trees[idx]

            if k == 1:
                for v in range(N):
                    result2[idx] = [v]
                    break
                continue

            if k == 2:
                found = False
                for u in range(N):
                    for w in range(u + 1, N):
                        ek = edge_key(u, w)
                        if ek not in used_edges2:
                            result2[idx] = [u, w]
                            used_edges2.add(ek)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    result2[idx] = None
                continue

            # Greedy: try to embed using simple backtracking with random restarts
            tree_n = k
            tree_adj = [[] for _ in range(tree_n)]
            for u, v in tree_edges:
                tree_adj[u].append(v)
                tree_adj[v].append(u)

            visited = [False] * tree_n
            bfs_order = [0]
            visited[0] = True
            q = deque([0])
            while q:
                u = q.popleft()
                for v in tree_adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        bfs_order.append(v)
                        q.append(v)

            assignment = [None] * tree_n

            def backtrack_greedy(pos):
                if time.time() > deadline - 1:
                    return False
                if pos == tree_n:
                    return True

                tree_v = bfs_order[pos]
                for host_v in range(N):
                    if host_v in assignment:
                        continue

                    valid = True
                    for nb in tree_adj[tree_v]:
                        if assignment[nb] is not None:
                            ek = edge_key(assignment[nb], host_v)
                            if ek in used_edges2:
                                valid = False
                                break

                    if not valid:
                        continue

                    assignment[tree_v] = host_v
                    temp_edges = set()
                    for nb in tree_adj[tree_v]:
                        if assignment[nb] is not None:
                            ek = edge_key(assignment[nb], host_v)
                            temp_edges.add(ek)

                    if backtrack_greedy(pos + 1):
                        for ek in temp_edges:
                            used_edges2.add(ek)
                        return True

                    assignment[tree_v] = None

                return False

            if backtrack_greedy(0):
                result2[idx] = assignment[:]
            else:
                result2[idx] = None

        ok2, total2 = _verify(result2, trees)
        if ok2:
            return result2

    # Final safety: return best valid result found
    return result
