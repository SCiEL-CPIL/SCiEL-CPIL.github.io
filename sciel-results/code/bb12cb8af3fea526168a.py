import time
import itertools
import random
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    start_time = time.time()
    deadline = start_time + 85.0

    def bfs_path(src, dst, forbidden):
        if src == dst:
            return [src]
        visited = {src}
        queue = deque([(src, [src])])
        while queue:
            node, path = queue.popleft()
            for nb in sorted(adj[node]):
                if nb == dst:
                    return path + [dst]
                if nb not in forbidden and nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    def check_feasible(branches, deadline):
        """Check if internally disjoint paths exist for all pairs using max-flow.
        Returns True if feasible, False otherwise."""
        if time.time() > deadline:
            return False
        branch_set = set(branches)
        k = len(branches)
        if k < 2:
            return True

        # Build a flow network: for each vertex v, split into v_in -> v_out with capacity 1
        # Branch vertices have capacity infinity (they can be used by multiple paths)
        # Source connects to all branch_in, sink connects from all branch_out
        # Actually, for pairwise disjoint paths, we need a different formulation.

        # Simpler check: try to find paths greedily and verify
        # This is a heuristic check, not exact
        used_internal = set()
        pairs = list(itertools.combinations(branches, 2))

        for u, v in pairs:
            forbidden = branch_set - {u, v} | used_internal
            path = bfs_path(u, v, forbidden)
            if path is None:
                return False
            for x in path[1:-1]:
                used_internal.add(x)

        return True

    def extract_paths(branches, deadline):
        """Extract internally disjoint paths for all pairs."""
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        paths = {}
        used_internal = set()

        # Order pairs: direct edges first, then by degree
        def pair_key(p):
            u, v = p
            direct = 0 if v in adj[u] else 1
            return (direct, len(adj[u]) + len(adj[v]))

        for ordering in range(4):
            if ordering == 0:
                pair_order = sorted(pairs, key=pair_key)
            elif ordering == 1:
                pair_order = sorted(pairs, key=pair_key, reverse=True)
            elif ordering == 2:
                pair_order = list(pairs)
                random.shuffle(pair_order)
            else:
                pair_order = sorted(pairs, key=lambda p: (1 if p[1] in adj[p[0]] else 0, len(adj[p[0]]) + len(adj[p[1]])))

            paths_try = {}
            used_internal_try = set()
            success = True

            for u, v in pair_order:
                if time.time() > deadline:
                    return None
                forbidden = branch_set - {u, v} | used_internal_try
                path = bfs_path(u, v, forbidden)
                if path is None:
                    success = False
                    break
                paths_try[(u, v)] = path
                for x in path[1:-1]:
                    used_internal_try.add(x)

            if success:
                return [paths_try[p] for p in pairs]

        return None

    def try_build(branches, deadline):
        """Try to build a subdivision for the given branch set."""
        if len(branches) < 2:
            return None
        result = extract_paths(branches, deadline)
        return result

    def expand_branches(current_branches, deadline):
        """Try to add more branches to the current set."""
        current_set = set(current_branches)
        best = list(current_branches)
        best_size = len(best)

        for _ in range(50):
            if time.time() > deadline - 2:
                break
            candidates = [v for v in range(n) if v not in current_set]
            # Sort by degree (higher degree = more connected)
            candidates.sort(key=lambda v: -len(adj[v]))
            random.shuffle(candidates[:10])  # Shuffle top candidates a bit

            added = False
            for v in candidates:
                if time.time() > deadline - 2:
                    break
                new_branches = best + [v]
                result = try_build(new_branches, deadline)
                if result is not None:
                    best = new_branches
                    current_set.add(v)
                    best_size = len(best)
                    added = True
                    break

            if not added:
                # Try swap: remove one, add another
                swapped = False
                for i in range(len(best)):
                    if time.time() > deadline - 2:
                        break
                    temp = best[:i] + best[i+1:]
                    temp_set = set(temp)
                    cands2 = [v for v in range(n) if v not in temp_set]
                    cands2.sort(key=lambda v: -len(adj[v]))
                    random.shuffle(cands2[:10])
                    for v in cands2[:15]:
                        if time.time() > deadline - 2:
                            break
                        new_b = temp + [v]
                        result = try_build(new_b, deadline)
                        if result is not None:
                            best = new_b
                            best_size = len(best)
                            swapped = True
                            break
                    if swapped:
                        break
                if not swapped:
                    break

        return best

    def find_max_clique(deadline):
        """Find maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []

        def bronk(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            if P or X:
                pivot = max(P | X, key=lambda v: len(adj[v] & P))
            else:
                return
            candidates = [v for v in P if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P = P - {v}
                X = X | {v}

        bronk(frozenset(), frozenset(range(n)), frozenset())
        cliques.sort(key=len, reverse=True)
        return cliques

    # Phase 1: Find largest clique and try to build subdivision
    cliques = find_max_clique(deadline)
    best_result = None
    best_size = 0

    for clique in cliques:
        if time.time() > deadline:
            break
        if len(clique) < 2:
            break
        if len(clique) <= best_size:
            continue
        result = try_build(clique, deadline)
        if result is not None:
            best_result = {'branches': list(clique), 'paths': result}
            best_size = len(clique)

    # Phase 2: Expand from best clique
    if best_size >= 2:
        base_branches = list(best_result['branches'])
    else:
        # Fallback: find any edge
        u, v = P['edges'][0]
        base_branches = [u, v]
        best_size = 2
        best_result = {'branches': [u, v], 'paths': [[u, v]]}

    current_branches = expand_branches(base_branches, deadline)
    if len(current_branches) > best_size:
        result = try_build(current_branches, deadline)
        if result is not None:
            best_size = len(current_branches)
            best_result = {'branches': list(current_branches), 'paths': result}

    # Phase 3: Try other initial sets
    if time.time() < deadline - 5:
        # Try random pairs and expand
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b:
                continue
            init = [a, b]
            result = try_build(init, deadline)
            if result is None:
                continue
            expanded = expand_branches(init, deadline)
            if len(expanded) > best_size:
                result = try_build(expanded, deadline)
                if result is not None:
                    best_size = len(expanded)
                    best_result = {'branches': list(expanded), 'paths': result}

        # Try high-degree vertex pairs
        vertices_by_degree = sorted(range(n), key=lambda v: -len(adj[v]))
        for i in range(min(10, len(vertices_by_degree))):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, min(10, len(vertices_by_degree))):
                if time.time() > deadline - 5:
                    break
                init = [vertices_by_degree[i], vertices_by_degree[j]]
                result = try_build(init, deadline)
                if result is None:
                    continue
                expanded = expand_branches(init, deadline)
                if len(expanded) > best_size:
                    result = try_build(expanded, deadline)
                    if result is not None:
                        best_size = len(expanded)
                        best_result = {'branches': list(expanded), 'paths': result}

    # Final: ensure we have a valid result
    if best_result is not None:
        branches = [int(b) for b in best_result['branches']]
        paths = [[int(x) for x in p] for p in best_result['paths']]
        return {'branches': branches, 'paths': paths}

    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
