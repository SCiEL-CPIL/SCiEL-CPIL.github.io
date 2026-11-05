import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            components.append(comp)

    components.sort(key=len, reverse=True)

    best_path = []
    best_len = 0
    deadline = time.time() + 50.0

    # Greedy initial lower bound (worst-first to avoid trapping)
    def greedy_path(start, comp_set):
        path = [start]
        visited = {start}
        while True:
            curr = path[-1]
            candidates = [nei for nei in adj[curr] if nei not in visited and nei in comp_set]
            if not candidates:
                break
            # Worst-first: pick neighbor with fewest unvisited neighbors
            best_nei = min(candidates, key=lambda x: len(adj[x] & comp_set - visited))
            path.append(best_nei)
            visited.add(best_nei)
        return path

    if components:
        largest_comp = components[0]
        largest_set = set(largest_comp)
        for start in largest_comp:
            if time.time() > deadline:
                break
            gp = greedy_path(start, largest_set)
            if len(gp) > best_len:
                best_len = len(gp)
                best_path = gp

    # Component-level upper bound: sum of sizes of components reachable from path endpoints
    def reachable_upper_bound(path, comp_set):
        """
        Upper bound on path extension: for each component of (comp_set - visited),
        it can only be visited if it's adjacent to one of the path endpoints.
        """
        visited_set = set(path)
        unvisited = comp_set - visited_set
        if not unvisited:
            return 0

        # Find components of unvisited subgraph
        unvisited_comp = []
        seen = set()
        for node in unvisited:
            if node not in seen:
                comp = []
                stack = [node]
                seen.add(node)
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for nei in adj[u]:
                        if nei in unvisited and nei not in seen:
                            seen.add(nei)
                            stack.append(nei)
                unvisited_comp.append(comp)

        # Endpoints of current path
        endpoints = set(path)
        if len(path) >= 2:
            endpoints = {path[0], path[-1]}
        else:
            endpoints = set(path)

        # A component is reachable if any of its vertices is adjacent to an endpoint
        reachable_size = 0
        for comp in unvisited_comp:
            comp_set_local = set(comp)
            for ep in endpoints:
                if any(nei in comp_set_local for nei in adj[ep]):
                    reachable_size += len(comp)
                    break

        return len(path) + reachable_size

    # Forced-endpoint check: count unvisited vertices with degree 1 in the
    # "active" subgraph (unvisited + path endpoints). If > 2, prune.
    def count_forced_endpoints(path, comp_set):
        """
        Count vertices in unvisited that have degree 1 in the active graph
        (unvisited vertices + endpoints of path, with original edges).
        Each such vertex MUST be an endpoint of the final path.
        If we have > 2 such vertices (and they can't all be the two endpoints), prune.
        """
        visited_set = set(path)
        unvisited = comp_set - visited_set
        if not unvisited:
            return 0

        endpoints = {path[0], path[-1]} if len(path) >= 2 else set(path)
        active = unvisited | endpoints

        forced = 0
        for v in unvisited:
            deg = 0
            for nei in adj[v]:
                if nei in active:
                    deg += 1
            if deg == 1:
                forced += 1

        # The final path has exactly 2 endpoints. At most 2 forced vertices can
        # be endpoints. If forced > 2, impossible.
        return forced

    # B&B on each component
    for comp in components:
        if len(comp) <= best_len:
            continue

        comp_set = set(comp)
        comp_nodes = comp

        for start in comp_nodes:
            if time.time() > deadline:
                break

            path = [start]
            visited = {start}

            def dfs(curr_path, visited_set):
                nonlocal best_path, best_len, deadline

                if time.time() > deadline:
                    return

                curr_len = len(curr_path)

                if curr_len > best_len:
                    best_len = curr_len
                    best_path = curr_path[:]

                # If we already have a Hamiltonian path for this component, stop
                if curr_len == len(comp):
                    return

                unvisited_in_comp = comp_set - visited_set
                if not unvisited_in_comp:
                    return

                # Upper bound via reachable components
                ub = reachable_upper_bound(curr_path, comp_set)
                if ub <= best_len:
                    return

                # Forced endpoint check
                forced = count_forced_endpoints(curr_path, comp_set)
                if forced > 2:
                    return

                last_node = curr_path[-1]
                candidates = [nei for nei in adj[last_node] if nei in unvisited_in_comp]
                if not candidates:
                    return

                # Sort candidates: prefer those that keep more options open
                # (fewer forced endpoints, higher degree in unvisited)
                def score_candidate(node):
                    # Prefer nodes whose addition doesn't create too many forced endpoints
                    # and which have more unvisited neighbors
                    unv_neis = len(adj[node] & unvisited_in_comp)
                    return unv_neis

                candidates.sort(key=score_candidate, reverse=True)

                for cand in candidates:
                    if time.time() > deadline:
                        return
                    new_path = curr_path + [cand]
                    new_visited = visited_set | {cand}
                    dfs(new_path, new_visited)

            dfs(path, visited)

            if best_len == len(comp):
                break

    if not best_path:
        if P['edges']:
            return [P['edges'][0][0], P['edges'][0][1]]
        return [0]

    return best_path
# EVOLVE-BLOCK-END
