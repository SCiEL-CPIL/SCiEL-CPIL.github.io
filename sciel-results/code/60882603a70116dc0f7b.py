"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 85  # Leave 5s safety margin

    # Build edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))

    # Adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Cycle 1: 0-1-2-...-20-0
    c1 = list(range(n))
    edges_c1 = set()
    for i in range(n):
        u, v = c1[i], c1[(i+1) % n]
        if u > v:
            u, v = v, u
        edges_c1.add((u, v))

    # Cycle 2: evens then odds: 0-2-4-...-20-1-3-...-19-0
    c2 = []
    for i in range(0, n, 2):
        c2.append(i)
    for i in range(1, n, 2):
        c2.append(i)
    edges_c2 = set()
    for i in range(n):
        u, v = c2[i], c2[(i+1) % n]
        if u > v:
            u, v = v, u
        edges_c2.add((u, v))

    if edges_c1 & edges_c2:
        return [c1]

    # Remaining edges after removing c1 and c2
    remaining_edges = edge_set - edges_c1 - edges_c2

    # Build adjacency for remaining graph
    rem_adj = {i: set() for i in range(n)}
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)

    def find_hamiltonian_cycle(adj_graph, deadline, start_vertex=0):
        """
        Find a Hamiltonian cycle using DFS with proper constraint propagation.
        Each vertex needs exactly 2 incident edges in the final cycle.
        We build a path from start_vertex; internal path vertices get their 2nd edge
        when the next vertex is appended. The last vertex needs to connect back to start.
        """
        n_local = len(adj_graph)

        # available[v] = set of neighbors of v whose edge to v is not yet used in the path
        # deg_used[v] = number of path-edges incident to v so far
        # For vertices in the path interior: deg_used should be 2
        # For path endpoints (start and current last): deg_used is 1
        # For vertices not in path: deg_used is 0

        def dfs(path, available, deg_used):
            if time.time() > deadline:
                return None

            last = path[-1]
            first = path[0]

            if len(path) == n_local:
                # Try to close: need edge (last, first)
                if first in available[last]:
                    return path[:]
                return None

            # Candidates: neighbors of last that are available
            candidates = []
            for v in available[last]:
                if v == first:
                    # Can only close at the end
                    if len(path) < n_local - 1:
                        continue
                elif v in path:
                    # Would create a premature cycle
                    continue
                candidates.append(v)

            if not candidates:
                return None

            # Warnsdorff heuristic: prefer candidates with fewer remaining options
            # After placing v, v will have deg_used[v]+1 used edges.
            # v needs total 2. So it needs (2 - deg_used[v] - 1) more from available.
            # If v is not first: it needs exactly 1 more edge after this placement.
            # If v is first (closing): it already has 1 used (from start's other end), needs 1 more which is this edge.
            def heuristic(v):
                if v == first:
                    return 0
                # After using edge (last, v), v has deg_used[v]+1 edges.
                # It needs 1 more. Count how many available neighbors v has (excluding last).
                rem = len(available[v]) - 1  # minus the edge to last which we're about to use
                return rem

            candidates.sort(key=heuristic)

            for v in candidates:
                # Check feasibility: after using edge (last, v)
                # - last loses v from available
                # - v loses last from available
                # - v's degree becomes deg_used[v] + 1
                # - if v != first, v needs exactly 1 more edge from its remaining available
                # - if v == first, we're done (handled above)
                # - For any vertex not in path (except v), it still needs 2 edges
                #   so it must have at least 2 available neighbors

                # Quick prune: if v != first and available[v] has only {last}, then v would
                # have no way to get its 2nd edge.
                if v != first:
                    if len(available[v]) <= 1:
                        continue

                # Apply move
                available[last].discard(v)
                available[v].discard(last)
                deg_used[last] += 1
                deg_used[v] += 1

                # Check all unvisited vertices still have >= 2 available neighbors
                valid = True
                for node in range(n_local):
                    if node not in path and node != v:
                        if len(available[node]) < 2:
                            valid = False
                            break

                if valid:
                    result = dfs(path + [v], available, deg_used)
                    if result is not None:
                        return result

                # Undo move
                available[last].add(v)
                available[v].add(last)
                deg_used[last] -= 1
                deg_used[v] -= 1

            return None

        init_available = {i: set(adj_graph[i]) for i in range(n_local)}
        init_deg = [0] * n_local
        return dfs([start_vertex], init_available, init_deg)

    # Try to find the third Hamiltonian cycle in the remaining graph
    c3 = None
    c3 = find_hamiltonian_cycle(rem_adj, deadline, start_vertex=0)

    if c3 is None and time.time() < deadline:
        # Try different starting vertices
        for sv in range(1, n):
            if time.time() > deadline:
                break
            c3 = find_hamiltonian_cycle(rem_adj, deadline, start_vertex=sv)
            if c3 is not None:
                break

    # Validate c3
    if c3 is not None:
        edges_c3 = set()
        valid = True
        for i in range(len(c3)):
            u, v = c3[i], c3[(i+1) % len(c3)]
            if u > v:
                u, v = v, u
            if (u, v) not in remaining_edges:
                valid = False
                break
            edges_c3.add((u, v))

        if valid and len(edges_c3) == len(c3) and len(c3) == n:
            return [c1, c2, c3]
        else:
            return [c1, c2]
    else:
        return [c1, c2]

# EVOLVE-BLOCK-END
