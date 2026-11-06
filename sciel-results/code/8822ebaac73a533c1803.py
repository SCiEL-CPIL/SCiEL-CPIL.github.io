import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def common_neighbors(adj, i, j):
        # Returns count of common neighbors of i and j (excluding i,j themselves)
        # adj is numpy int array
        ai = adj[i]
        aj = adj[j]
        # common neighbors: vertices k (k!=i,j) with ai[k]=1 and aj[k]=1
        common = np.logical_and(ai, aj)
        common[i] = 0
        common[j] = 0
        return int(common.sum())

    def is_c4_free(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    if common_neighbors(adj, i, j) >= 2:
                        return False
        return True

    def min_degree(adj, n):
        return int(adj.sum(axis=1).min())

    def c4_pair_count(adj, n):
        """Count pairs (i,j) with >=2 common neighbors (each such pair implies a C4)."""
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j]:
                    if common_neighbors(adj, i, j) >= 2:
                        cnt += 1
        return cnt

    def try_n(n):
        target = n - 39
        if target < 1:
            target = 1

        best_adj = None
        best_md = -1

        # Strategy: start with complete graph, remove edges to kill C4s,
        # prefer removing edges incident to higher-degree vertices.
        attempts = 30
        for attempt in range(attempts):
            if elapsed() > deadline - 5:
                break

            # Start with complete graph
            adj = np.ones((n, n), dtype=np.int8)
            np.fill_diagonal(adj, 0)

            # Randomize edge removal order by permuting vertices (symmetry breaking)
            # Actually, we'll do greedy: while C4 exists, find a C4-inducing pair,
            # remove an edge to break it.
            max_iter = 5000
            for it in range(max_iter):
                if elapsed() > deadline - 5:
                    break
                # Find a pair (i,j) with >=2 common neighbors
                found = False
                for i in range(n):
                    if elapsed() > deadline - 5:
                        break
                    for j in range(i + 1, n):
                        if adj[i][j]:
                            cn = common_neighbors(adj, i, j)
                            if cn >= 2:
                                # This pair forms a C4. Remove an edge to break it.
                                # Options: remove (i,j), or remove (i,k) or (j,k) for a common neighbor k.
                                # Removing (i,j) is simplest but reduces both degrees.
                                # Better: remove an edge from a high-degree vertex to a common neighbor.
                                # Get common neighbors list
                                common = np.logical_and(adj[i], adj[j])
                                common[i] = 0
                                common[j] = 0
                                cn_list = np.nonzero(common)[0]
                                # Choose which edge to remove: try to remove edge incident to higher-degree vertex
                                deg_i = adj[i].sum()
                                deg_j = adj[j].sum()
                                # If we remove (i,j): both lose 1
                                # If we remove (i,k): i loses 1, k loses 1
                                # We want to keep min degree high, so remove edge from higher degree vertex
                                # to a common neighbor, preferring the higher-degree endpoint.
                                if deg_i >= deg_j:
                                    # Remove (i, k) for some k in cn_list
                                    # Pick k that is also high degree to balance
                                    # Just pick first common neighbor
                                    k = cn_list[0]
                                    adj[i][k] = 0
                                    adj[k][i] = 0
                                else:
                                    k = cn_list[0]
                                    adj[j][k] = 0
                                    adj[k][j] = 0
                                found = True
                                break
                    if found:
                        break
                if not found:
                    break

            # Now check if C4-free
            if not is_c4_free(adj, n):
                # Try to fix by removing more edges
                for it in range(2000):
                    if elapsed() > deadline - 5:
                        break
                    found = False
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i][j]:
                                if common_neighbors(adj, i, j) >= 2:
                                    adj[i][j] = 0
                                    adj[j][i] = 0
                                    found = True
                                    break
                        if found:
                            break
                    if not found:
                        break

            if not is_c4_free(adj, n):
                continue

            md = min_degree(adj, n)
            if md > best_md:
                best_md = md
                best_adj = adj.copy()

            if md >= target:
                break

            # Try to add edges back to boost min degree (local improvement)
            # For each vertex with degree < target, try adding edges to non-neighbors
            # that don't create C4s.
            for it in range(500):
                if elapsed() > deadline - 5:
                    break
                improved = False
                degs = adj.sum(axis=1)
                for v in range(n):
                    if degs[v] < target:
                        # Try adding edge (v, u) for u not adjacent to v
                        # Check if adding (v,u) creates C4: need common_neighbors(v,u) >= 1 after adding
                        # Actually adding (v,u) creates C4 if there exist a,b with v-a-u-b-v
                        # i.e., a and b are common neighbors of v and u (in current graph)
                        # plus the new edge... Actually C4 after adding (v,u):
                        # v-a-u-b-v requires edges v-a, a-u, u-b, b-v
                        # a-u and b-v already exist (a,b common neighbors of v,u)
                        # and v-a, u-b exist. So if common_neighbors(v,u) >= 2, adding (v,u) creates C4.
                        # If common_neighbors(v,u) == 1, say a, then v-a-u and we need b with u-b and b-v,
                        # b must be adjacent to both u and v, so b is also common neighbor. Contradiction if only 1.
                        # So adding (v,u) creates C4 iff common_neighbors(v,u) >= 2.
                        non_neighbors = np.nonzero(1 - adj[v])[0]
                        non_neighbors = non_neighbors[non_neighbors != v]
                        for u in non_neighbors:
                            if common_neighbors(adj, v, u) < 2:
                                # Safe to add
                                adj[v][u] = 1
                                adj[u][v] = 1
                                degs[v] += 1
                                degs[u] += 1
                                improved = True
                                break
                if not improved:
                    break

            md = min_degree(adj, n)
            if md > best_md:
                best_md = md
                best_adj = adj.copy()

            if md >= target:
                break

        if best_adj is not None and best_md >= target:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if best_adj[i][j]:
                        edges.append([i, j])
            return n, edges

        return None

    # Try from largest n down
    for n in [47, 46, 45, 44, 43, 42]:
        result = try_n(n)
        if result is not None:
            return {'n': result[0], 'edges': result[1]}

    # Fallback: trivial graph
    return {'n': 39, 'edges': []}
