import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 120.0
    n = 31
    target_edges = 91

    def get_points(q):
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for x in range(q):
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        return points

    def build_polarity_graph(points, q, a, b, c, d, e, f):
        adj = np.zeros((n, n), dtype=np.int8)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                val = (a * p1[0] * p2[0] +
                       b * (p1[0] * p2[1] + p1[1] * p2[0]) +
                       c * (p1[0] * p2[2] + p1[2] * p2[0]) +
                       d * p1[1] * p2[1] +
                       e * (p1[1] * p2[2] + p1[2] * p2[1]) +
                       f * p1[2] * p2[2]) % q
                if val == 0:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    edges.append((i, j))
        return adj, edges

    def count_c4s(adj):
        A2 = np.dot(adj, adj)
        total = 0
        for u in range(n):
            for w in range(u + 1, n):
                k = A2[u, w]
                total += k * (k - 1) // 2
        return total // 2

    def min_cost_non_edge(adj, edge_set):
        best_cost = float('inf')
        best_pair = None
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in edge_set:
                    continue
                count = np.sum(adj[u][:, None] * adj[v][None, :] * adj)
                if count < best_cost:
                    best_cost = count
                    best_pair = (u, v)
        return best_pair, best_cost

    q = 5
    points = get_points(q)

    # Standard polarity: x1*x2 + y1*y2 - z1*z2 = 0 mod 5
    # f = -1 mod 5 = 4
    adj, edges = build_polarity_graph(points, q, 1, 0, 0, 1, 0, 4)
    edge_set = set(edges)
    m = len(edges)

    # If we have more than 90 edges, remove (m - 90) edges
    # to get a 90-edge C4-free base
    if m > 90:
        need_to_remove = m - 90
        # Remove edges that are in the "worst" length-3 path positions
        # Compute cost of each edge: sum over non-adjacent pairs (u,v) of
        # contribution to length-3 paths. Simpler: remove edges with highest
        # "removal benefit" = number of length-3 paths that use this edge.
        # A length-3 path u-a-b-v uses edges (u,a), (a,b), (b,v).
        # For each edge (a,b), count how many length-3 paths use it:
        # sum over u in N(a)\{b} and v in N(b)\{a} of 1 = (deg(a)-1)*(deg(b)-1)
        # This is a proxy. Better: for each edge, count the number of
        # non-edges (u,v) such that u-a-b-v is a path.
        # That's: for edge (a,b), count = sum_{u in N(a), u!=b} sum_{v in N(b), v!=a} [u!=v and (u,v) not in edge_set]
        # But for removal purposes, we want to remove edges that reduce the
        # minimum length-3 path count the most.
        # Simplest: remove edges with highest (deg(a)-1)*(deg(b)-1).

        deg = np.array([np.sum(adj[i, :]) for i in range(n)])
        edge_costs = []
        for i, (a, b) in enumerate(edges):
            cost = (deg[a] - 1) * (deg[b] - 1)
            edge_costs.append((cost, i))
        edge_costs.sort(reverse=True)

        # Remove the top (m-90) edges
        remove_indices = set(idx for _, idx in edge_costs[:need_to_remove])
        edges = [e for i, e in enumerate(edges) if i not in remove_indices]
        edge_set = set(edges)
        # Rebuild adj
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1

    m = len(edges)
    if m != 90:
        # Fallback: if not exactly 90, try other polarities
        polarities = [
            (1, 0, 0, 1, 0, 4),
            (1, 0, 0, 1, 0, 1),
            (1, 0, 0, 0, 0, 1),
            (0, 1, 0, 1, 0, 0),
            (1, 0, 1, 0, 0, 0),
        ]
        best_total_c4 = float('inf')
        best_edges_final = None
        for pol in polarities:
            if time.time() - start_time > DEADLINE:
                break
            adj2, edges2 = build_polarity_graph(points, q, *pol)
            if len(edges2) != 90:
                continue
            edge_set2 = set(edges2)
            best_pair, cost = min_cost_non_edge(adj2, edge_set2)
            if best_pair is None:
                continue
            u, v = best_pair
            adj2[u, v] = 1
            adj2[v, u] = 1
            edges2.append(best_pair)
            c4 = count_c4s(adj2)
            if c4 < best_total_c4:
                best_total_c4 = c4
                best_edges_final = list(edges2)
        if best_edges_final is not None:
            return [[u, v] for u, v in best_edges_final]
        # If still no 90-edge graph, use whatever we have
        if len(edges) > 90:
            edges = edges[:90]
        elif len(edges) < 90:
            # Add random edges
            adj = np.zeros((n, n), dtype=np.int8)
            edge_set = set()
            for u, v in edges:
                adj[u, v] = 1
                adj[v, u] = 1
                edge_set.add((u, v))
            while len(edges) < 90:
                best_pair, _ = min_cost_non_edge(adj, edge_set)
                if best_pair is None:
                    break
                u, v = best_pair
                edges.append(best_pair)
                edge_set.add(best_pair)
                adj[u, v] = 1
                adj[v, u] = 1
        return [[u, v] for u, v in edges[:90]]

    # Now we have exactly 90 edges. Find the best non-edge to add.
    best_pair, best_cost = min_cost_non_edge(adj, edge_set)

    if best_pair is not None:
        u, v = best_pair
        adj[u, v] = 1
        adj[v, u] = 1
        edges.append(best_pair)
        edge_set.add(best_pair)

    # Local search: try swapping edges to reduce C4 count
    if time.time() - start_time < DEADLINE - 10:
        current_c4 = count_c4s(adj)
        improved = True
        while improved and time.time() - start_time < DEADLINE - 5:
            improved = False
            # Try removing each edge and re-adding the best non-edge
            for i in range(len(edges)):
                if time.time() - start_time > DEADLINE - 5:
                    break
                u0, v0 = edges[i]
                # Remove edge
                adj[u0, v0] = 0
                adj[v0, u0] = 0
                edge_set.remove((u0, v0))
                edges.remove((u0, v0))

                # Find best non-edge to add
                best_pair2, _ = min_cost_non_edge(adj, edge_set)
                if best_pair2 is None:
                    # Restore
                    edges.append((u0, v0))
                    edge_set.add((u0, v0))
                    adj[u0, v0] = 1
                    adj[v0, u0] = 1
                    continue

                if best_pair2 == (u0, v0) or best_pair2 == (v0, u0):
                    # Same edge, no improvement
                    edges.append((u0, v0))
                    edge_set.add((u0, v0))
                    adj[u0, v0] = 1
                    adj[v0, u0] = 1
                    continue

                # Add new edge
                u1, v1 = best_pair2
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                edges.append(best_pair2)
                edge_set.add(best_pair2)

                new_c4 = count_c4s(adj)
                if new_c4 < current_c4:
                    current_c4 = new_c4
                    improved = True
                else:
                    # Revert
                    adj[u1, v1] = 0
                    adj[v1, u1] = 0
                    edges.remove(best_pair2)
                    edge_set.remove(best_pair2)
                    # Restore original edge
                    edges.append((u0, v0))
                    edge_set.add((u0, v0))
                    adj[u0, v0] = 1
                    adj[v0, u0] = 1

    return [[u, v] for u, v in edges]
