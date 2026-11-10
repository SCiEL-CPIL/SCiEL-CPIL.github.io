import numpy as np

def construct():
    # Build a 31-vertex graph with 91 edges using a projective-geometry-inspired
    # polarity construction on PG(2,5), then add the single extra edge that
    # minimises the number of 4-cycles created.
    #
    # Vertices: the 31 points of PG(2,5)
    # Base edges: pairs {u,v} with u^T M v = 0 (M = diag(1,1,4))
    # This gives 90 edges (one fewer than required).
    # We then add the non-edge that creates the fewest new 4-cycles.

    # Generate all 31 points of PG(2,5)
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    pts = (1, (y * inv_x) % 5, (z * inv_x) % 5)
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    pts = (0, 1, (z * inv_y) % 5)
                else:
                    pts = (0, 0, 1)
                if pts not in points:
                    points.append(pts)

    point_to_idx = {p: i for i, p in enumerate(points)}
    n = 31

    # Polarity matrix M = diag(1, 1, 4)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            val = (u[0] * v[0] + u[1] * v[1] + 4 * u[2] * v[2]) % 5
            if val == 0:
                edges.append([i, j])

    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int32)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1

    # The number of 4-cycles through a non-edge (u,v) equals (A^2)[u][v]
    # because each pair of common neighbours of u and v forms a 4-cycle u-w1-v-w2-u.
    # Actually, the number of 4-cycles created by adding edge (u,v) is
    # the number of pairs (w1, w2) such that u~w1, w1~v, v~w2, w2~u with w1 != w2.
    # This is exactly (A^2)[u][v] - common_neighbours(u,v) ... no.
    #
    # More carefully: a 4-cycle containing the new edge (u,v) is a path of length 3
    # from u to v: u - a - b - v. The number of such paths is (A^3)[u][v].
    # Each 4-cycle u-a-b-v-u uses the edge (u,v) and corresponds to exactly one
    # such path (going the other direction gives the same cycle, but we count
    # subgraphs so each 4-cycle is counted once per edge in it).
    #
    # So adding edge (u,v) creates (A^3)[u][v] new 4-cycles.
    # Wait - actually each 4-cycle has 4 edges, and when we add one edge, the
    # number of new 4-cycles is the number of length-3 paths between u and v.
    # Each such path u-a-b-v combined with the new edge u-v forms a 4-cycle
    # (assuming a != b and all distinct, which holds in simple graphs for
    # length-3 paths in triangle-free-ish graphs, but let's be careful).
    #
    # In a simple graph, a path u-a-b-v with u,v,a,b distinct gives a 4-cycle
    # u-a-b-v-u. If a == v or b == u, that's not a valid path. Since A^3 counts
    # walks of length 3, we need to subtract invalid ones. But for our purposes,
    # (A^3)[u][v] is a good proxy and likely exact for counting distinct 4-cycles
    # that use the new edge, since the graph is sparse enough.
    #
    # Let's just use (A^3)[u][v] as the cost.

    A3 = np.linalg.matrix_power(A, 3)

    # Find the non-edge with minimum (A^3)[u][v]
    min_val = float('inf')
    best_edge = None

    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                val = A3[i, j]
                if val < min_val:
                    min_val = val
                    best_edge = [i, j]

    # Add the best edge
    if best_edge is not None:
        edges.append(best_edge)

    # Ensure exactly 91 edges
    if len(edges) < 91:
        edge_set = set((min(u, v), max(u, v)) for u, v in edges)
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) not in edge_set:
                    edges.append([i, j])
                    if len(edges) == 91:
                        break
            if len(edges) == 91:
                break
    elif len(edges) > 91:
        # This shouldn't happen, but trim if needed
        edges = edges[:91]

    return edges
