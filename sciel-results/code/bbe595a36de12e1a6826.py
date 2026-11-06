"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    n = 31
    target_edges = 91

    # Build the polarity graph of PG(2,5) - a C4-free graph with 31 vertices and 90 edges
    # PG(2,5) has 31 points. We use the standard construction.

    def pg2_points(q):
        points = []
        seen = set()
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    idx = 0
                    while idx < 3 and (x, y, z)[idx] == 0:
                        idx += 1
                    factor = (x, y, z)[idx]
                    inv_factor = pow(factor, q - 2, q)
                    norm = tuple((x, y, z)[i] * inv_factor % q for i in range(3))
                    if norm not in seen:
                        seen.add(norm)
                        points.append(norm)
        return points

    points = pg2_points(5)
    assert len(points) == 31

    # Try various bilinear forms to find one giving exactly 90 edges (C4-free)
    forms = [
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 + z1*x2 + z2*x1) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 + z1*y2 + z2*y1) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + 2*x1*y2 + 2*y1*x2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*z2 + z1*x2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + y1*z2 + z1*y2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + x1*y2 + y1*x2 + x1*z2 + z1*x2) % 5,
        lambda x1,y1,z1,x2,y2,z2: (x1*x2 + y1*y2 + z1*z2 + 3*x1*y2 + 3*y1*x2) % 5,
    ]

    best_adj_90 = None
    for form in forms:
        adj_temp = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2, p3 = points[i]
                q1, q2, q3 = points[j]
                if form(p1, p2, p3, q1, q2, q3) == 0:
                    adj_temp[i, j] = 1
                    adj_temp[j, i] = 1
        ec = int(adj_temp.sum() // 2)
        if ec == 90:
            best_adj_90 = adj_temp
            break

    if best_adj_90 is None:
        # Fallback: random graph
        best_adj_90 = np.zeros((n, n), dtype=np.int8)
        random.seed(42)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        for i, j in all_pairs[:90]:
            best_adj_90[i, j] = 1
            best_adj_90[j, i] = 1

    adj = best_adj_90.copy()

    # Count C4s efficiently
    def count_c4(adj_matrix):
        A = adj_matrix.astype(np.int32)
        A2 = A @ A
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                cn = A2[i, j]
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 2

    # Count C4s containing a specific edge (i,j)
    def c4s_containing_edge(adj_matrix, i, j):
        count = 0
        # A C4 containing edge (i,j): i-j-a-b-i
        # a is neighbor of j (a != i), b is neighbor of i (b != j), a-b is edge
        for a in range(n):
            if a == i or a == j:
                continue
            if adj_matrix[j, a] == 1:
                for b in range(n):
                    if b == i or b == j or b == a:
                        continue
                    if adj_matrix[i, b] == 1 and adj_matrix[a, b] == 1:
                        count += 1
        return count

    # Add one edge to get from 90 to 91, minimizing C4 increase
    best_add = None
    best_c4_after = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                # New C4s created by adding edge (i,j) = c4s_containing_edge on current graph
                # But we need to be careful: the edge (i,j) itself doesn't exist yet
                # c4s_containing_edge counts C4s that would contain (i,j) if it existed
                # This is exactly the number of new C4s created
                new_c4s = c4s_containing_edge(adj, i, j)
                if new_c4s < best_c4_after:
                    best_c4_after = new_c4s
                    best_add = (i, j)

    if best_add is not None:
        i, j = best_add
        adj[i, j] = 1
        adj[j, i] = 1

    current_c4 = count_c4(adj)
    best_adj = adj.copy()
    best_c4 = current_c4

    # Local search: swap edges to reduce C4 count
    random.seed(12345)
    no_improve = 0
    max_no_improve = 200000

    # Precompute edge and non-edge lists
    def get_edges_non_edges(adj_mat):
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mat[i, j] == 1:
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        return edges, non_edges

    edges, non_edges = get_edges_non_edges(adj)

    while time.time() < deadline and no_improve < max_no_improve:
        # Try random swaps
        improved = False
        batch_size = 100 if no_improve < 1000 else 50

        for _ in range(batch_size):
            i, j = random.choice(edges)
            k, l = random.choice(non_edges)

            # Skip if it's the same edge
            if (k == i and l == j) or (k == j and l == i):
                continue

            # Compute C4 change efficiently
            # Remove edge (i,j): removes c4s_containing_edge(adj, i, j) C4s
            # But some of those C4s might also contain edge (k,l) if (k,l) were present
            # Since (k,l) is not present, no overlap issue for removal
            # Add edge (k,l): adds c4s_containing_edge(adj, k, l) C4s
            # But we need to subtract C4s that contain both (i,j) and (k,l) - impossible since they'd share vertices
            # Actually, if {i,j} and {k,l} share a vertex, a C4 can't contain both edges
            # (a C4 has 4 edges, and two edges sharing a vertex in a C4 must be adjacent)
            # If they share a vertex, the C4 containing both would need that vertex to have degree >= 4 in the C4
            # which is impossible in a 4-cycle
            # So the delta is simply: -c4s_containing_edge(adj, i, j) + c4s_containing_edge(adj, k, l)
            # But wait, this is an approximation. Removing (i,j) might change the count for (k,l) and vice versa.
            # For accuracy, let's use the full count for promising moves.

            removed_c4 = c4s_containing_edge(adj, i, j)
            added_c4 = c4s_containing_edge(adj, k, l)
            delta = added_c4 - removed_c4

            if delta < 0:
                # Promising move, verify with full count
                adj_temp = adj.copy()
                adj_temp[i, j] = 0
                adj_temp[j, i] = 0
                adj_temp[k, l] = 1
                adj_temp[l, k] = 1

                if int(adj_temp.sum() // 2) != target_edges:
                    continue

                c4_temp = count_c4(adj_temp)
                if c4_temp < best_c4:
                    adj = adj_temp
                    best_adj = adj.copy()
                    best_c4 = c4_temp
                    no_improve = 0
                    improved = True
                    edges, non_edges = get_edges_non_edges(adj)
                    break
                else:
                    no_improve += 1
            else:
                no_improve += 1

        if not improved and no_improve > 5000:
            # Try targeted approach: find worst edge and best non-edge
            if time.time() < deadline:
                # Compute C4 counts for all edges
                edge_c4 = {}
                for e in edges:
                    edge_c4[e] = c4s_containing_edge(adj, e[0], e[1])

                # Find edges with high C4 count
                sorted_edges = sorted(edge_c4.items(), key=lambda x: -x[1])
                worst_edges = [e for e, c in sorted_edges[:20]]

                # Find non-edges with low C4 potential
                non_edge_c4 = {}
                for ne in non_edges:
                    non_edge_c4[ne] = c4s_containing_edge(adj, ne[0], ne[1])

                sorted_non_edges = sorted(non_edge_c4.items(), key=lambda x: x[1])
                best_non_edges = [e for e, c in sorted_non_edges[:20]]

                # Try all combinations of worst edges and best non-edges
                for we in worst_edges:
                    for bne in best_non_edges:
                        if set(we) == set(bne):
                            continue
                        i, j = we
                        k, l = bne

                        adj_temp = adj.copy()
                        adj_temp[i, j] = 0
                        adj_temp[j, i] = 0
                        adj_temp[k, l] = 1
                        adj_temp[l, k] = 1

                        if int(adj_temp.sum() // 2) != target_edges:
                            continue

                        c4_temp = count_c4(adj_temp)
                        if c4_temp < best_c4:
                            adj = adj_temp
                            best_adj = adj.copy()
                            best_c4 = c4_temp
                            no_improve = 0
                            edges, non_edges = get_edges_non_edges(adj)
                            break
                    if no_improve == 0:
                        break

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                result.append([i, j])

    # Ensure exactly 91 edges
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"

    return result
