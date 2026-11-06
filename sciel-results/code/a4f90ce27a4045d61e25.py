import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 150s kill

    n = 16
    target_edges = 65

    # Precompute all pairs for edge representation
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_set = set(all_pairs)

    # Bitset representation: adj[i] is an int where bit j is set iff edge (i,j) exists
    # This makes triangle detection: adj[i] & adj[j] & adj[k] style checks fast

    def build_adj(edges):
        adj = [0] * n
        deg = [0] * n
        for (u, v) in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def compute_score(adj, deg):
        """Compute max degree sum over all triangles using bitsets."""
        max_sum = 0
        for i in range(n):
            ai = adj[i]
            for j in range(i + 1, n):
                if ai & (1 << j):
                    # Common neighbors of i and j, with index > j to avoid duplicates
                    common = ai & adj[j]
                    # Only consider k > j
                    common &= ~((1 << (j + 1)) - 1)
                    while common:
                        lb = common & (-common)
                        k = lb.bit_length() - 1
                        s = deg[i] + deg[j] + deg[k]
                        if s > max_sum:
                            max_sum = s
                        common ^= lb
        return max_sum

    def build_turan_plus_one(partition, extra_edge):
        """Build K_{a,b} + one extra edge. partition = (list_A, list_B), extra_edge = (u,v)."""
        edges = []
        A, B = partition
        for u in A:
            for v in B:
                edges.append((u, v))
        edges.append(extra_edge)
        return edges

    def random_turan_start():
        """Generate a random near-Turán graph: random bipartition + one extra edge."""
        # Random bipartition of 16 vertices
        perm = list(range(n))
        random.shuffle(perm)
        # Try various splits; 8-8 is the Turán split
        a = 8
        A = perm[:a]
        B = perm[a:]
        # Extra edge: pick two vertices in the same part
        if random.random() < 0.5:
            u, v = random.sample(A, 2)
        else:
            u, v = random.sample(B, 2)
        edges = build_turan_plus_one((A, B), (u, v))
        return edges

    def try_local_search(edges, time_limit):
        """Targeted hill-climbing: attack the worst triangle by moving edges."""
        adj, deg = build_adj(edges)
        current_score = compute_score(adj, deg)
        best_score = current_score
        best_edges = list(edges)
        edge_set = set(edges)

        # Precompute: for each vertex, its neighbors as a set
        def neighbor_set(v):
            return [u for u in range(n) if adj[v] & (1 << u)]

        # Find worst triangle
        def find_worst_triangle(adj, deg):
            worst = None
            worst_sum = -1
            for i in range(n):
                ai = adj[i]
                for j in range(i + 1, n):
                    if ai & (1 << j):
                        common = ai & adj[j]
                        common &= ~((1 << (j + 1)) - 1)
                        while common:
                            lb = common & (-common)
                            k = lb.bit_length() - 1
                            s = deg[i] + deg[j] + deg[k]
                            if s > worst_sum:
                                worst_sum = s
                                worst = (i, j, k)
                            common ^= lb
            return worst, worst_sum

        elapsed = time.time() - start_time
        while time.time() - start_time < min(start_time + time_limit, deadline):
            worst_tri, worst_sum = find_worst_triangle(adj, deg)
            if worst_tri is None:
                break
            i, j, k = worst_tri

            # Strategy: try to "break" this triangle by removing one of its edges
            # and adding an edge elsewhere that doesn't create a worse triangle.
            # Prefer removing an edge incident to the highest-degree vertex in the triangle.
            tri_verts = [i, j, k]
            tri_verts.sort(key=lambda v: -deg[v])
            hi = tri_verts[0]  # highest degree

            # Try to remove edge (hi, x) for x in the other two, and add a new edge
            # that connects low-degree vertices.
            candidates_to_remove = []
            for x in tri_verts[1:]:
                if (hi, x) in edge_set or (x, hi) in edge_set:
                    candidates_to_remove.append((hi, x))

            # Also consider removing any edge of the triangle
            for a, b in [(i, j), (j, k), (i, k)]:
                if (a, b) in edge_set:
                    candidates_to_remove.append((a, b))

            random.shuffle(candidates_to_remove)

            improved = False
            for (rm_u, rm_v) in candidates_to_remove:
                # Try adding a random new edge
                for _ in range(5):
                    new_u = random.randint(0, n - 1)
                    new_v = random.randint(0, n - 1)
                    if new_u == new_v:
                        continue
                    if new_u > new_v:
                        new_u, new_v = new_v, new_u
                    if (new_u, new_v) in edge_set:
                        continue
                    if (new_u, new_v) == (rm_u, rm_v) or (new_u, new_v) == (rm_v, rm_u):
                        continue

                    # Apply swap
                    edge_set.discard((rm_u, rm_v))
                    edge_set.discard((rm_v, rm_u))
                    edge_set.add((new_u, new_v))
                    if new_u > new_v:
                        edge_set.discard((new_u, new_v))
                        edge_set.add((new_v, new_u))
                    else:
                        edge_set.discard((new_v, new_u))

                    # Update degrees
                    deg[rm_u] -= 1
                    deg[rm_v] -= 1
                    deg[new_u] += 1
                    deg[new_v] += 1

                    # Update adj
                    adj[rm_u] &= ~(1 << rm_v)
                    adj[rm_v] &= ~(1 << rm_u)
                    adj[new_u] |= (1 << new_v)
                    adj[new_v] |= (1 << new_u)

                    new_score = compute_score(adj, deg)

                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if new_score < best_score:
                            best_score = new_score
                            best_edges = list(edge_set)
                        break
                    else:
                        # Revert
                        edge_set.discard((new_u, new_v))
                        edge_set.discard((new_v, new_u))
                        edge_set.add((rm_u, rm_v))
                        if rm_u > rm_v:
                            edge_set.discard((rm_u, rm_v))
                            edge_set.add((rm_v, rm_u))
                        else:
                            edge_set.discard((rm_v, rm_u))

                        deg[rm_u] += 1
                        deg[rm_v] += 1
                        deg[new_u] -= 1
                        deg[new_v] -= 1

                        adj[rm_u] |= (1 << rm_v)
                        adj[rm_v] |= (1 << rm_u)
                        adj[new_u] &= ~(1 << new_v)
                        adj[new_v] &= ~(1 << new_u)

                if improved:
                    break

            if not improved:
                # Stuck: try a random perturbation
                for _ in range(3):
                    rm_u, rm_v = random.choice(list(edge_set))
                    new_u = random.randint(0, n - 1)
                    new_v = random.randint(0, n - 1)
                    if new_u == new_v:
                        continue
                    if new_u > new_v:
                        new_u, new_v = new_v, new_u
                    if (new_u, new_v) in edge_set:
                        continue

                    edge_set.discard((rm_u, rm_v))
                    edge_set.discard((rm_v, rm_u))
                    edge_set.add((new_u, new_v))

                    deg[rm_u] -= 1
                    deg[rm_v] -= 1
                    deg[new_u] += 1
                    deg[new_v] += 1

                    adj[rm_u] &= ~(1 << rm_v)
                    adj[rm_v] &= ~(1 << rm_u)
                    adj[new_u] |= (1 << new_v)
                    adj[new_v] |= (1 << new_u)

                    new_score = compute_score(adj, deg)
                    if new_score < current_score:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_edges = list(edge_set)
                        break
                    else:
                        # Revert
                        edge_set.discard((new_u, new_v))
                        edge_set.add((rm_u, rm_v))
                        deg[rm_u] += 1
                        deg[rm_v] += 1
                        deg[new_u] -= 1
                        deg[new_v] -= 1
                        adj[rm_u] |= (1 << rm_v)
                        adj[rm_v] |= (1 << rm_u)
                        adj[new_u] &= ~(1 << new_v)
                        adj[new_v] &= ~(1 << new_u)

        return best_edges, best_score

    # Multi-restart local search
    best_overall_score = 999
    best_overall_edges = None

    # First, try the canonical construction
    canonical_edges = build_turan_plus_one((list(range(8)), list(range(8, 16))), (0, 1))
    adj, deg = build_adj(canonical_edges)
    s = compute_score(adj, deg)
    best_overall_score = s
    best_overall_edges = list(canonical_edges)

    # Now do multiple restarts with different random near-Turán starts
    restarts = 0
    while time.time() - start_time < deadline:
        restarts += 1
        edges = random_turan_start()
        found_edges, found_score = try_local_search(edges, time_limit=8.0)
        if found_score < best_overall_score:
            best_overall_score = found_score
            best_overall_edges = found_edges
            print(f"Restart {restarts}: improved to {found_score}")

        # Also try a few random perturbations of the best found so far
        if best_overall_edges is not None and random.random() < 0.5:
            perturbed = list(best_overall_edges)
            # Remove a random edge, add a random edge
            rm = random.choice(perturbed)
            perturbed.remove(rm)
            while True:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (u, v) in perturbed:
                    continue
                break
            perturbed.append((u, v))
            found_edges, found_score = try_local_search(perturbed, time_limit=5.0)
            if found_score < best_overall_score:
                best_overall_score = found_score
                best_overall_edges = found_edges
                print(f"Perturbation: improved to {found_score}")

    # Final polish: one more targeted local search on the best
    if best_overall_edges is not None:
        found_edges, found_score = try_local_search(best_overall_edges, time_limit=10.0)
        if found_score < best_overall_score:
            best_overall_score = found_score
            best_overall_edges = found_edges

    # Convert to edge list format
    edges_list = [[u, v] for (u, v) in best_overall_edges]
    return {'n': n, 'edges': edges_list}
