"""Construct 91 edges on 31 vertices minimizing C4s.
Strategy:
1. Build C4-free core (90 edges) from PG(2,5) polarity graph.
2. Add one edge that creates minimum C4s.
3. Local search with fast incremental C4 delta to reduce C4 count.
4. Final exhaustive single-swap polish.
"""
import time
import random
import math

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin under 150s

    n = 31
    q = 5

    # --- Step 1: Build C4-free core with 90 edges using PG(2,5) polarity ---
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, 3, 5)
                    nx, ny, nz = 1, (y * inv_x) % 5, (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, 3, 5)
                    nx, ny, nz = 0, 1, (z * inv_y) % 5
                else:
                    nx, ny, nz = 0, 0, 1
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)

    assert len(points) == 31, f"Expected 31 points, got {len(points)}"

    def adjacent(i, j):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[j]
        return (x1 * x2 + y1 * y2 - z1 * z2) % 5 == 0

    edges_core = []
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adjacent(i, j):
                edges_core.append([i, j])
                adj[i].add(j)
                adj[j].add(i)

    # Fallback if PG construction fails
    if len(edges_core) != 90:
        adj = [set() for _ in range(n)]
        edges_core = []
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.seed(42)
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if len(edges_core) >= 90:
                break
            creates_c4 = False
            for a in adj[u]:
                if a in adj[v]:
                    creates_c4 = True
                    break
            if not creates_c4:
                edges_core.append([u, v])
                adj[u].add(v)
                adj[v].add(u)

    # --- Step 2: Find best single edge to add to minimize C4s ---
    def count_added_c4s(u, v, adj_sets):
        count = 0
        for a in adj_sets[u]:
            for b in adj_sets[v]:
                if b in adj_sets[a]:
                    count += 1
        return count

    best_edge = None
    best_add = float('inf')

    for u in range(n):
        if time.time() > deadline - 30:
            break
        for v in range(u + 1, n):
            if v in adj[u]:
                continue
            add = count_added_c4s(u, v, adj)
            if add < best_add:
                best_add = add
                best_edge = [u, v]

    if best_edge is None:
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    best_edge = [u, v]
                    break
            if best_edge:
                break

    adj[best_edge[0]].add(best_edge[1])
    adj[best_edge[1]].add(best_edge[0])

    # --- Step 3: Set up bitsets and incremental C4 delta ---
    adj_bits = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)

    def count_c4_full(adj_bits_local):
        total = 0
        for u in range(n):
            au = adj_bits_local[u]
            for v in range(u + 1, n):
                cn = bin(au & adj_bits_local[v]).count('1')
                total += cn * (cn - 1) // 2
        return total // 4

    current_c4s = count_c4_full(adj_bits)

    random.seed(123)

    edges_set = set(tuple(sorted(e)) for e in edges_core)
    edges_set.add(tuple(sorted(best_edge)))
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges_list = list(edges_set)
    non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]

    def compute_swap_delta_fast(e1, e2, adj_bits_ref, current_c4s_ref):
        """Compute delta in C4 count for removing e1=(u1,v1) and adding e2=(u2,v2)."""
        u1, v1 = e1
        u2, v2 = e2

        # Count C4s destroyed by removing (u1,v1):
        destroyed = 0
        # C4 containing edge (u1,v1): u1-v1-x-y-u1 where x in N(u1)\{v1}, y in N(v1)\{u1}, x~y
        for x in range(n):
            if x == u1 or x == v1:
                continue
            if not (adj_bits_ref[u1] >> x) & 1:
                continue
            ny_mask = adj_bits_ref[v1] & ~(1 << u1) & ~(1 << x)
            cnt = bin(ny_mask & adj_bits_ref[x]).count('1')
            destroyed += cnt

        # Count C4s created by adding (u2,v2) in the graph WITHOUT e1:
        # Temporarily remove e1 from adj_bits
        adj_bits_ref[u1] &= ~(1 << v1)
        adj_bits_ref[v1] &= ~(1 << u1)

        created = 0
        for x in range(n):
            if x == u2 or x == v2:
                continue
            if not (adj_bits_ref[u2] >> x) & 1:
                continue
            ny_mask = adj_bits_ref[v2] & ~(1 << u2) & ~(1 << x)
            cnt = bin(ny_mask & adj_bits_ref[x]).count('1')
            created += cnt

        # Restore e1
        adj_bits_ref[u1] |= (1 << v1)
        adj_bits_ref[v1] |= (1 << u1)

        delta = created - destroyed
        return delta

    def apply_swap(e1, e2, adj_bits_ref):
        u1, v1 = e1
        u2, v2 = e2
        adj_bits_ref[u1] &= ~(1 << v1)
        adj_bits_ref[v1] &= ~(1 << u1)
        adj_bits_ref[u2] |= (1 << v2)
        adj_bits_ref[v2] |= (1 << u2)

    def revert_swap(e1, e2, adj_bits_ref):
        u1, v1 = e1
        u2, v2 = e2
        adj_bits_ref[u2] &= ~(1 << v2)
        adj_bits_ref[v2] &= ~(1 << u2)
        adj_bits_ref[u1] |= (1 << v1)
        adj_bits_ref[v1] |= (1 << u1)

    # --- Phase 1: Greedy first-improvement local search ---
    improved = True
    stall_count = 0
    while improved and time.time() < deadline - 40:
        improved = False
        stall_count = 0
        for _ in range(5000):
            if time.time() > deadline - 40:
                break
            e1 = random.choice(edges_list)
            e2 = random.choice(non_edges_list)

            delta = compute_swap_delta_fast(e1, e2, adj_bits, current_c4s)

            if delta < 0:
                apply_swap(e1, e2, adj_bits)
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                current_c4s += delta
                improved = True
                edges_list = list(edges_set)
                non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]
                stall_count = 0
            else:
                stall_count += 1
            if stall_count > 2000 and not improved:
                break

    # --- Phase 2: Best-improvement local search ---
    if time.time() < deadline - 20:
        for phase in range(30):
            if time.time() > deadline - 20:
                break
            edges_list = list(edges_set)
            non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]

            best_improvement = 0
            best_swap = None

            for e1 in edges_list:
                if time.time() > deadline - 20:
                    break
                for e2 in non_edges_list:
                    if time.time() > deadline - 20:
                        break
                    delta = compute_swap_delta_fast(e1, e2, adj_bits, current_c4s)
                    if delta < best_improvement:
                        best_improvement = delta
                        best_swap = (e1, e2)

            if best_swap is not None and best_improvement < 0:
                e1, e2 = best_swap
                apply_swap(e1, e2, adj_bits)
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                current_c4s += best_improvement
            else:
                break

    # --- Phase 3: Simulated annealing ---
    if time.time() < deadline - 10:
        T = 5.0
        T_min = 0.05
        cool_rate = 0.9998

        best_c4s = current_c4s
        best_edges = set(edges_set)

        for step in range(20000):
            if time.time() > deadline - 10:
                break
            if not non_edges_list:
                break
            e1 = random.choice(edges_list)
            e2 = random.choice(non_edges_list)

            delta = compute_swap_delta_fast(e1, e2, adj_bits, current_c4s)

            accept = False
            if delta < 0:
                accept = True
            else:
                if T > T_min:
                    if random.random() < math.exp(-delta / T):
                        accept = True

            if accept:
                apply_swap(e1, e2, adj_bits)
                edges_set.discard(e1)
                edges_set.add(tuple(sorted(e2)))
                current_c4s += delta
                edges_list = list(edges_set)
                non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]

                if current_c4s < best_c4s:
                    best_c4s = current_c4s
                    best_edges = set(edges_set)

            T *= cool_rate
            if T < T_min:
                T = T_min

        if best_c4s < current_c4s:
            edges_set = set(best_edges)
            current_c4s = best_c4s
            adj_bits = [0] * n
            for (u, v) in edges_set:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)

    # --- Phase 4: Final exhaustive single-swap polish ---
    if time.time() < deadline - 2:
        edges_list = list(edges_set)
        non_edges_list = [(i, j) for i, j in all_pairs if (i, j) not in edges_set]

        best_improvement = 0
        best_swap = None

        for e1 in edges_list:
            if time.time() > deadline - 2:
                break
            for e2 in non_edges_list:
                delta = compute_swap_delta_fast(e1, e2, adj_bits, current_c4s)
                if delta < best_improvement:
                    best_improvement = delta
                    best_swap = (e1, e2)

        if best_swap is not None and best_improvement < 0:
            e1, e2 = best_swap
            apply_swap(e1, e2, adj_bits)
            edges_set.discard(e1)
            edges_set.add(tuple(sorted(e2)))
            current_c4s += best_improvement

    # Build final edge list
    final_edges = [list(e) for e in edges_set]
    final_edges.sort()

    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91

    return final_edges
