import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n = 20
    best_edges = 0
    best_adj = None

    # Precompute edge list
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i

    def can_add(u, v, adj):
        """
        Returns True if adding edge (u, v) keeps the graph K3,3-free.
        """
        # Check triples (u, x, y) and (v, x, y) for x < y, x,y != u,v
        for x in range(n):
            if x == u or x == v:
                continue
            for y in range(x + 1, n):
                if y == u or y == v:
                    continue

                v_in_xy = (adj[x] >> v) & 1 and (adj[y] >> v) & 1
                u_in_xy = (adj[x] >> u) & 1 and (adj[y] >> u) & 1

                if v_in_xy:
                    count_uxy = (adj[u] & adj[x] & adj[y]).bit_count()
                    if count_uxy + 1 >= 3:
                        return False

                    count_vxy = (adj[v] & adj[x] & adj[y]).bit_count()
                    if u_in_xy:
                        if count_vxy + 1 >= 3:
                            return False
                    else:
                        if count_vxy >= 3:
                            return False

                elif u_in_xy:
                    count_uxy = (adj[u] & adj[x] & adj[y]).bit_count()
                    if count_uxy >= 3:
                        return False

                    count_vxy = (adj[v] & adj[x] & adj[y]).bit_count()
                    if count_vxy + 1 >= 3:
                        return False

        # Check triples (u, v, x) for x != u, v
        for x in range(n):
            if x == u or x == v:
                continue
            count = (adj[u] & adj[v] & adj[x]).bit_count()
            if count >= 3:
                return False

        return True

    def is_k33_free(adj):
        for i in range(n):
            for j in range(i + 1, n):
                common_ij = adj[i] & adj[j]
                if common_ij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (common_ij & adj[k]).bit_count() >= 3:
                        return False
        return True

    def edge_count(adj):
        return sum(x.bit_count() for x in adj) // 2

    # Greedy construction
    def greedy_build():
        adj = [0] * n
        edges = all_edges[:]
        random.shuffle(edges)
        for u, v in edges:
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    # Local search: add missing edges
    def local_add(adj):
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not ((adj[u] >> v) & 1):
                    if can_add(u, v, adj):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        return adj

    # Local search: swaps
    def local_swap(adj, rounds=30):
        for _ in range(rounds):
            if time.time() > deadline - 5.0:
                break

            existing = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            missing = [(u, v) for u, v in all_edges if not (adj[u] >> v) & 1]

            if not existing or not missing:
                break

            random.shuffle(existing)
            random.shuffle(missing)

            for u1, v1 in existing[:15]:
                for u2, v2 in missing[:15]:
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)

                    if can_add(u2, v2, adj):
                        adj[u2] |= (1 << v2)
                        adj[v2] |= (1 << u2)
                        break
                    else:
                        adj[u1] |= (1 << v1)
                        adj[v1] |= (1 << u1)
                else:
                    continue
                break

        return adj

    # Simulated annealing for swaps
    def simulated_annealing(adj, initial_temp=1.0, cooling=0.99, max_iter=500):
        temp = initial_temp
        current_edges = edge_count(adj)
        best_adj = adj[:]
        best_edges = current_edges

        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            existing = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            missing = [(u, v) for u, v in all_edges if not (adj[u] >> v) & 1]

            if not existing or not missing:
                break

            u1, v1 = random.choice(existing)
            u2, v2 = random.choice(missing)

            # Remove u1-v1
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)

            # Try to add u2-v2
            if can_add(u2, v2, adj):
                adj[u2] |= (1 << v2)
                adj[v2] |= (1 << u2)
                new_edges = current_edges
                # Accept (same edge count, but different structure)
                if new_edges > best_edges:
                    best_edges = new_edges
                    best_adj = adj[:]
            else:
                # Revert
                adj[u1] |= (1 << v1)
                adj[v1] |= (1 << u1)

            # Occasionally try to add a random missing edge
            if random.random() < 0.3:
                u2, v2 = random.choice(missing)
                if can_add(u2, v2, adj):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]

            temp *= cooling

        return best_adj, best_edges

    # Main loop
    for restart in range(300):
        if time.time() > deadline - 10.0:
            break

        # Greedy construction
        adj = greedy_build()

        # Local search: add edges
        adj = local_add(adj)

        # Local search: swaps
        adj = local_swap(adj, rounds=20)

        # Simulated annealing
        adj, sa_edges = simulated_annealing(adj, initial_temp=1.0, cooling=0.995, max_iter=300)

        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]

    if best_adj:
        return [[u, v] for u in range(n) for v in range(u + 1, n) if (best_adj[u] >> v) & 1]
    else:
        return []
