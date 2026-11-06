import time
import random

def construct():
    n = 20
    deadline = time.time() + 110

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def validate_and_add(adj, u, v):
        """Check if edge (u,v) can be added given current adj. Returns True if valid."""
        # Condition 1: u and v must have at most 2 common neighbors
        if (adj[u] & adj[v]).bit_count() > 2:
            return False

        # Condition 2: For each neighbor b of v, u and b must have at most 1 common neighbor (excluding v which is added)
        mask = adj[v]
        while mask:
            b = (mask & -mask).bit_length() - 1
            if (adj[u] & adj[b]).bit_count() > 1:
                return False
            mask &= mask - 1

        # Condition 3: For each neighbor a of u, v and a must have at most 1 common neighbor (excluding u which is added)
        mask = adj[u]
        while mask:
            a = (mask & -mask).bit_length() - 1
            if (adj[v] & adj[a]).bit_count() > 1:
                return False
            mask &= mask - 1

        return True

    def remove_edge_from_adj(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def add_edge_to_adj(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def local_search(adj, current_edges, max_iter=2000):
        """Aggressive local search with multiple swap strategies."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if time.time() > deadline:
                break

            # 1. Try adding any missing edge
            added_any = False
            for u in range(n):
                if added_any:
                    break
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    if validate_and_add(adj, u, v):
                        add_edge_to_adj(adj, u, v)
                        current_edges.append((u, v))
                        added_any = True
                        break

            if added_any:
                improved = True
                continue

            # 2. Try 1-for-2 swap: remove one edge, try to add two
            swapped_any = False
            for i in range(len(current_edges)):
                u, v = current_edges[i]
                old_adj = adj[:]
                old_edges = current_edges[:]
                remove_edge_from_adj(adj, u, v)
                current_edges.pop(i)

                added = 0
                # Try to add edges incident to u
                for w in range(n):
                    if added == 2:
                        break
                    if w == u or (adj[u] >> w) & 1:
                        continue
                    if validate_and_add(adj, u, w):
                        add_edge_to_adj(adj, u, w)
                        current_edges.append((u, w))
                        added += 1

                # Try to add edges incident to v
                if added < 2:
                    for w in range(n):
                        if added == 2:
                            break
                        if w == v or (adj[v] >> w) & 1:
                            continue
                        if validate_and_add(adj, v, w):
                            add_edge_to_adj(adj, v, w)
                            current_edges.append((v, w))
                            added += 1

                # Try to add other edges
                if added < 2:
                    for a in range(n):
                        if added == 2:
                            break
                        for b in range(a + 1, n):
                            if added == 2:
                                break
                            if (adj[a] >> b) & 1:
                                continue
                            # Skip if this involves u or v in a way that conflicts
                            if a == u or b == u or a == v or b == v:
                                continue
                            if validate_and_add(adj, a, b):
                                add_edge_to_adj(adj, a, b)
                                current_edges.append((a, b))
                                added += 1

                if added < 2:
                    # Revert
                    adj[:] = old_adj
                    current_edges[:] = old_edges
                else:
                    swapped_any = True
                    break

            if swapped_any:
                improved = True
                continue

            # 3. Try 2-for-3 swap (limited scope for speed)
            if time.time() < deadline - 5:
                swapped23 = False
                for i in range(len(current_edges)):
                    if swapped23:
                        break
                    u1, v1 = current_edges[i]
                    for j in range(i + 1, min(i + 15, len(current_edges))):
                        u2, v2 = current_edges[j]

                        old_adj = adj[:]
                        old_edges = current_edges[:]

                        # Remove both edges
                        remove_edge_from_adj(adj, u1, v1)
                        remove_edge_from_adj(adj, u2, v2)
                        # Remove from list (handle order)
                        if j > i:
                            current_edges.pop(j)
                            current_edges.pop(i)
                        else:
                            current_edges.pop(i)
                            current_edges.pop(j)

                        # Try to add up to 3 edges
                        added = 0
                        # Priority: edges incident to the freed vertices
                        candidates = []
                        for w in range(n):
                            if w != u1 and (adj[u1] >> w) & 1 == 0:
                                candidates.append((u1, w))
                            if w != v1 and (adj[v1] >> w) & 1 == 0:
                                candidates.append((v1, w))
                            if w != u2 and (adj[u2] >> w) & 1 == 0:
                                candidates.append((u2, w))
                            if w != v2 and (adj[v2] >> w) & 1 == 0:
                                candidates.append((v2, w))

                        # Deduplicate and try
                        tried = set()
                        for a, b in candidates:
                            if added == 3:
                                break
                            if (a, b) in tried or (b, a) in tried:
                                continue
                            tried.add((a, b))
                            if (adj[a] >> b) & 1:
                                continue
                            if validate_and_add(adj, a, b):
                                add_edge_to_adj(adj, a, b)
                                current_edges.append((a, b))
                                added += 1

                        # Try other edges if needed
                        if added < 3:
                            for a in range(n):
                                if added == 3:
                                    break
                                for b in range(a + 1, n):
                                    if added == 3:
                                        break
                                    if (adj[a] >> b) & 1:
                                        continue
                                    if (a, b) in tried or (b, a) in tried:
                                        continue
                                    tried.add((a, b))
                                    if validate_and_add(adj, a, b):
                                        add_edge_to_adj(adj, a, b)
                                        current_edges.append((a, b))
                                        added += 1

                        if added < 3:
                            # Revert
                            adj[:] = old_adj
                            current_edges[:] = old_edges
                        else:
                            swapped23 = True
                            break

                if swapped23:
                    improved = True
                    continue

        return current_edges

    def final_polish(adj, current_edges):
        """Final high-precision polish pass."""
        if time.time() > deadline - 2:
            return current_edges

        # Multiple rounds of local search
        for _ in range(3):
            if time.time() > deadline - 1:
                break
            current_edges = local_search(adj, current_edges, max_iter=3000)

        # Final scan for any addable edges
        for _ in range(50):
            if time.time() > deadline - 1:
                break
            added_any = False
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    if validate_and_add(adj, u, v):
                        add_edge_to_adj(adj, u, v)
                        current_edges.append((u, v))
                        added_any = True
                        break
                if added_any:
                    break
            if not added_any:
                break

        return current_edges

    best_edges = []
    best_count = 0

    # Multiple restarts with different strategies
    while time.time() < deadline - 15:
        edges_copy = all_edges[:]
        random.shuffle(edges_copy)

        adj = [0] * n
        current_edges = []

        for u, v in edges_copy:
            if validate_and_add(adj, u, v):
                add_edge_to_adj(adj, u, v)
                current_edges.append((u, v))

        current_edges = local_search(adj, current_edges, max_iter=1000)

        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = current_edges[:]

            # Intensive polish on best found
            if time.time() < deadline - 10:
                adj2 = [0] * n
                for u, v in best_edges:
                    add_edge_to_adj(adj2, u, v)
                best_edges = final_polish(adj2, best_edges)
                if len(best_edges) > best_count:
                    best_count = len(best_edges)

    # Final polish on the best solution
    if best_edges and time.time() < deadline - 2:
        adj_final = [0] * n
        for u, v in best_edges:
            add_edge_to_adj(adj_final, u, v)
        best_edges = final_polish(adj_final, best_edges)
        best_count = len(best_edges)

    return [[u, v] for u, v in best_edges]
