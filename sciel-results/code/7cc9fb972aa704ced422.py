"""Construct a dense C4-rainbow edge-coloured graph on 20 vertices."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    n = 20
    num_colours = 20

    def elapsed():
        return time.time() - start

    def time_ok():
        return elapsed() < deadline - 5.0

    def can_add_edge(adj, u, v, c):
        """Check if adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
        # For each x in N(v), for each y in N(u) \ {v,x}, if x~y then cycle u-v-x-y-u
        nv = np.where(adj[v] != -1)[0]
        nu = np.where(adj[u] != -1)[0]
        for x in nv:
            if x == u:
                continue
            for y in nu:
                if y == v or y == x:
                    continue
                if adj[x, y] == -1:
                    continue
                c_vx = adj[v, x]
                c_xy = adj[x, y]
                c_yu = adj[y, u]
                if len({c, c_vx, c_xy, c_yu}) < 4:
                    return False
        return True

    def can_recolor_edge(adj, u, v, new_c):
        """Check if recoloring existing edge (u,v) to new_c keeps every 4-cycle rainbow."""
        old_c = adj[u, v]
        if old_c == new_c:
            return True
        adj[u, v] = new_c
        adj[v, u] = new_c
        valid = True
        # Check all 4-cycles containing edge (u,v)
        for x in range(n):
            if x == u or x == v:
                continue
            if adj[v, x] == -1:
                continue
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if adj[x, y] == -1:
                    continue
                if adj[y, u] == -1:
                    continue
                c_vx = adj[v, x]
                c_xy = adj[x, y]
                c_yu = adj[y, u]
                if len({new_c, c_vx, c_xy, c_yu}) < 4:
                    valid = False
                    break
            if not valid:
                break
        adj[u, v] = old_c
        adj[v, u] = old_c
        return valid

    def build_greedy(seed):
        """Greedy construction with recoloring to unblock stuck edges."""
        rng = random.Random(seed)
        adj = -np.ones((n, n), dtype=int)
        np.fill_diagonal(adj, -1)

        # Start with a random tree (no 4-cycles, always valid)
        vertices = list(range(n))
        rng.shuffle(vertices)
        for i in range(1, n):
            u = vertices[i]
            v = vertices[rng.randint(0, i - 1)]
            c = rng.randint(0, num_colours - 1)
            adj[u, v] = c
            adj[v, u] = c

        # Try to add all remaining edges
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)

        for (u, v) in all_pairs:
            if not time_ok():
                break
            if adj[u, v] != -1:
                continue

            added = False
            cols = list(range(num_colours))
            rng.shuffle(cols)
            for c in cols:
                if can_add_edge(adj, u, v, c):
                    adj[u, v] = c
                    adj[v, u] = c
                    added = True
                    break

            if not added:
                # Try to recolor an edge in a conflicting cycle to unblock (u,v)
                # Find a conflicting cycle: for all c, some 4-cycle u-v-x-y-u conflicts.
                # Heuristic: try recoloring (v,x) for x in N(v), or (y,u) for y in N(u)
                recolor_done = False
                # Collect candidate edges to recolor: (v,x) for x in N(v), (y,u) for y in N(u)
                candidates = []
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if adj[v, x] != -1:
                        candidates.append((v, x))
                for y in range(n):
                    if y == u or y == v:
                        continue
                    if adj[y, u] != -1:
                        candidates.append((y, u))
                rng.shuffle(candidates)
                for (a, b) in candidates:
                    if not time_ok():
                        break
                    old_ab = adj[a, b]
                    for new_c in range(num_colours):
                        if new_c == old_ab:
                            continue
                        if can_recolor_edge(adj, a, b, new_c):
                            adj[a, b] = new_c
                            adj[b, a] = new_c
                            # Now try to add (u,v) again
                            cols2 = list(range(num_colours))
                            rng.shuffle(cols2)
                            for c2 in cols2:
                                if can_add_edge(adj, u, v, c2):
                                    adj[u, v] = c2
                                    adj[v, u] = c2
                                    added = True
                                    recolor_done = True
                                    break
                            if not added:
                                # Revert the recolor
                                adj[a, b] = old_ab
                                adj[b, a] = old_ab
                            break
                    if recolor_done:
                        break
        return adj

    def local_improve(adj, max_iter=50):
        """Add missing edges (with recoloring) to increase density."""
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if not time_ok():
                break
            rng = random.Random(iter_count + 7)
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            missing = [(u, v) for (u, v) in all_pairs if adj[u, v] == -1]
            rng.shuffle(missing)
            for (u, v) in missing:
                if not time_ok():
                    break
                cols = list(range(num_colours))
                rng.shuffle(cols)
                for c in cols:
                    if can_add_edge(adj, u, v, c):
                        adj[u, v] = c
                        adj[v, u] = c
                        improved = True
                        break
                # If no colour works, try one recolor attempt
                if not improved:
                    candidates = []
                    for x in range(n):
                        if x == u or x == v:
                            continue
                        if adj[v, x] != -1:
                            candidates.append((v, x))
                    for y in range(n):
                        if y == u or y == v:
                            continue
                        if adj[y, u] != -1:
                            candidates.append((y, u))
                    rng.shuffle(candidates)
                    for (a, b) in candidates[:3]:  # limit to 3 attempts for speed
                        if not time_ok():
                            break
                        old_ab = adj[a, b]
                        for new_c in range(num_colours):
                            if new_c == old_ab:
                                continue
                            if can_recolor_edge(adj, a, b, new_c):
                                adj[a, b] = new_c
                                adj[b, a] = new_c
                                for c2 in range(num_colours):
                                    if can_add_edge(adj, u, v, c2):
                                        adj[u, v] = c2
                                        adj[v, u] = c2
                                        improved = True
                                        break
                                if not improved:
                                    adj[a, b] = old_ab
                                    adj[b, a] = old_ab
                                break
        return adj

    def edge_count(adj):
        return int(np.sum(adj != -1) // 2)

    # Try multiple seeds
    best_adj = None
    best_count = -1

    for seed in range(300):
        if not time_ok():
            break
        adj = build_greedy(seed)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj
        if best_count >= 190:
            break

    # Local improvement on best
    if best_adj is not None and time_ok():
        improved = local_improve(best_adj.copy())
        if edge_count(improved) > best_count:
            best_adj = improved
            best_count = edge_count(best_adj)

    # More seeds if time permits
    if time_ok():
        for seed in range(300, 600):
            if not time_ok():
                break
            adj = build_greedy(seed)
            cnt = edge_count(adj)
            if cnt > best_count:
                best_count = cnt
                best_adj = adj
            if best_count >= 190:
                break

    # Final improvement
    if best_adj is not None and time_ok():
        improved = local_improve(best_adj.copy())
        if edge_count(improved) > best_count:
            best_adj = improved
            best_count = edge_count(best_adj)

    if best_adj is None:
        # Fallback: star
        return [[0, v, 0] for v in range(1, n)]

    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] != -1:
                result.append([u, v, int(best_adj[u, v])])
    return result
# EVOLVE-BLOCK-END
