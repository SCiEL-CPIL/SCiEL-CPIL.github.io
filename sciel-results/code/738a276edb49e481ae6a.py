import time
import random

def construct():
    n = 15
    deadline = time.time() + 82  # Safety margin under 90s limit

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_idx = {}
    for idx, (i, j) in enumerate(all_pairs):
        pair_idx[(i, j)] = idx

    # Start with K_{7,8}
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))

    def build_state(edge_set):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (a, b) in edge_set:
            deg[a] += 1
            deg[b] += 1
            adj[a].add(b)
            adj[b].add(a)
        # V[s][t] = number of common neighbors of s and t
        V = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in adj[a]:
                if b > a:
                    for x in adj[a]:
                        for y in adj[b]:
                            if x != y and x != b and y != a:
                                V[x][y] += 1
                                V[y][x] += 1
        return deg, adj, V

    def is_valid_state(deg, V):
        for s in range(n):
            for t in range(s + 1, n):
                if deg[s] == deg[t] and V[s][t] >= 1:
                    return False
        return True

    def is_valid(edge_set):
        deg, adj, V = build_state(edge_set)
        return is_valid_state(deg, V)

    # Verify initial
    if not is_valid(edges):
        # Should not happen; fallback to empty
        edges = set()

    best_edges = set(edges)
    best_count = len(edges)

    def update_best(edge_set):
        nonlocal best_edges, best_count
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges = set(edge_set)

    def greedy_add(edge_set, deg, V, adj, time_limit):
        """Try to add edges greedily; prefer edges that connect vertices with
        different degrees (avoids creating new equal-degree pairs)."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            missing = [(i, j) for (i, j) in all_pairs if (i, j) not in edge_set]
            # Heuristic: prefer edges (i,j) where deg[i] != deg[j] (safer),
            # and among those, higher degree sum (denser).
            def key(e):
                i, j = e
                safer = 1 if deg[i] != deg[j] else 0
                return (safer, deg[i] + deg[j])
            missing.sort(key=key, reverse=True)
            for e in missing:
                if time.time() > time_limit:
                    break
                i, j = e
                new_set = edge_set | {e}
                new_deg = deg[:]
                new_deg[i] += 1
                new_deg[j] += 1
                # Update V: for each x in adj[i] (x!=j), y in adj[j] (y!=i): V[x][y]+=1
                new_V = [row[:] for row in V]
                for x in adj[i]:
                    if x == j:
                        continue
                    for y in adj[j]:
                        if y == i:
                            continue
                        if x != y:
                            new_V[x][y] += 1
                            new_V[y][x] += 1
                if is_valid_state(new_deg, new_V):
                    edge_set = new_set
                    deg = new_deg
                    V = new_V
                    adj[i].add(j)
                    adj[j].add(i)
                    improved = True
                    break
        return edge_set, deg, V, adj

    def local_search(edge_set, deg, V, adj, time_limit):
        """Random swaps (remove one, add one) to escape local optima, then greedy."""
        while time.time() < time_limit:
            # Try a few random swaps
            made_swap = False
            for _ in range(30):
                if time.time() > time_limit:
                    break
                if not edge_set:
                    break
                remove_e = random.choice(list(edge_set))
                missing = [(i, j) for (i, j) in all_pairs if (i, j) not in edge_set]
                if not missing:
                    break
                add_e = random.choice(missing)
                new_set = (edge_set - {remove_e}) | {add_e}
                new_deg, new_adj, new_V = build_state(new_set)
                if is_valid_state(new_deg, new_V):
                    edge_set = new_set
                    deg = new_deg
                    V = new_V
                    adj = new_adj
                    made_swap = True
                    break
            # After swap (or even without), try greedy add
            if time.time() < time_limit - 0.5:
                edge_set, deg, V, adj = greedy_add(edge_set, deg, V, adj, time_limit)
            update_best(edge_set)
        return edge_set, deg, V, adj

    # Phase 1: Greedy add
    deg, adj, V = build_state(edges)
    edges, deg, V, adj = greedy_add(edges, deg, V, adj, deadline - 25)
    update_best(edges)

    # Phase 2: Local search with random restarts on the current best
    random.seed(12345)
    while time.time() < deadline - 5:
        # Work from current best
        cur = set(best_edges)
        cur_deg, cur_adj, cur_V = build_state(cur)
        cur, cur_deg, cur_V, cur_adj = local_search(cur, cur_deg, cur_V, cur_adj, deadline - 5)
        update_best(cur)

    # Final greedy polish
    if time.time() < deadline - 2:
        final_deg, final_adj, final_V = build_state(best_edges)
        best_edges, _, _, _ = greedy_add(best_edges, final_deg, final_V, final_adj, deadline - 2)
        update_best(best_edges)

    # Convert to list of lists
    result = []
    for (u, v) in best_edges:
        result.append([int(u), int(v)])

    return result
