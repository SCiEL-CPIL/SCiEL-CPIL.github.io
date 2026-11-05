import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145
    n = 40

    adj = [set() for _ in range(n)]

    # 8 disjoint copies of K5 (no C6, 80 edges)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = 5 * k + a, 5 * k + b
                adj[u].add(v)
                adj[v].add(u)

    def edge_count():
        return sum(len(s) for s in adj) // 2

    def has_path_length_5(u, v):
        """Return True if there is a simple path of length 5 from u to v."""
        visited = [False] * n
        visited[u] = True

        def dfs(node, depth):
            if depth == 5:
                return v in adj[node]
            for nxt in adj[node]:
                if not visited[nxt] and nxt != v:
                    visited[nxt] = True
                    if dfs(nxt, depth + 1):
                        return True
                    visited[nxt] = False
            return False

        return dfs(u, 1)

    best_adj = [set(s) for s in adj]
    best_count = edge_count()

    def save_best():
        nonlocal best_adj, best_count
        c = edge_count()
        if c > best_count:
            best_count = c
            best_adj = [set(s) for s in adj]

    save_best()

    random.seed(42)

    def get_edges():
        return [(u, v) for u in range(n) for v in adj[u] if u < v]

    def greedy_add():
        """Try to add as many edges as possible. Returns number added."""
        added = 0
        for _pass in range(20):
            imp = False
            for u in range(n):
                if time.time() - start_time > DEADLINE - 3:
                    return added
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        if not has_path_length_5(u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            added += 1
                            imp = True
            if not imp:
                break
        return added

    def kick_and_recover():
        """Remove random edges, then greedily re-add. Returns True if improved."""
        edges = get_edges()
        if not edges:
            return False
        saved = [set(s) for s in adj]
        saved_count = edge_count()

        num_remove = random.randint(2, 6)
        to_remove = random.sample(edges, min(num_remove, len(edges)))
        for u, v in to_remove:
            adj[u].discard(v)
            adj[v].discard(u)

        greedy_add()

        if edge_count() > saved_count:
            return True
        else:
            adj[:] = [set(s) for s in saved]
            return False

    # Main loop
    while time.time() - start_time < DEADLINE:
        # Phase 1: greedy add
        greedy_add()
        save_best()

        if time.time() - start_time > DEADLINE - 15:
            break

        # Phase 2: kicks
        for _ in range(100):
            if time.time() - start_time > DEADLINE - 5:
                break
            kick_and_recover()
            save_best()

    return [[u, v] for u in range(n) for v in best_adj[u] if u < v]
