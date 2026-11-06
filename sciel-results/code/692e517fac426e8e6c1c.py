"""Construct a K4-free graph with alpha <= 5, maximizing n within time limit."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Start with Paley graph on 17 vertices (known valid)
    n = 17
    squares = {x * x % 17 for x in range(1, 17)}
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % 17 in squares:
                adj[i].add(j)
                adj[j].add(i)

    best_n = n
    best_adj = [set(s) for s in adj]

    # Verify initial graph is valid
    def has_k4(adj, n):
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    continue
                for k in range(j + 1, n):
                    if k not in adj[i] or k not in adj[j]:
                        continue
                    for l in range(k + 1, n):
                        if l not in adj[i] or l not in adj[j] or l not in adj[k]:
                            continue
                        return True
        return False

    def has_independent_6(adj, n):
        # Check for independent set of size 6 using backtracking
        def backtrack(remaining, current):
            if len(current) == 6:
                return True
            if len(current) + len(remaining) < 6:
                return False
            # Pick vertex with max degree in remaining for better pruning
            v = max(remaining, key=lambda x: len(adj[x] & remaining))
            # Include v
            new_remaining = remaining - {v} - adj[v]
            if backtrack(new_remaining, current + [v]):
                return True
            # Exclude v
            if backtrack(remaining - {v}, current):
                return True
            return False
        return backtrack(set(range(n)), [])

    def is_valid(adj, n):
        if has_k4(adj, n):
            return False
        if has_independent_6(adj, n):
            return False
        return True

    if not is_valid(adj, n):
        # Fallback to a small valid graph
        n = 8
        adj = [set() for _ in range(n)]
        for i in range(4):
            for j in range(4, 8):
                adj[i].add(j)
                adj[j].add(i)
        best_n = n
        best_adj = [set(s) for s in adj]

    # Try to grow the graph by adding vertices
    while time.time() < deadline:
        new_n = n + 1
        if new_n > 40:
            break

        success = False
        for attempt in range(300):
            if time.time() > deadline:
                break

            new_adj = [set(s) for s in adj]
            new_adj.append(set())

            # Randomly decide edges from new vertex to existing vertices
            candidate_edges = set()
            for i in range(n):
                if random.random() < 0.5:
                    candidate_edges.add(i)

            for i in candidate_edges:
                new_adj[i].add(new_n - 1)
                new_adj[new_n - 1].add(i)

            if is_valid(new_adj, new_n):
                adj = new_adj
                n = new_n
                success = True
                best_n = n
                best_adj = [set(s) for s in adj]
                break

        if not success:
            # Try local search: modify some edges in current graph to enable growth
            improved = False
            for _ in range(100):
                if time.time() > deadline:
                    break

                if n < 2:
                    break
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue

                had_edge = v in adj[u]
                if had_edge:
                    adj[u].discard(v)
                    adj[v].discard(u)
                else:
                    adj[u].add(v)
                    adj[v].add(u)

                if is_valid(adj, n):
                    # Try to add a vertex again
                    new_n = n + 1
                    if new_n <= 40:
                        for attempt2 in range(100):
                            if time.time() > deadline:
                                break
                            trial_adj = [set(s) for s in adj]
                            trial_adj.append(set())
                            for i in range(n):
                                if random.random() < 0.5:
                                    trial_adj[i].add(new_n - 1)
                                    trial_adj[new_n - 1].add(i)
                            if is_valid(trial_adj, new_n):
                                adj = trial_adj
                                n = new_n
                                best_n = n
                                best_adj = [set(s) for s in adj]
                                improved = True
                                break
                    if improved:
                        break
                else:
                    # Revert
                    if had_edge:
                        adj[u].add(v)
                        adj[v].add(u)
                    else:
                        adj[u].discard(v)
                        adj[v].discard(u)

            if improved:
                continue
            else:
                # Revert to best known
                adj = [set(s) for s in best_adj]
                n = best_n
                if time.time() > deadline - 30:
                    break

    # Return best result
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
