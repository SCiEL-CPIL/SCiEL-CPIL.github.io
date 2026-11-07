import random
import time

def construct():
    start_time = time.time()
    deadline = 80.0  # seconds, safety margin before 90s kill

    n = 15
    all_possible = set((i, j) for i in range(n) for j in range(i+1, n))

    def is_valid(edges_set):
        """
        Check if the graph represented by edges_set is valid.
        Violation: Simple path s-u-v-t where d(s) == d(t).
        """
        adj = [set() for _ in range(n)]
        degrees = [0] * n

        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1

        # Check all pairs s, t at distance 3
        for s in range(n):
            for t in range(n):
                if s == t:
                    continue
                if degrees[s] != degrees[t]:
                    continue

                # Check if path s-u-v-t exists
                found_path = False
                for u in adj[s]:
                    if u == t:
                        continue
                    for v in adj[u]:
                        if v == s or v == t or v == u:
                            continue
                        if t in adj[v]:
                            found_path = True
                            break
                    if found_path:
                        break
                if found_path:
                    return False
        return True

    def get_violating_pair(edges_set):
        """Return (s, t) if a violation exists, else None."""
        adj = [set() for _ in range(n)]
        degrees = [0] * n

        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
            degrees[u] += 1
            degrees[v] += 1

        for s in range(n):
            for t in range(n):
                if s == t:
                    continue
                if degrees[s] != degrees[t]:
                    continue

                for u in adj[s]:
                    if u == t:
                        continue
                    for v in adj[u]:
                        if v == s or v == t or v == u:
                            continue
                        if t in adj[v]:
                            return (s, t)
        return None

    def local_search(initial_edges):
        edges = set(tuple(sorted(e)) for e in initial_edges)

        # Ensure validity first
        if not is_valid(edges):
            attempts = 0
            while not is_valid(edges) and attempts < 2000:
                e_to_remove = random.choice(list(edges))
                edges.remove(e_to_remove)
                attempts += 1
            if not is_valid(edges):
                return None, 0

        current_score = len(edges)

        # Phase 1: Greedy addition
        improved = True
        while improved:
            improved = False
            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)

            for e in missing:
                new_edges = edges | {e}
                if is_valid(new_edges):
                    edges = new_edges
                    current_score = len(edges)
                    improved = True
                    break

        # Phase 2: Swap moves to escape local optima
        swap_improved = True
        max_swap_iterations = 100
        swap_iter = 0
        while swap_improved and swap_iter < max_swap_iterations:
            swap_improved = False
            swap_iter += 1

            if time.time() - start_time > deadline:
                return list(edges), current_score

            missing = [e for e in all_possible if e not in edges]
            random.shuffle(missing)

            existing = list(edges)
            random.shuffle(existing)

            for e_remove in existing:
                if time.time() - start_time > deadline:
                    return list(edges), current_score
                edges_without = edges - {e_remove}

                for e_add in missing:
                    if e_add == e_remove:
                        continue
                    new_edges = edges_without | {e_add}
                    if is_valid(new_edges):
                        edges = new_edges
                        current_score = len(edges)
                        swap_improved = True
                        break
                if swap_improved:
                    break

            # After a swap, try to add more edges greedily
            if swap_improved:
                add_improved = True
                while add_improved:
                    add_improved = False
                    missing = [e for e in all_possible if e not in edges]
                    random.shuffle(missing)
                    for e in missing:
                        new_edges = edges | {e}
                        if is_valid(new_edges):
                            edges = new_edges
                            current_score = len(edges)
                            add_improved = True
                            break

        return list(edges), current_score

    best_edges = None
    best_score = -1

    # Run several iterations
    num_restarts = 60
    for i in range(num_restarts):
        if time.time() - start_time > deadline:
            break

        all_edges = list(all_possible)

        # Try different densities, biased toward higher densities
        if i < num_restarts // 3:
            density = 0.7 + 0.3 * random.random()
        elif i < 2 * num_restarts // 3:
            density = 0.5 + 0.5 * random.random()
        else:
            density = 0.3 + 0.7 * random.random()

        num_initial = int(len(all_edges) * density)
        initial_edges = random.sample(all_edges, num_initial)

        res_edges, score = local_search(initial_edges)

        if res_edges is not None and score > best_score:
            best_score = score
            best_edges = res_edges

        if best_score > 95:
            break

    if best_edges is None:
        return []

    result = [[int(u), int(v)] for u, v in best_edges]
    return result
