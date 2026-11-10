"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = P['n']
    edges = P['edges']

    # Build adjacency and edge set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def get_cycle_edges(cycle):
        es = set()
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            es.add((min(u, v), max(u, v)))
        return es

    def validate_cycle(cycle, used_edges):
        """Validate that cycle is a valid Hamiltonian cycle with no edge conflicts."""
        if len(cycle) != n:
            return False
        if len(set(cycle)) != n:
            return False
        ce = get_cycle_edges(cycle)
        if ce & used_edges:
            return False
        for e in ce:
            if e not in edge_set:
                return False
        return True

    def find_ham_cycle_fast(used_edges, start=0):
        """Find a Hamiltonian cycle using optimized backtracking."""
        # Build remaining adjacency as bitmasks for speed
        rem_adj = [0] * n
        for u in range(n):
            mask = 0
            for v in adj[u]:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    mask |= (1 << v)
            rem_adj[u] = mask

        # Check minimum degree >= 2
        for u in range(n):
            if bin(rem_adj[u]).count('1') < 2:
                return None

        # Check connectivity using BFS with bitmasks
        visited_mask = 1 << start
        stack = [start]
        while stack:
            node = stack.pop()
            nbrs = rem_adj[node] & ~visited_mask
            while nbrs:
                lsb = nbrs & (-nbrs)
                nb = lsb.bit_length() - 1
                visited_mask |= lsb
                stack.append(nb)
                nbrs &= nbrs - 1
        if visited_mask != (1 << n) - 1:
            return None

        # Backtracking with MRV
        path = [start]
        visited_mask = 1 << start
        edges_used_local = set()

        def dfs(current):
            if time.time() > deadline - 2:
                return None

            if len(path) == n:
                last = path[-1]
                e = (min(last, start), max(last, start))
                if e not in edges_used_local and e in edge_set:
                    return list(path)
                return None

            # Get unvisited neighbors sorted by degree (MRV)
            nbrs = rem_adj[current] & ~visited_mask
            candidates = []
            while nbrs:
                lsb = nbrs & (-nbrs)
                nb = lsb.bit_length() - 1
                e = (min(current, nb), max(current, nb))
                if e not in edges_used_local:
                    deg = bin(rem_adj[nb]).count('1')
                    candidates.append((deg, nb))
                nbrs &= nbrs - 1

            candidates.sort()

            for _, nb in candidates:
                e = (min(current, nb), max(current, nb))
                if e in edges_used_local:
                    continue

                new_visited = visited_mask | (1 << nb)
                new_edges = edges_used_local | {e}

                # Pruning: unvisited vertices must still have degree >= 2
                feasible = True
                remaining = (1 << n) - 1 & ~new_visited
                while remaining:
                    lsb = remaining & (-remaining)
                    v = lsb.bit_length() - 1
                    # Count available edges from v to unvisited vertices
                    avail_mask = rem_adj[v] & ~new_visited
                    if bin(avail_mask).count('1') < 1:  # Need at least 1 edge to unvisited + 1 back
                        # Actually need degree >= 2 in remaining graph
                        # v can connect to visited (via closing) and unvisited
                        # For now, just check it has at least 1 unvisited neighbor
                        feasible = False
                        break
                    remaining &= remaining - 1

                if not feasible:
                    continue

                path.append(nb)
                edges_used_local.add(e)
                result = dfs(nb)
                if result is not None:
                    return result
                path.pop()
                edges_used_local.discard(e)

            return None

        return dfs(start)

    # Precomputed known good cycles based on graph structure
    # The graph has a clear pattern - try structured cycles first
    def try_structured_cycles():
        """Try to find cycles based on the graph's structural patterns."""
        used_edges = set()
        cycles = []

        # Try cycles with different step patterns
        # Pattern 1: sequential
        for offset in range(7):
            if time.time() > deadline - 20:
                break
            cycle = []
            valid = True
            for i in range(n):
                v = (i + offset) % n
                if i > 0:
                    prev = (i - 1 + offset) % n
                    e = (min(prev, v), max(prev, v))
                    if e not in edge_set or e in used_edges:
                        valid = False
                        break
                cycle.append(v)
            if valid:
                e = (min(cycle[-1], cycle[0]), max(cycle[-1], cycle[0]))
                if e in edge_set and e not in used_edges:
                    ce = get_cycle_edges(cycle)
                    if not (ce & used_edges):
                        used_edges |= ce
                        cycles.append(cycle)

        return cycles

    # Main search
    best_cycles = []

    # First try structured approach
    structured = try_structured_cycles()
    if len(structured) > len(best_cycles):
        best_cycles = list(structured)

    # Then use backtracking to fill in
    if len(best_cycles) < 3:
        used_edges = set()
        for c in best_cycles:
            used_edges |= get_cycle_edges(c)

        for start_v in range(n):
            if time.time() > deadline - 10:
                break
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                cycle = find_ham_cycle_fast(used_edges, start=start_v)
                if cycle is None:
                    break
                ce = get_cycle_edges(cycle)
                if ce & used_edges:
                    break
                used_edges |= ce
                best_cycles.append(cycle)
                if len(best_cycles) >= 3:
                    break
            if len(best_cycles) >= 3:
                break

    # Verify all cycles are valid
    if best_cycles:
        final_used = set()
        valid_cycles = []
        for c in best_cycles:
            if not validate_cycle(c, final_used):
                break
            ce = get_cycle_edges(c)
            final_used |= ce
            valid_cycles.append(c)
        best_cycles = valid_cycles

    if best_cycles:
        result = []
        for c in best_cycles:
            result.append([int(v) for v in c])
        return result

    # Fallback: find any valid cycle
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        return [[int(u), int(v), int(w)]]

    return [[0, 1, 2]]

# EVOLVE-BLOCK-END
