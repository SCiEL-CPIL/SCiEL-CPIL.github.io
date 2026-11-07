import time
import sys
import random
from collections import defaultdict, deque

sys.setrecursionlimit(10000)

P = {
    'n': 21,
    'edges': [
        [0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20],
        [1, 2], [1, 3], [1, 6], [1, 17], [1, 20],
        [2, 3], [2, 4], [2, 7], [2, 18],
        [3, 4], [3, 5], [3, 8], [3, 19],
        [4, 5], [4, 6], [4, 9], [4, 20],
        [5, 6], [5, 7], [5, 10],
        [6, 7], [6, 8], [6, 11],
        [7, 8], [7, 9], [7, 12],
        [8, 9], [8, 10], [8, 13],
        [9, 10], [9, 11], [9, 14],
        [10, 11], [10, 12], [10, 15],
        [11, 12], [11, 13], [11, 16],
        [12, 13], [12, 14], [12, 17],
        [13, 14], [13, 15], [13, 18],
        [14, 15], [14, 16], [14, 19],
        [15, 16], [15, 17], [15, 20],
        [16, 17], [16, 18],
        [17, 18], [17, 19],
        [18, 19], [18, 20],
        [19, 20]
    ]
}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0

    # Build edge set for O(1) lookup
    edge_set = set()
    for u, v in P['edges']:
        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b))

    def valid_hamiltonian_cycle(cyc):
        """Check if cyc is a valid Hamiltonian cycle on all n vertices."""
        if len(cyc) != n:
            return False
        if len(set(cyc)) != n:
            return False
        for i in range(n):
            a, b = cyc[i], cyc[(i + 1) % n]
            key = (a, b) if a < b else (b, a)
            if key not in edge_set:
                return False
        return True

    def cycle_edges(cyc):
        """Return set of edges used by a cycle."""
        edges = set()
        for i in range(len(cyc)):
            a, b = cyc[i], cyc[(i + 1) % len(cyc)]
            key = (a, b) if a < b else (b, a)
            edges.add(key)
        return edges

    # Strategy 1: Try circulant Hamiltonian cycles with various steps
    # For n=21, gcd(k,21)=1 means k in {1,2,4,5,8,10,11,13,16,17,19,20}
    # Try to find 3+ edge-disjoint ones
    candidates = []
    for k in [1, 2, 4, 5, 8, 10, 11, 13, 16, 17, 19, 20]:
        cyc = [0]
        cur = 0
        for _ in range(n - 1):
            cur = (cur + k) % n
            cyc.append(cur)
        if valid_hamiltonian_cycle(cyc):
            candidates.append(cyc)

    # Greedily pick edge-disjoint cycles from candidates
    cycles = []
    used_edges = set()
    for cyc in candidates:
        if time.time() > deadline:
            break
        cyc_e = cycle_edges(cyc)
        if not cyc_e & used_edges:
            cycles.append(cyc)
            used_edges |= cyc_e
            if len(cycles) >= 10:
                break

    # If we got 3+ cycles, we're likely optimal (73 edges / 21 per cycle = 3 max)
    # But let's also try to find more with a different approach if possible

    # Strategy 2: If we have fewer than 3, try randomized DFS to find more
    if len(cycles) < 3:
        # Build adjacency
        adj = defaultdict(set)
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)

        rng = random.Random(42)

        def find_hamiltonian(adj_cur, start, deadline):
            """Find Hamiltonian cycle in adj_cur starting from start."""
            visited = [False] * n
            path = [start]
            visited[start] = True

            def dfs():
                if time.time() > deadline:
                    return False
                if len(path) == n:
                    last = path[-1]
                    first = path[0]
                    return first in adj_cur[last]
                u = path[-1]
                # Warnsdorff heuristic: prefer neighbors with fewer unvisited neighbors
                nbrs = [w for w in adj_cur[u] if not visited[w]]
                if not nbrs:
                    return False
                nbrs.sort(key=lambda w: sum(1 for x in adj_cur[w] if not visited[x]))
                # Add some randomization
                if len(nbrs) > 1:
                    # Try the best few in random order
                    k = min(3, len(nbrs))
                    selected = nbrs[:k]
                    rng.shuffle(selected)
                    for w in selected:
                        visited[w] = True
                        path.append(w)
                        if dfs():
                            return True
                        path.pop()
                        visited[w] = False
                    # If those didn't work, try the rest
                    for w in nbrs[k:]:
                        visited[w] = True
                        path.append(w)
                        if dfs():
                            return True
                        path.pop()
                        visited[w] = False
                else:
                    w = nbrs[0]
                    visited[w] = True
                    path.append(w)
                    if dfs():
                        return True
                    path.pop()
                    visited[w] = False
                return False

            if dfs():
                return list(path)
            return None

        # Try to find additional edge-disjoint cycles
        attempts = 0
        while len(cycles) < 3 and time.time() < deadline:
            attempts += 1
            adj_cur = {i: set() for i in range(n)}
            for u, v in P['edges']:
                if (min(u,v), max(u,v)) not in used_edges:
                    adj_cur[u].add(v)
                    adj_cur[v].add(u)

            # Check if adj_cur still has a Hamiltonian cycle possibility
            # Each vertex must have degree >= 2
            feasible = True
            for v in range(n):
                if len(adj_cur[v]) < 2:
                    feasible = False
                    break

            if not feasible:
                break

            start = rng.randrange(n)
            cyc = find_hamiltonian(adj_cur, start, deadline)
            if cyc is not None and valid_hamiltonian_cycle(cyc):
                cyc_e = cycle_edges(cyc)
                if not cyc_e & used_edges:
                    cycles.append(cyc)
                    used_edges |= cyc_e

    # Ensure we have at least 1 cycle
    if not cycles:
        # Fallback: just return a single valid cycle if one exists
        for k in [1, 2, 5]:
            cyc = [0]
            cur = 0
            for _ in range(n - 1):
                cur = (cur + k) % n
                cyc.append(cur)
            if valid_hamiltonian_cycle(cyc):
                cycles.append(cyc)
                break

    # Final verification and type conversion
    result = []
    for cyc in cycles:
        # Convert to list of Python ints
        result.append([int(v) for v in cyc])

    # Verify edge-disjointness one more time
    if result:
        final_used = set()
        ok = True
        for cyc in result:
            for i in range(len(cyc)):
                a, b = cyc[i], cyc[(i + 1) % len(cyc)]
                key = (a, b) if a < b else (b, a)
                if key in final_used:
                    ok = False
                    break
                final_used.add(key)
            if not ok:
                break

    return result
