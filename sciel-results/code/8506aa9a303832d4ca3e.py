import time
import random

def construct():
    t_start = time.time()
    deadline = t_start + 75.0  # safety margin before 90s kill

    n = 21
    edges = [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]

    # Build edge set and adjacency
    edge_set = set()
    adj = [set() for _ in range(n)]
    for e in edges:
        a, b = e
        edge_set.add((min(a, b), max(a, b)))
        adj[a].add(b)
        adj[b].add(a)

    def is_connected(available_edges):
        """Check if the graph formed by available_edges is connected and spans all n vertices."""
        if not available_edges:
            return False
        # Build adjacency from available edges
        temp_adj = [set() for _ in range(n)]
        for (u, v) in available_edges:
            temp_adj[u].add(v)
            temp_adj[v].add(u)
        # BFS from vertex 0
        visited = set([0])
        stack = [0]
        while stack:
            u = stack.pop()
            for v in temp_adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        return len(visited) == n

    def find_hamiltonian_cycle(available_edges, start, randomize=False):
        """Backtracking to find a Hamiltonian cycle starting and ending at 'start'."""
        if time.time() > deadline:
            return None
        
        # Quick connectivity check
        if not is_connected(available_edges):
            return None

        path = [start]
        visited = set([start])
        used = set()  # set of (u,v) with u<v for edges used in this cycle

        def backtrack():
            if time.time() > deadline:
                return None
            if len(path) == n:
                # Check if last vertex connects back to start
                last = path[-1]
                key = (min(last, start), max(last, start))
                if key in available_edges and key not in used:
                    return list(path)
                return None

            last = path[-1]
            # Get available neighbors
            neighbors = []
            for v in adj[last]:
                if v not in visited:
                    key = (min(last, v), max(last, v))
                    if key in available_edges and key not in used:
                        neighbors.append(v)

            if not neighbors:
                return None

            # Sort neighbors by degree in remaining graph (Warnsdorff heuristic)
            if randomize:
                random.shuffle(neighbors)
            
            def neighbor_degree(v):
                count = 0
                for u in adj[v]:
                    if u not in visited and u != last:
                        key = (min(u, v), max(u, v))
                        if key in available_edges and key not in used:
                            count += 1
                return count
            
            neighbors.sort(key=neighbor_degree)

            for nb in neighbors:
                key = (min(last, nb), max(last, nb))
                if key not in available_edges:
                    continue
                if key in used:
                    continue
                used.add(key)
                path.append(nb)
                visited.add(nb)
                result = backtrack()
                if result is not None:
                    return result
                path.pop()
                visited.discard(nb)
                used.discard(key)
            return None

        return backtrack()

    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    cycles = []
    available = set(edge_set)

    # Strategy 1: Deterministic with multiple start orderings
    start_orderings = [
        list(range(n)),
        list(range(n - 1, -1, -1)),
        [0, 20, 1, 19, 2, 18, 3, 17, 4, 16, 5, 15, 6, 14, 7, 13, 8, 12, 9, 11, 10],
        [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    ]

    for ordering in start_orderings:
        if len(cycles) >= 3:
            break
        if time.time() > deadline:
            break
        for start in ordering:
            if len(cycles) >= 3:
                break
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(available, start, randomize=False)
            if cycle is not None:
                cycles.append(cycle)
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i + 1) % len(cycle)]
                    key = (min(a, b), max(a, b))
                    available.discard(key)

    # Strategy 2: Randomized restarts if we haven't found 3
    if len(cycles) < 3 and time.time() < deadline:
        # Reset and try with randomization
        cycles = []
        available = set(edge_set)
        
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            if len(cycles) >= 3:
                break
            
            # Try to find next cycle with randomization
            found = False
            for _ in range(5):  # Try a few times per restart
                if time.time() > deadline:
                    break
                if len(cycles) >= 3:
                    break
                if not is_connected(available):
                    break
                start = random.randint(0, n - 1)
                cycle = find_hamiltonian_cycle(available, start, randomize=True)
                if cycle is not None:
                    cycles.append(cycle)
                    for i in range(len(cycle)):
                        a = cycle[i]
                        b = cycle[(i + 1) % len(cycle)]
                        key = (min(a, b), max(a, b))
                        available.discard(key)
                    found = True
                    break
            
            if not found and len(cycles) < 3:
                # If we got stuck, reset and try fresh
                if len(cycles) > 0:
                    # Keep what we have and try to add more
                    pass
                else:
                    # Reset completely
                    cycles = []
                    available = set(edge_set)

    # Ensure we have at least 1 cycle
    if len(cycles) == 0:
        available = set(edge_set)
        for s in range(n):
            if time.time() > deadline:
                break
            cycle = find_hamiltonian_cycle(available, s, randomize=False)
            if cycle is not None:
                cycles = [cycle]
                break

    # Ensure we return at most 10 cycles
    cycles = cycles[:10]

    # Convert to list of lists of Python ints
    result = []
    for c in cycles:
        result.append([int(v) for v in c])

    return result
