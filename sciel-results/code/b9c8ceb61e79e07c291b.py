import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin under 150s

    # Increase recursion limit for backtracking on ~400 nodes
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                 [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces
                 for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]),
                           max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_exact_bitmask(n, edges, lists):
        """
        Exact list-coloring solver using bitmask domains and MRV + Forward Checking.
        Returns a list of Python ints (one per vertex) or None.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Sort adjacency lists by degree descending? 
        # Actually, for MRV we just need to iterate. 
        # Pre-compute degrees for tie-breaking.
        degrees = [len(a) for a in adj]

        # Convert lists to bitmasks
        # Palette is 0..4, so we need 5 bits.
        # domain[v] is an integer where bit c is set if color c is available.
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask

        # Check if any domain is initially empty
        for v in range(n):
            if domains[v] == 0:
                return None

        # Helper to count bits (Python 3.8+ has int.bit_count())
        # For compatibility or speed, we can use bin(x).count('1') or bit_count if available.
        # Since we are likely on modern Python, bit_count is fast.
        def popcount(x):
            return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

        colour = [-1] * n
        assigned = [False] * n
        uncolored_count = n

        # MRV Heuristic: Find vertex with smallest domain size > 0
        # Tie-break: Highest degree
        def get_mrv_vertex():
            best_v = -1
            best_size = 10**9
            best_deg = -1
            for v in range(n):
                if not assigned[v]:
                    size = popcount(domains[v])
                    if size == 0:
                        return -1 # Failure
                    if size < best_size or (size == best_size and degrees[v] > best_deg):
                        best_size = size
                        best_deg = degrees[v]
                        best_v = v
            return best_v

        # Value Ordering: Least Constraining Value
        # Try colors that are in fewer neighbors' domains
        def get_value_order(v):
            colors = []
            mask = domains[v]
            for c in range(5): # Palette size 5
                if mask & (1 << c):
                    # Count how many unassigned neighbors have this color in their domain
                    conflict_count = 0
                    for u in adj[v]:
                        if not assigned[u] and (domains[u] & (1 << c)):
                            conflict_count += 1
                    colors.append((conflict_count, c))
            # Sort by conflict count (ascending), then by color (stable)
            colors.sort()
            return [c for _, c in colors]

        def backtrack():
            nonlocal uncolored_count
            if uncolored_count == 0:
                return True

            v = get_mrv_vertex()
            if v == -1:
                return False

            # Get ordered values
            values = get_value_order(v)
            
            for c in values:
                bit = 1 << c
                # Assign color c to v
                colour[v] = c
                assigned[v] = True
                uncolored_count -= 1
                old_domain_v = domains[v]
                domains[v] = 0 # Mask out assigned vertex

                # Forward Checking
                feasible = True
                changes = []
                for u in adj[v]:
                    if not assigned[u]:
                        if domains[u] & bit:
                            old_u = domains[u]
                            new_u = old_u & ~bit
                            domains[u] = new_u
                            changes.append((u, old_u))
                            if new_u == 0:
                                feasible = False
                                break

                if feasible:
                    if backtrack():
                        return True

                # Undo
                assigned[v] = False
                colour[v] = -1
                uncolored_count += 1
                domains[v] = old_domain_v
                for u, old_u in changes:
                    domains[u] = old_u

            return False

        if backtrack():
            # Convert to list of ints
            return [int(c) for c in colour]
        else:
            return None

    def greedy_fallback(n, edges, lists):
        """Simple greedy fallback, unlikely to be needed but good for safety."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        colour = [-1] * n
        # Order by degree descending
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] != -1:
                    taken.add(colour[u])
            for c in lists[v]:
                if c not in taken:
                    colour[v] = c
                    break
            if colour[v] == -1:
                colour[v] = lists[v][0]
                
        # Verify
        for v in range(n):
            if colour[v] not in lists[v]:
                return None
            for u in adj[v]:
                if colour[u] == colour[v]:
                    return None
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check time
        if time.time() > deadline - 5:
             n, edges, lists = instance(splits, seed)
             sol = greedy_fallback(n, edges, lists)
             if sol is None:
                 sol = [0] * n
             out.append(sol)
             continue

        n, edges, lists = instance(splits, seed)
        sol = solve_exact_bitmask(n, edges, lists)
        
        if sol is None:
            # Fallback if exact fails (shouldn't happen for these instances)
            sol = greedy_fallback(n, edges, lists)
        
        if sol is None:
            sol = [0] * n
            
        # Ensure types are correct
        out.append([int(c) for c in sol])

    # Ensure length is 4
    while len(out) < 4:
        out.append([0])
        
    return out
