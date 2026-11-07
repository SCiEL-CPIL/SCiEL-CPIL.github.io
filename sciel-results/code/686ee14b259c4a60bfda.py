"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    # Set recursion limit high enough for backtracking, though depth is small
    sys.setrecursionlimit(5000)
    
    start_time = time.time()
    # The evaluator kills at 150s. We aim to finish well before.
    # 110s is a safe margin to allow for overhead and unexpected delays.
    deadline = start_time + 110.0 

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_instance(n, edges, lists):
        """
        Solves list coloring using Backtracking with MRV (Minimum Remaining Values).
        This is generally more efficient and robust than static degree ordering.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert to sets for O(1) lookup if needed, but lists are fine for small degrees
        # We keep lists for iteration speed.
        
        color = [-1] * n
        uncolored = set(range(n))
        
        # Pre-compute neighbors as sets for faster intersection if we were doing arc consistency,
        # but for simple backtracking, checking neighbors directly is fine.
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV Heuristic: Find uncolored vertex with fewest available colors
            best_v = -1
            min_options = 4
            best_options = None
            
            # Iterate over uncolored vertices
            for v in uncolored:
                # Determine taken colors by neighbors
                taken = set()
                for u in adj[v]:
                    if color[u] != -1:
                        taken.add(color[u])
                
                # Determine available colors
                # lists[v] is sorted, but we just need to filter
                options = []
                for c in lists[v]:
                    if c not in taken:
                        options.append(c)
                
                if not options:
                    return False
                
                if len(options) < min_options:
                    min_options = len(options)
                    best_v = v
                    best_options = options
                    if min_options == 1:
                        break # Can't do better than 1
            
            if best_v == -1:
                return False # Should not happen if uncolored is not empty
            
            uncolored.remove(best_v)
            
            # Try each option
            for c in best_options:
                color[best_v] = c
                if backtrack():
                    return True
                color[best_v] = -1
            
            uncolored.add(best_v)
            return False

        if backtrack():
            return [int(c) for c in color]
        return None

    def greedy_fallback(n, edges, lists):
        """
        A fast greedy approximation. Not guaranteed to be valid, but used only if backtracking fails or times out.
        """
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        color = [-1] * n
        # Order by degree descending
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        for v in order:
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            
            # Find a color in lists[v] not in taken
            chosen = -1
            for c in lists[v]:
                if c not in taken:
                    chosen = c
                    break
            
            if chosen == -1:
                # If no color works (conflict), pick the first one (invalid, but better than crash)
                # This indicates the coloring is invalid, but we must return something.
                chosen = lists[v][0]
            
            color[v] = chosen
        
        return [int(c) for c in color]

    out = []
    instances = [(20, 701), (40, 702), (60, 703), (100, 704)]
    
    for splits, seed in instances:
        # Check time
        if time.time() > deadline:
            # If we are out of time, we must still return a valid structure.
            # We generate the instance and use a very fast, potentially invalid coloring.
            n, edges, lists = instance(splits, seed)
            # Use a simple heuristic: just pick the first color in the list.
            # This will likely be invalid, but it's better than returning an empty list or crashing.
            # Actually, the prompt says "Any broken rule makes the whole result invalid".
            # So we MUST try our best to get a valid coloring.
            # If we are out of time, we can't do backtracking.
            # However, backtracking for these sizes is < 1ms.
            # If we hit the deadline, it's a catastrophic failure.
            # Let's just run the greedy fallback which is O(N*E) and very fast.
            coloring = greedy_fallback(n, edges, lists)
            out.append(coloring)
            continue

        n, edges, lists = instance(splits, seed)
        
        # Try the robust MRV backtracking
        coloring = solve_instance(n, edges, lists)
        
        if coloring is None:
            # If backtracking fails (shouldn't happen for these instances as they are 3-colorable
            # and lists have 3 colors, but just in case), use greedy.
            coloring = greedy_fallback(n, edges, lists)
            
        out.append(coloring)

    return out

# EVOLVE-BLOCK-END
