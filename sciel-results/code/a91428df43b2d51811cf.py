"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    # Set recursion limit high enough for backtracking, though depth is small
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    # The evaluator kills at 150s. We aim to finish well before.
    # 110s is a safe margin to allow for overhead and unexpected delays.
    deadline = start_time + 110.0 

    def instance(splits, seed, palette=5):
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
        Solves list coloring using Backtracking with MRV (Minimum Remaining Values)
        and Forward Checking to prune the search space more effectively.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert adjacency to sets for faster lookup if needed, but lists are fine for small degrees
        # We keep lists for iteration speed.
        
        # Initial domains for each vertex: the list of allowed colors
        # We use lists of colors, not just booleans, to easily check availability
        domains = [list(l) for l in lists]
        
        color = [-1] * n
        uncolored = set(range(n))
        
        # Pre-compute neighbor sets for potential use, though we iterate adj lists
        # adj_sets = [set(a) for a in adj]
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV Heuristic: Find uncolored vertex with fewest available colors
            # We also incorporate Degree Heuristic as a tie-breaker: among vertices with min options,
            # pick the one with highest degree.
            best_v = -1
            min_options = n + 1
            best_degree = -1
            best_options = None
            
            # Iterate over uncolored vertices
            for v in uncolored:
                # Determine taken colors by colored neighbors
                taken = set()
                for u in adj[v]:
                    if color[u] != -1:
                        taken.add(color[u])
                
                # Determine available colors
                options = []
                for c in domains[v]:
                    if c not in taken:
                        options.append(c)
                
                if not options:
                    return False
                
                deg = len(adj[v])
                
                if len(options) < min_options or (len(options) == min_options and deg > best_degree):
                    min_options = len(options)
                    best_v = v
                    best_options = options
                    best_degree = deg
                    if min_options == 1:
                        # Can't do better than 1 option, but might find higher degree with 1 option?
                        # Usually MRV dominates. Let's just break for speed if we find 1,
                        # unless we are strictly optimizing tie-breaking which is negligible here.
                        # However, to be safe and consistent:
                        # If we find 1 option, we can break early as it's the minimum possible.
                        # But wait, another vertex might have 1 option and higher degree.
                        # Does it matter? Not really for correctness.
                        # Let's just break to save time.
                        break
            
            if best_v == -1:
                return False # Should not happen if uncolored is not empty
            
            uncolored.remove(best_v)
            
            # Try each option
            # Order colors by least constraining value? 
            # LCV: pick the color that rules out the fewest choices of neighbors.
            # This is a bit expensive to compute. Given small domain size (3), it's fine.
            # But for simplicity and speed, we can just try them in order.
            # Let's stick to simple order first. If this is too slow, we can add LCV.
            
            for c in best_options:
                color[best_v] = c
                
                # Forward Checking: Check if any uncolored neighbor has no available colors
                # We don't modify domains permanently, we just check.
                # If we modify domains, we need to backtrack them.
                # For small graphs, checking is O(degree).
                
                valid = True
                # Check neighbors
                for u in adj[best_v]:
                    if color[u] != -1:
                        continue
                    # Check if u still has options
                    taken_u = set()
                    for w in adj[u]:
                        if color[w] != -1:
                            taken_u.add(color[w])
                    # Also consider best_v's color
                    taken_u.add(c)
                    
                    has_option = False
                    for col in domains[u]:
                        if col not in taken_u:
                            has_option = True
                            break
                    if not has_option:
                        valid = False
                        break
                
                if valid:
                    if backtrack():
                        return True
                else:
                    # Backtrack: color is reset below
                    pass
                    
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
