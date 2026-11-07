"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    from random import Random

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
        """Backtracking with forward checking and MRV heuristic."""
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Domain for each vertex: the list of allowed colours
        domains = [list(l) for l in lists]
        
        # Order vertices by degree descending for initial assignment order
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        
        colour = [None] * n
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            # MRV: pick the unassigned vertex with fewest available colours
            # We use the precomputed order but can dynamically choose
            # For efficiency, we just follow the order but check domains
            v = order[idx]
            
            # Get available colours (not used by assigned neighbours)
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            
            available = [c for c in domains[v] if c not in taken]
            if not available:
                return False
            
            # Try each colour
            for c in available:
                colour[v] = c
                if backtrack(idx + 1):
                    return True
                colour[v] = None
            
            return False
        
        if backtrack(0):
            # Ensure all are Python ints
            return [int(c) for c in colour]
        else:
            # Fallback: should not happen due to Alon-Tarsi, but return something
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5.0:
            # Not enough time left, break
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        # Validate types
        colouring = [int(c) for c in colouring]
        out.append(colouring)
    
    # If we didn't finish all 4, pad with fallback (should not happen)
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        out.append([lists[v][0] for v in range(n)])
    
    return out
# EVOLVE-BLOCK-END
