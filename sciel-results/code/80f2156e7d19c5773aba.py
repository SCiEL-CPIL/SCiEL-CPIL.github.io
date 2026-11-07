"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
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
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # MRV + Forward Checking Backtracking
        # We will maintain domains as a list of sets (or lists)
        domains = [list(l) for l in lists]
        
        # Order vertices by MRV, breaking ties by degree (highest first)
        # For static ordering in this small problem, we can pick dynamically
        color = [None] * n
        
        # To be fast, we use a simple recursive backtrack with forward checking
        # and variable selection based on current domain size (MRV).
        
        import sys
        sys.setrecursionlimit(10000)
        
        # Convert adj to lists for faster iteration if needed, but sets are fine for 'in'
        # Let's stick with sets for adj.
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find uncolored vertex with smallest domain (MRV)
            # Tie-breaker: highest degree
            min_size = 4
            best_v = -1
            best_deg = -1
            for v in range(n):
                if color[v] is None:
                    d = len(domains[v])
                    deg = len(adj[v])
                    # If domain is empty, dead end
                    if d == 0:
                        return False
                    if d < min_size or (d == min_size and deg > best_deg):
                        min_size = d
                        best_v = v
                        best_deg = deg
                        
            if best_v == -1:
                return True # All colored
            
            # Try each color in domain of best_v
            # Heuristic: try colors that constrain neighbors least? Or just arbitrary?
            # Let's try arbitrary first, MRV usually suffices for these graphs.
            for c in domains[best_v]:
                # Check if c conflicts with any colored neighbor
                valid = True
                for u in adj[best_v]:
                    if color[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign
                color[best_v] = c
                # Forward Checking: remove c from neighbors' domains
                undo_stack = []
                valid_fc = True
                for u in adj[best_v]:
                    if color[u] is None:
                        if c in domains[u]:
                            domains[u].remove(c)
                            undo_stack.append(u)
                            if len(domains[u]) == 0:
                                valid_fc = False
                                break
                            
                if valid_fc:
                    if backtrack():
                        return True
                else:
                    pass # already failed
                
                # Undo
                color[best_v] = None
                for u in undo_stack:
                    domains[u].append(c)
                    # Maintain sorted order if needed, but not strictly necessary for logic
                    # However, for determinism and potential optimizations, sorting might help?
                    # No, 'in' check and removal is O(k). Keeping unsorted is fine for k<=3.
        
        return backtrack()

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Make a deep copy of lists for solver to mutate
        local_lists = [list(l) for l in lists]
        
        if solve_instance(n, edges, local_lists):
            # Ensure types are Python int
            result = [int(c) for c in color]
            out.append(result)
        else:
            # Fallback if backtracking fails (should not happen per Alon-Tarsi, but just in case)
            # The contract says "Any broken rule makes the whole result invalid"
            # But wait, the task says "Maximize; all are colourable".
            # If it fails, we return something? The seed returned greedy which might be invalid.
            # The score is number of *properly coloured* instances.
            # If we return an invalid coloring, that instance scores 0.
            # But we need to return a list of ints.
            # Let's try greedy as a fallback just to have *something*, even if invalid?
            # Actually, the prompt says "wrong length or type makes the whole result invalid".
            # It does not say "wrong color makes the whole result invalid", it says "An instance whose colouring breaks a list or an edge scores 0 for that instance".
            # So we should return a list of ints.
            
            # Simple greedy fallback
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            color = [None] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {color[u] for u in adj[v] if color[u] is not None}
                free = [c for c in local_lists[v] if c not in taken]
                color[v] = free[0] if free else local_lists[v][0]
            result = [int(c) for c in color]
            out.append(result)
            
    return out
# EVOLVE-BLOCK-END
