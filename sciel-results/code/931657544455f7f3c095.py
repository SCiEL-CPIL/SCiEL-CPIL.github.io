"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(splits, seed, palette=5):
        rng = random.Random(seed)
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

    def solve_instance(n, edges, lists, deadline):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order vertices by descending degree (most constrained first)
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        # Map position in order to vertex
        pos_in_order = [0] * n
        for i, v in enumerate(order):
            pos_in_order[v] = i
        
        # Backtracking with forward checking
        # colour[v] = -1 if uncoloured
        colour = [-1] * n
        # For each vertex, maintain a set of available colours (from its list minus taken by coloured neighbours)
        # But for efficiency, we'll recompute available colours on the fly or use a domain array.
        
        # Let's use a domain array: domain[v] is a list of available colours
        domain = [list(lists[v]) for v in range(n)]
        
        # We process vertices in 'order'
        # For backtracking, we need to undo changes.
        # A simpler approach for small n: standard backtracking with recursion.
        # n is at most 8 + 4*100 = 408. Recursion depth 408 is fine.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
            
            v = order[idx]
            
            # Get available colours for v considering already coloured neighbours
            # Since we process in order, some neighbours might be uncoloured (if they come later in order).
            # But in a proper coloring backtracking, we only need to check against *coloured* neighbours.
            # However, if we use forward checking, we should also update domains of future neighbours.
            # For simplicity and robustness, let's just check against coloured neighbours.
            # This is "plain" backtracking. It might be slow for hard instances, but these are likely easy.
            
            # Optimization: Check against all neighbours, but only those that are already coloured.
            taken = set()
            for u in adj[v]:
                if colour[u] != -1:
                    taken.add(colour[u])
            
            # Try each colour in v's list
            for c in lists[v]:
                if c in taken:
                    continue
                
                colour[v] = c
                if backtrack(idx + 1):
                    return True
                colour[v] = -1
            
            return False
        
        if backtrack(0):
            return colour
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            # Fallback: return empty lists or something valid but incorrect?
            # The task says "Any broken rule makes the whole result invalid".
            # If we time out, we should try to return something.
            # But if we can't finish, the score is 0.
            # Let's try to return the current best effort? No, the interface requires a list of 4 lists.
            # If we haven't solved an instance, we can't return a valid coloring for it.
            # So we must solve them.
            # Given the small size, backtracking should be fast.
            pass
        
        n, edges, lists = instance(splits, seed)
        sol = solve_instance(n, edges, lists, DEADLINE)
        
        if sol is None:
            # Fallback: try a greedy approach or return a dummy (which will score 0 for this instance)
            # But the task says "Any broken rule makes the whole result invalid".
            # Wait, "An instance whose colouring breaks a list or an edge scores 0 for that instance".
            # "but a wrong length or type makes the whole result invalid."
            # So if I return a list of correct length and type, but the coloring is invalid, that instance scores 0.
            # If I return None or wrong type, the whole result is invalid (score 0 for all).
            # So I should return a list of ints, even if invalid, to avoid total invalidity?
            # "The raw value is the number of properly coloured instances."
            # So if I return an invalid coloring for one instance, I get 0 for that one, but keep the others.
            # So it's better to return a "best effort" or even a random valid-length list than to crash or return wrong type.
            
            # Let's try a greedy coloring as a fallback
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            sol = colour
        
        # Ensure types are correct (Python int)
        sol = [int(c) for c in sol]
        out.append(sol)
    
    return out
# EVOLVE-BLOCK-END
