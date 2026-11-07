"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock kill

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

    def ac3(domains, adj, n):
        """Arc consistency enforcement. Returns False if a domain becomes empty."""
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        while queue:
            if time.time() > deadline:
                return False
            u, v = queue.popleft()
            # Check if domain of u is consistent with domain of v
            for c in list(domains[u]):
                if not any(c != d for d in domains[v]):
                    domains[u].remove(c)
                    if not domains[u]:
                        return False
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
        return True

    def solve_instance(n, edges, lists):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initialize domains as sets for fast operations
        domains = [set(l) for l in lists]
        
        # Run AC-3 to prune domains
        if not ac3(domains, adj, n):
            # Should not happen for these instances, but handle gracefully
            # Fall back to simple backtracking
            pass
        
        # Order vertices by ascending domain size (MRV), tie-break by degree
        order = sorted(range(n), key=lambda u: (len(domains[u]), -len(adj[u])))
        
        # Iterative backtracking with forward checking
        color = [-1] * n
        # domains_remaining tracks available colors for each unassigned vertex
        domains_remaining = [set(d) for d in domains]
        
        # Use a stack for iterative DFS
        # Each stack entry: (vertex_index_in_order, color_choice_index)
        # We need to track the state carefully
        
        # Simpler: use recursion but with increased limit and careful time checking
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, n + 100))
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == len(order):
                return True
            v = order[pos]
            # Get available colors considering assigned neighbors
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            
            # Try colors in the domain
            for c in sorted(domains_remaining[v]):
                if c not in taken:
                    color[v] = c
                    # Forward checking: check if any unassigned neighbor loses all colors
                    consistent = True
                    for u in adj[v]:
                        if color[u] == -1:
                            # Check if u still has available colors
                            taken_u = set()
                            for w in adj[u]:
                                if color[w] != -1:
                                    taken_u.add(color[w])
                            if not any(cc not in taken_u for cc in domains_remaining[u]):
                                consistent = False
                                break
                    if consistent:
                        if backtrack(pos + 1):
                            return True
                    color[v] = -1
            return False
        
        if backtrack(0):
            sys.setrecursionlimit(old_limit)
            return color
        else:
            sys.setrecursionlimit(old_limit)
            # Should not happen due to Alon-Tarsi, but return best effort
            return color

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 10:
            # Not enough time for this instance, break
            break
        n, edges, lists = instance(splits, seed)
        coloring = solve_instance(n, edges, lists)
        # Ensure types are Python int
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    
    # If out has less than 4 items, pad with empty lists to maintain valid structure?
    # No, wrong length makes result invalid. We must return exactly 4 lists.
    # If we run out of time, we're stuck. But for these small instances, it should be fast.
    while len(out) < 4:
        out.append([])
    
    # Convert to list of lists of ints
    return [list(map(int, c)) for c in out]

# EVOLVE-BLOCK-END
