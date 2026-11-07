"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random

    # Use a deadline slightly below the hard limit to account for overhead
    # The hard limit is 120s (or 150s kill). 110s is safe.
    deadline = time.monotonic() + 110.0

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        # Cube faces
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        
        # Grow the graph
        for _ in range(splits):
            face_idx = rng.randrange(len(faces))
            face = faces.pop(face_idx)
            inner = list(range(n, n + 4))
            n += 4
            
            # Add edges for the new square and connect to old face
            for i in range(4):
                a = face[i]
                b = face[(i + 1) % 4]
                u = inner[i]
                v = inner[(i + 1) % 4]
                
                edges.add((min(a, u), max(a, u)))
                edges.add((min(b, v), max(b, v)))
                edges.add((min(u, v), max(u, v)))
                
                # New face: a, b, v, u
                faces.append([a, b, v, u])
            
            # Inner face: u0, u1, u2, u3
            faces.append(inner)

        # Generate random lists of 3 colors from 0..4
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        
        # Return sorted edges for consistency
        return n, sorted(edges), lists

    def solve_csp(n, edges, color_lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Pre-compute degrees for potential tie-breaking
        degrees = [len(adj[i]) for i in range(n)]
        
        # Domains
        # Using lists for domains is faster than sets for small sizes (size 3)
        # But we need to check membership. Sets are better for removal.
        # Let's stick to sets for correctness of forward checking, but optimize access.
        orig_domains = [list(color_lists[v]) for v in range(n)]
        domains = [set(color_lists[v]) for v in range(n)]
        assignment = [None] * n
        
        # Local references for speed
        _adj = adj
        _assignment = assignment
        _domains = domains
        _deadline = deadline
        _deg = degrees

        def forward_check(v, c):
            """
            Remove c from domains of unassigned neighbors.
            Returns list of (neighbor, old_domain) for undo, or None if wipeout.
            """
            changes = []
            for u in _adj[v]:
                if _assignment[u] is None:
                    dom = _domains[u]
                    if c in dom:
                        old = dom
                        new = dom - {c}
                        if not new:
                            # Undo changes
                            for (uu, old_dom) in changes:
                                _domains[uu] = old_dom
                            return None
                        changes.append((u, old))
                        _domains[u] = new
            return changes

        def undo_forward_check(changes):
            for (u, old_dom) in changes:
                _domains[u] = old_dom

        def get_mrv_vertex():
            """Pick unassigned vertex with minimum remaining values."""
            min_size = 4 # Max domain size is 3, so 4 is inf
            best = -1
            for v in range(n):
                if _assignment[v] is None:
                    sz = len(_domains[v])
                    if sz == 0:
                        return v
                    if sz < min_size:
                        min_size = sz
                        best = v
            return best

        def backtrack():
            if time.monotonic() > _deadline:
                return None

            v = get_mrv_vertex()
            if v == -1:
                return _assignment[:]

            d = _domains[v]
            
            # Heuristic: Sort colors by least constraining value (LCV)
            # LCV: choose color that rules out the fewest choices for neighbors
            # Since domain size is small (<=3), this is cheap.
            def count_eliminations(c):
                count = 0
                for u in _adj[v]:
                    if _assignment[u] is None and c in _domains[u]:
                        count += 1
                return count
            
            # We want to try colors with *fewest* eliminations first?
            # Actually, standard LCV is to try the value that rules out the *least* 
            # for neighbors, leaving them more options.
            sorted_colors = sorted(d, key=count_eliminations)

            for c in sorted_colors:
                _assignment[v] = c
                changes = forward_check(v, c)
                if changes is not None:
                    result = backtrack()
                    if result is not None:
                        return result
                # Undo
                if changes is not None:
                    undo_forward_check(changes)
                _assignment[v] = None

            return None

        result = backtrack()
        if result is not None:
            # Ensure types are correct
            return [int(c) for c in result]
        
        # Fallback: Greedy (should not be needed for these instances)
        # If backtracking fails due to timeout, we return a best-effort greedy
        # which might be invalid, but avoids crashing.
        colour = [None] * n
        # Order by degree descending
        order = sorted(range(n), key=lambda u: -_deg[u])
        for v in order:
            taken = {colour[u] for u in _adj[v] if colour[u] is not None}
            free = [c for c in color_lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                # Pick color with fewest conflicts
                best_c = color_lists[v][0]
                min_conf = float('inf')
                for c in color_lists[v]:
                    conf = sum(1 for u in _adj[v] if colour[u] == c)
                    if conf < min_conf:
                        min_conf = conf
                        best_c = c
                colour[v] = best_c
        return [int(c) for c in colour]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.monotonic() > deadline:
            # Emergency fallback for remaining instances
            n, edges, lists = instance(splits, seed)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colour = [None] * n
            order = sorted(range(n), key=lambda u: -len(adj[u]))
            for v in order:
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                if free:
                    colour[v] = free[0]
                else:
                    colour[v] = lists[v][0]
            out.append([int(c) for c in colour])
            continue
        
        n, edges, lists = instance(splits, seed)
        out.append(solve_csp(n, edges, lists))
        
    return out
# EVOLVE-BLOCK-END
