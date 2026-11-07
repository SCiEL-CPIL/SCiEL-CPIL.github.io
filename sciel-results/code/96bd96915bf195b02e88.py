"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [set(lst) for lst in lists]
        # We will store the current assignment as a list of ints, -1 for unassigned
        assignment = [-1] * n
        
        # Helper to check if assignment is valid so far
        # We use forward checking, so we maintain domains dynamically?
        # It's easier to just backtrack on assignment and check constraints.
        # Given small N (max 100 splits -> 8 + 4*100 = 408 vertices), 
        # and small degree, a simple recursive backtracking with MRV should be fast.
        
        # To speed up MRV, we can pre-compute a static order or just calculate on the fly.
        # Calculating on the fly is safer.
        
        sys.setrecursionlimit(2000)
        
        def is_assigned(v):
            return assignment[v] != -1
            
        def get_domain(v):
            # Returns list of colors in lists[v] that are not used by assigned neighbors
            avail = []
            for c in lists[v]:
                if is_assigned(v):
                    # This shouldn't happen if we only call for unassigned
                    pass
                conflict = False
                for u in adj[v]:
                    if is_assigned(u) and assignment[u] == c:
                        conflict = True
                        break
                if not conflict:
                    avail.append(c)
            return avail

        # MRV: pick unassigned vertex with fewest available colors
        def select_mrv_vertex():
            min_count = float('inf')
            min_v = -1
            for v in range(n):
                if not is_assigned(v):
                    # Count available colors
                    count = 0
                    for c in lists[v]:
                        conflict = False
                        for u in adj[v]:
                            if is_assigned(u) and assignment[u] == c:
                                conflict = True
                                break
                        if not conflict:
                            count += 1
                    if count < min_count:
                        min_count = count
                        min_v = v
                        if count == 0:
                            return v, [] # Dead end
                        if count == 1:
                            # Can break early? No, we want the absolute min. 
                            # But 1 is the minimum possible for a valid state (unless 0).
                            # Actually, if we find 0, we return immediately.
                            # If we find 1, we can't break because another vertex might have 0.
                            pass
            if min_v == -1:
                return -1, [] # All assigned
            # Re-calculate domain for min_v to return it
            domain = get_domain(min_v)
            return min_v, domain

        def backtrack():
            # Check if all assigned
            # Optimization: count assigned? Or just let select_mrv return -1.
            v, domain = select_mrv_vertex()
            
            if v == -1:
                return True # All vertices assigned successfully
            
            if not domain:
                return False # No color available for v
            
            # Try each color in domain
            # Order of colors? Maybe try colors that are least constraining?
            # For now, just iterate.
            for c in domain:
                assignment[v] = c
                if backtrack():
                    return True
                assignment[v] = -1
            
            return False

        # Start backtracking
        if backtrack():
            return assignment
        else:
            # Should not happen per Alon-Tarsi, but return empty or raise?
            # Task says "An instance whose colouring breaks a list or an edge scores 0".
            # If we can't find one, we return a dummy list of 0s which will likely fail, 
            # or we can try a different strategy?
            # Given the guarantee, this shouldn't happen.
            return [0] * n

    out = []
    start_time = time.time()
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Time check
        if time.time() - start_time > 140:
            # If we run out of time, we need to return something valid length-wise.
            # We will just fill with 0s for remaining instances, which will score 0 for those.
            # But we must ensure the list length is correct.
            # We don't know n without generating.
            # We must generate to know n.
            pass
            
        n, edges, lists = instance(splits, seed)
        
        # If time is tight, we might skip solving and return zeros?
        # But we need to know n first.
        # Generation is fast. Solving is the bottleneck.
        
        if time.time() - start_time > 110:
             # Safety margin: if we are close to limit, return dummy for remaining
             # We have already generated n.
             out.append([0] * n)
             continue
            
        colouring = solve_instance(n, edges, lists)
        
        # Ensure types are Python int
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
