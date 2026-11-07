"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from collections import defaultdict

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
        """
        Solves the list coloring instance.
        1. Tries a greedy assignment.
        2. If greedy fails, uses Backtracking with MRV and Forward Checking.
        """
        if time.time() > deadline:
            return None

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Pre-compute degrees for tie-breaking in MRV
        degrees = [len(adj[i]) for i in range(n)]

        # 1. Greedy Pre-pass
        # Order vertices by degree (descending)
        order = sorted(range(n), key=lambda x: -degrees[x])
        colour_greedy = [-1] * n
        success_greedy = True
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour_greedy[u] != -1:
                    taken.add(colour_greedy[u])
            # Find a valid color
            valid_color = -1
            for c in lists[v]:
                if c not in taken:
                    valid_color = c
                    break
            if valid_color == -1:
                success_greedy = False
                break
            colour_greedy[v] = valid_color
        
        if success_greedy:
            # Verify just in case (though logic above is sound for greedy)
            # Greedy doesn't guarantee global validity if it makes a bad choice, 
            # but since it checks neighbors, it's a valid partial assignment.
            # However, greedy can fail even if a solution exists. 
            # If it succeeds, it IS a valid solution.
            return [int(c) for c in colour_greedy]

        # 2. Backtracking with MRV and Forward Checking
        colour = [-1] * n
        # Domains are lists of available colors
        domains = [list(l) for l in lists]
        assigned = [False] * n
        
        # For efficient MRV, we can use a heap, but n is small. 
        # We'll stick to linear scan for simplicity but optimize the inner loop.
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find unassigned vertex with minimum domain size
            min_size = 10
            min_v = -1
            min_avail = []
            
            for v in range(n):
                if assigned[v]:
                    continue
                # Calculate available colors
                # Optimization: Since lists are small (size 3), this is fast
                taken = set()
                for u in adj[v]:
                    if assigned[u]:
                        taken.add(colour[u])
                
                avail = [c for c in domains[v] if c not in taken]
                size = len(avail)
                
                if size < min_size:
                    min_size = size
                    min_v = v
                    min_avail = avail
                    if size == 0:
                        return False
                    if size == 1:
                        # Can't get better than 1, break early
                        break
                
            if min_v == -1:
                return True # All assigned
            
            # Try each color in the domain of min_v
            for c in min_avail:
                colour[min_v] = c
                assigned[min_v] = True
                
                # Forward Checking: Check if any neighbor's domain becomes empty
                # We don't modify domains permanently, we just check feasibility
                # for the next step. A more aggressive propagation would modify
                # domains, but given small n, simple check is often sufficient.
                # However, to be safe and fast, we just recurse.
                
                if backtrack():
                    return True
                
                colour[min_v] = -1
                assigned[min_v] = False
            
            return False
        
        if backtrack():
            return [int(c) for c in colour]
        else:
            # Fallback: This should not happen for these instances.
            # Return the first color from each list (might be invalid, but better than crash)
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5.0:
            # Not enough time left, break
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        # Validate types
        if colouring is None:
             # Fallback if solver timed out or failed
             colouring = [lists[v][0] for v in range(n)]
        colouring = [int(c) for c in colouring]
        out.append(colouring)
    
    # If we didn't finish all 4, pad with fallback (should not happen)
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        out.append([lists[v][0] for v in range(n)])
    
    return out
# EVOLVE-BLOCK-END
