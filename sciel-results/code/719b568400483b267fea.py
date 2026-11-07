"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
from random import Random

def construct():
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

    def solve_instance(n, edges, lists, deadline, rng):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Degree for tie-breaking in MRV
        degree = [len(adj[i]) for i in range(n)]

        # Domains as lists for faster iteration, but we need fast removal.
        # Since domain size is small (<=3), sets are fine, or we can just check against assigned neighbors.
        # Let's use the "check against assigned neighbors" approach for domains to avoid complex undo logic.
        # This is effectively Forward Checking without explicit domain sets, just checking validity on the fly.
        
        assignment = [-1] * n
        
        sys.setrecursionlimit(10000)

        def get_available_colors(v):
            """Returns a list of colors in lists[v] that do not conflict with assigned neighbors."""
            avail = []
            for c in lists[v]:
                conflict = False
                for u in adj[v]:
                    if assignment[u] == c:
                        conflict = True
                        break
                if not conflict:
                    avail.append(c)
            return avail

        def backtrack():
            if time.time() > deadline:
                return 'timeout'
            
            # MRV: Find unassigned vertex with fewest available colors
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            
            for v in range(n):
                if assignment[v] == -1:
                    # Count available colors
                    count = 0
                    for c in lists[v]:
                        conflict = False
                        for u in adj[v]:
                            if assignment[u] == c:
                                conflict = True
                                break
                        if not conflict:
                            count += 1
                            if count == 3: # Max possible, no need to check further for this v
                                break
                    
                    if count < best_count:
                        best_count = count
                        best_v = v
                        best_deg = degree[v]
                        if count == 0:
                            return False # Dead end
                        if count == 1:
                            # Can't get better than 1 (except 0 which we caught)
                            # But we should check if another vertex has 0? 
                            # The loop continues to check all, but we can optimize:
                            # If we find 1, we still need to ensure no vertex has 0.
                            # However, if we find 0, we return immediately.
                            # So we must scan all vertices to be safe, or assume 0s are caught early?
                            # Scanning all is safer.
                            pass

            if best_v == -1:
                return True # All assigned
            
            # Get the actual available colors for the best vertex
            avail_colors = get_available_colors(best_v)
            
            # Shuffle colors to break symmetry and allow randomized restarts to explore different branches
            if len(avail_colors) > 1:
                rng.shuffle(avail_colors)

            for c in avail_colors:
                if time.time() > deadline:
                    return 'timeout'
                
                assignment[best_v] = c
                
                if backtrack():
                    return True
                
                assignment[best_v] = -1
            
            return False

        result = backtrack()
        if result is True:
            return assignment
        return None

    out = []
    start_time = time.time()
    hard_deadline = start_time + 140.0  # 10s margin before 150s kill

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        if time.time() > hard_deadline - 10.0:
             # Not enough time, return dummy (will score 0 for this instance if not solved)
             # But we should try to solve if possible quickly
             pass
        
        remaining_instances = 4 - len(out)
        time_left = hard_deadline - time.time()
        # Allocate time per instance, with a small buffer
        instance_deadline = time.time() + (time_left / remaining_instances) * 0.95
        
        colouring = None
        # Try multiple restarts with different RNG seeds for the shuffle
        for retry_seed in range(10):
            if time.time() > instance_deadline:
                break
            rng_solver = Random(retry_seed)
            colouring = solve_instance(n, edges, lists, instance_deadline, rng_solver)
            if colouring is not None:
                break
        
        if colouring is None:
            # Fallback: This shouldn't happen for these instances, but if it does,
            # we must return something of the correct type to avoid "invalid result" for the whole thing?
            # No, the contract says "An instance whose colouring breaks a list or an edge scores 0 for that instance".
            # But "a wrong length or type makes the whole result invalid".
            # So we must return a list of ints of length n.
            # If we can't find a valid one, we return a dummy that will score 0 for this instance.
            colouring = [0] * n
        
        # Ensure types are exact Python ints
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
