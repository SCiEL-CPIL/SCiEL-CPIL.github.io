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

    def solve(n, edges, lists, time_limit=110.0):
        start_time = time.time()
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Sort vertices by degree descending to prioritize high-constraint vertices
        # This helps in finding conflicts early
        order = sorted(range(n), key=lambda x: -len(adj[x]))
        
        # Map original index to position in order for quick lookup if needed, 
        # but we will just iterate in 'order'
        
        # Pre-compute neighbor lists in the order of 'order' for faster access?
        # Actually, standard backtracking with forward checking is sufficient.
        
        color = [-1] * n
        # available_colors[v] is a list of colors that are still possible for v
        available = [list(l) for l in lists]
        
        # To make forward checking efficient, we can maintain a count of how many 
        # neighbors have a specific color, but since lists are small, 
        # simply checking neighbors is fine.
        
        def backtrack(idx):
            if time.time() - start_time > time_limit:
                return False
            
            if idx == n:
                return True
            
            v = order[idx]
            
            # Try each available color for v
            # We use a copy of the list to try, or iterate and backtrack
            # Since lists are small (size 3), iterating is cheap.
            
            # Heuristic: Try colors that are most constrained for neighbors? 
            # Simple approach: try all available colors.
            
            for c in available[v]:
                # Check if c is compatible with all already colored neighbors
                conflict = False
                for u in adj[v]:
                    if color[u] != -1 and color[u] == c:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Assign
                color[v] = c
                
                # Forward checking: Remove c from neighbors' available lists
                # We need to restore this on backtrack.
                # Since lists are small, we can just filter or mark.
                # A simpler robust way: just check validity in the loop above.
                # Forward checking optimization:
                # If we want to prune, we check if any neighbor becomes empty.
                
                # Let's stick to the simpler "check on assignment" first. 
                # If that's too slow, we add forward checking.
                # Given N<=400 and degree ~4-8, simple check is O(N*Deg).
                # Total nodes visited in tree could be large, but branching factor is small.
                
                if backtrack(idx + 1):
                    return True
                
                # Un-assign
                color[v] = -1
            
            return False
        
        # The simple backtracking above might be slow if it explores a huge tree.
        # Let's implement a slightly smarter version with forward checking to be safe.
        
        # Reset state for smarter solver
        color = [-1] * n
        # avail[v] is a set of available colors for v
        avail = [set(l) for l in lists]
        
        def backtrack_fast(idx):
            if time.time() - start_time > time_limit:
                return False
            
            if idx == n:
                return True
            
            v = order[idx]
            
            # Order colors? Maybe try colors that appear least in neighbors?
            # Simple: iterate over sorted available colors for consistency
            for c in sorted(avail[v]):
                # Check neighbors
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign
                color[v] = c
                old_avails = {}
                all_valid = True
                for u in adj[v]:
                    if color[u] == -1:
                        if c in avail[u]:
                            avail[u].remove(c)
                            if not avail[u]:
                                all_valid = False
                                break
                            old_avails[u] = avail[u].copy()
                        else:
                            # c was already not in avail[u], so no change needed, 
                            # but we must ensure we don't modify if not present
                            pass
                
                if all_valid:
                    if backtrack_fast(idx + 1):
                        return True
                
                # Backtrack
                color[v] = -1
                for u in old_avails:
                    avail[u] = old_avails[u]
            
            return False
        
        if backtrack_fast(0):
            return color
        else:
            # If failed, return a dummy valid structure? 
            # The task says "Any broken rule makes the whole result invalid".
            # If we can't find a solution, we return a list of -1s which will score 0 for that instance.
            # But we must return a list of ints.
            return [-1] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve(n, edges, lists)
        # Ensure types are Python ints
        out.append([int(c) for c in coloring])
        
    return out
# EVOLVE-BLOCK-END
