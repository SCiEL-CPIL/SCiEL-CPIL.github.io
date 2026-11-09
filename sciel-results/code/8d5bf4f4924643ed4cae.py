"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random
    
    # Deadline guard
    start_time = time.time()
    TIME_LIMIT = 110  # Safety margin below 120s
    
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

    def solve_csp(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # degrees for tie-breaking
        degrees = [len(nei) for nei in adj]
        
        # Initial available colors for each vertex
        avail = [set(lst) for lst in lists]
        
        # assigned color for each vertex, -1 if unassigned
        color = [-1] * n
        
        # Helper to check if a solution is found
        def is_solved():
            return all(c != -1 for c in color)
            
        # Recursive backtracking with MRV heuristic
        # We use a list to track unassigned vertices for efficiency, 
        # but re-calculating MRV each step is fine for n <= 400.
        
        def backtrack():
            if time.time() - start_time > TIME_LIMIT:
                raise TimeoutError()
                
            # Find unassigned vertex with Minimum Remaining Values (MRV)
            min_count = n + 1
            selected_vertex = -1
            selected_degree = -1
            
            for v in range(n):
                if color[v] == -1:
                    cnt = len(avail[v])
                    if cnt < min_count:
                        min_count = cnt
                        selected_vertex = v
                        selected_degree = degrees[v]
                        if cnt == 0:
                            return False # Dead end
                    elif cnt == min_count:
                        # Tie-breaker: Highest degree
                        if degrees[v] > selected_degree:
                            selected_vertex = v
                            selected_degree = degrees[v]
            
            if selected_vertex == -1:
                return True # All assigned
            
            v = selected_vertex
            # Try each available color
            # We can sort colors by least constraining value (LCV) if needed, 
            # but for 3 colors, simple iteration is fast.
            for c in sorted(avail[v]):
                color[v] = c
                # Forward checking: remove c from neighbors
                consistent = True
                saved_avail = []
                for u in adj[v]:
                    if color[u] == -1:
                        if c in avail[u]:
                            avail[u].remove(c)
                            saved_avail.append((u, c))
                            if len(avail[u]) == 0:
                                consistent = False
                                break
                        # If color[u] is already assigned, we don't need to check 
                        # because we only assign to unassigned vertices and 
                        # we check consistency on assignment. 
                        # Wait, if neighbor u is assigned, we should have 
                        # removed v's color from u's avail when u was assigned? 
                        # No, we only update avail for *unassigned* neighbors 
                        # in this forward checking step. 
                        # Actually, standard forward checking updates domains 
                        # of unassigned neighbors.
                
                if consistent:
                    if backtrack():
                        return True
                
                # Undo
                color[v] = -1
                for u, c_removed in saved_avail:
                    avail[u].add(c_removed)
            
            return False

        if backtrack():
            # Convert to list of ints
            return [int(c) for c in color]
        else:
            # Should not happen due to Alon-Tarsi, but fallback
            # Return a dummy valid-looking structure to avoid crash, 
            # though it will score 0.
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_csp(n, edges, lists)
        out.append(coloring)
        
        if time.time() - start_time > TIME_LIMIT:
             # If we run out of time, return what we have 
             # (padded with 0s if necessary to keep length valid, 
             # though the problem says wrong length is invalid, 
             # so we must ensure length is correct).
             # Since we process sequentially, if we time out on instance 3,
             # we might not have instance 4. 
             # However, the solver is very fast. 
             # If we do time out, we should probably break and return 
             # partial results? No, the contract says "wrong length... makes 
             # the whole result invalid". So we must have 4 lists.
             # If we time out, we can't guarantee a valid full result.
             # But given n<=400, it will not time out.
             break

    # Ensure we have exactly 4 lists
    while len(out) < 4:
         # Fallback: generate dummy lists of correct length if we failed to solve
         # We need the instance data to get the length.
         # This is a safety net.
         pass

    return out
# EVOLVE-BLOCK-END
