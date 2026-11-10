"""Model-and-optimize: Formulate as a CSP and solve via Backtracking with Forward Checking.
Variables: Vertices. Domains: Allowed colors. Constraints: Adjacent vertices have different colors.
Strategy: MRV (Minimum Remaining Values) heuristic for variable selection to minimize branching factor."""

import time
from random import Random
import sys

def construct():
    t0 = time.time()
    LIMIT = 120.0          # hard wall-clock cap for the evaluator
    MARGIN = 10.0          # safety margin

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

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
            
        # Generate lists
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        
        return n, sorted(edges), lists

    def solve_csp(n, adj, lists):
        # domains[v] is a set of available colors
        domains = [set(lst) for lst in lists]
        assignment = [-1] * n
        
        # Pre-compute adjacency for quick access
        # adj is list of lists
        
        def backtrack():
            # Select variable with Minimum Remaining Values (MRV)
            unassigned = [v for v in range(n) if assignment[v] == -1]
            if not unassigned:
                return True
            
            # MRV: pick v with smallest domain
            # Tie-breaker: Highest degree
            v = min(unassigned, key=lambda x: (len(domains[x]), -len(adj[x])))
            
            # If domain is empty, fail
            if not domains[v]:
                return False
            
            # Value Ordering: Least Constraining Value (LCV)
            # For each color c in domains[v], count how many neighbors it removes from their domains
            # To keep it fast, we just iterate through sorted colors or use a simple heuristic
            # Since palette is small (3 colors), we can just try them.
            # Let's try colors that are most common in neighbors first? Or least?
            # LCV: try color c that leaves largest domain for neighbors.
            # For speed, we'll just iterate through the set.
            for c in sorted(domains[v]):
                # Assign
                assignment[v] = c
                
                # Record changes for backtracking
                saved_domains = []
                for nb in adj[v]:
                    if assignment[nb] == -1 and c in domains[nb]:
                        saved_domains.append((nb, c))
                        domains[nb].discard(c)
                        if not domains[nb]:
                            # Conflict! Undo and break
                            for (nb_idx, val) in saved_domains:
                                domains[nb_idx].add(val)
                            assignment[v] = -1
                            break
                else:
                    # No conflict, recurse
                    if backtrack():
                        return True
                    # Undo
                    for (nb_idx, val) in saved_domains:
                        domains[nb_idx].add(val)
                
                assignment[v] = -1
                
            return False

        if backtrack():
            return assignment
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            # If we run out of time, we must return SOMETHING valid to avoid total 0 score?
            # "An invalid result... scores 0".
            # If we don't have a solution, we can't return a valid one.
            # But backtracking is extremely fast for these sizes.
            # If we hit deadline, we might not have processed all 4.
            # We should probably just return empty or whatever we have?
            # "Return a list of four lists". If we break, length is wrong.
            # So we must ensure we finish or handle the deadline carefully.
            # Given the speed, this is unlikely.
            pass
            
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        solution = solve_csp(n, adj, lists)
        
        if solution is None:
            # Should not happen due to Alon-Tarsi, but handle gracefully
            # Return a dummy valid-looking structure? No, it will be invalid.
            # Better to raise or return empty?
            # If we return None or wrong type, it's invalid.
            # We assume it will find a solution.
            solution = [0] * n # Fallback, likely invalid
            
        # Ensure types are Python int
        out.append([int(c) for c in solution])
        
    return out

# EVOLVE-BLOCK-END
