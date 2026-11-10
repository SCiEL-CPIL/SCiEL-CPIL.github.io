"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def instance(splits, seed, palette=5):
        from random import Random
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
        
        # Initialize domains
        domains = [set(lst) for lst in lists]
        
        # AC-3 propagation
        def ac3(domains):
            queue = deque()
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                if revise(domains, u, v):
                    if not domains[u]:
                        return False
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
            return True

        def revise(domains, u, v):
            revised = False
            for c in list(domains[u]):
                if not any(c != d for d in domains[v]):
                    domains[u].remove(c)
                    revised = True
            return revised

        # Backtracking search
        def backtrack(domains, assigned):
            if time.time() > deadline:
                return None
            
            # Find unassigned variable with smallest domain (MRV heuristic)
            unassigned = [i for i in range(n) if i not in assigned]
            if not unassigned:
                return assigned
            
            # Sort by domain size
            unassigned.sort(key=lambda i: len(domains[i]))
            u = unassigned[0]
            
            # Try each color in domain
            for c in sorted(domains[u]):
                # Check consistency with assigned neighbors
                if all(assigned.get(v, None) != c for v in adj[u] if v in assigned):
                    # Assign
                    assigned[u] = c
                    # Propagate
                    new_domains = [set(d) for d in domains]
                    new_domains[u] = {c}
                    # Revise neighbors
                    valid = True
                    for v in adj[u]:
                        if v in assigned:
                            if assigned[v] == c:
                                valid = False
                                break
                        else:
                            if c in new_domains[v]:
                                new_domains[v].remove(c)
                                if not new_domains[v]:
                                    valid = False
                                    break
                    
                    if valid:
                        # Run AC-3 on the rest to prune further
                        if ac3(new_domains):
                            result = backtrack(new_domains, assigned)
                            if result is not None:
                                return result
                    
                    # Unassign
                    del assigned[u]
            
            return None

        # Initial AC-3
        if not ac3(domains):
            return None
        
        return backtrack(domains, {})

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback: return empty lists or best effort, but this shouldn't happen
            # We need to return a list of 4 lists of ints.
            # If we run out of time, we might not have solved all.
            # The task says "Maximize; all are colourable".
            # If we fail, we score 0 for that instance.
            # We should try to return what we have.
            # But the structure must be valid.
            pass
        
        n, edges, lists = instance(splits, seed)
        coloring = solve_csp(n, edges, lists)
        
        if coloring is None:
            # Fallback: try a simple greedy if backtracking fails or times out?
            # Or just return a dummy valid-length list of 0s? 
            # But 0s might violate constraints.
            # Since the problem is guaranteed solvable, failure is a bug or timeout.
            # Given the small size, timeout is unlikely.
            # Let's just return a list of 0s as a placeholder if it fails, 
            # though it will likely score 0 for that instance.
            coloring = [0] * n
            
        # Ensure types are correct (Python int)
        out.append([int(c) for c in coloring])
        
    return out
# EVOLVE-BLOCK-END
