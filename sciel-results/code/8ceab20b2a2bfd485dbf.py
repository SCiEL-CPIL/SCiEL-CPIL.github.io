"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque
from random import Random

def construct():
    # Set a higher recursion limit just in case, though iterative backtracking is safer
    sys.setrecursionlimit(10000)
    
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit

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
        
        # Initialize domains
        domains = [list(lst) for lst in lists]
        
        # AC-3 Algorithm for initial propagation
        def ac3(domains):
            queue = deque()
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                xi, xj = queue.popleft()
                if revise(domains, xi, xj):
                    if not domains[xi]:
                        return False
                    for xk in adj[xi]:
                        if xk != xj:
                            queue.append((xk, xi))
            return True

        def revise(domains, xi, xj):
            revised = False
            to_remove = []
            for val in domains[xi]:
                if not any(val != val_j for val_j in domains[xj]):
                    to_remove.append(val)
            for val in to_remove:
                domains[xi].remove(val)
                revised = True
            return revised

        # If AC-3 fails, no solution
        if not ac3(domains):
            return None

        # Backtracking with Forward Checking
        # We use a stack-based approach to avoid recursion depth issues, 
        # but given the size, recursion with high limit is also fine. 
        # Let's use a simple recursive backtracker for clarity, as n <= 408.
        
        assignment = {}
        
        def backtrack():
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
                
            # Find unassigned variable with MRV (Minimum Remaining Values)
            unassigned = [v for v in range(n) if v not in assignment]
            if not unassigned:
                return True
                
            # MRV: choose variable with smallest domain
            v = min(unassigned, key=lambda u: len(domains[u]))
            
            # Order values (we can just use the sorted order from initialization)
            for val in domains[v]:
                # Check if val is consistent with assigned neighbours
                consistent = True
                for u in adj[v]:
                    if u in assignment:
                        if assignment[u] == val:
                            consistent = False
                            break
                
                if consistent:
                    # Assign
                    assignment[v] = val
                    
                    # Forward Checking: Check if any neighbour has an empty domain
                    save_domains = {}
                    consistent_fc = True
                    for u in adj[v]:
                        if u not in assignment:
                            save_domains[u] = domains[u][:]
                            if val in domains[u]:
                                domains[u].remove(val)
                                if not domains[u]:
                                    consistent_fc = False
                                    break
                    
                    if consistent_fc:
                        if backtrack():
                            return True
                    
                    # Undo
                    for u in save_domains:
                        domains[u] = save_domains[u]
                    del assignment[v]
            
            return False

        if backtrack():
            return [assignment[v] for v in range(n)]
        else:
            return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Solve the CSP
        colouring = solve_csp(n, edges, lists)
        
        if colouring is None:
            # Fallback: If CSP fails (should not happen due to Alon-Tarsi), 
            # we might try a random restart or just return an invalid colouring 
            # which scores 0 for that instance. 
            # However, the problem states they are colourable. 
            # If it fails, it's likely a bug or timeout.
            # To be safe against "invalid result" if types are wrong, 
            # we ensure we return a list of ints.
            # If solve_csp returns None, we can't provide a valid colouring.
            # But we must return something of the correct type.
            # Let's try a simple greedy as a last resort, even if it fails validity.
            # Actually, if it fails, the score for that instance is 0.
            # We just need to return a list of length n.
            colouring = [0] * n # Placeholder, will be invalid but type correct
        else:
            # Ensure types are Python int
            colouring = [int(c) for c in colouring]
            
        out.append(colouring)
        
        # Check deadline
        if time.time() > deadline:
             # If we run out of time, we return what we have.
             # The remaining instances will be missing or invalid.
             # The contract says "Any broken rule makes the whole result invalid".
             # So we must return 4 lists. If we haven't solved all, we must fill them.
             # But if we time out mid-solve, we might not have the list.
             # The loop above handles this by appending.
             # If we break early, out will have < 4 items.
             # We must ensure out has 4 items.
             pass

    # Ensure we have exactly 4 lists
    while len(out) < 4:
         out.append([0,0,0,0,0,0,0,0]) # Dummy

    return out

# EVOLVE-BLOCK-END
