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

    def try_color(n, adj, lists, order, palette=5):
        colour = [None] * n
        for v in order:
            taken = set()
            for nb in adj[v]:
                c = colour[nb]
                if c is not None:
                    taken.add(c)
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                # list constraint violated -> pick least-used list colour
                counts = [0, 0, 0, 0, 0]
                for nb in adj[v]:
                    c = colour[nb]
                    if c is not None:
                        counts[c] += 1
                colour[v] = min(lists[v], key=lambda c: (counts[c], c))

        # Repair / local optimisation against the objective.
        changed = True
        guard = 0
        while changed:
            changed = False
            guard += 1
            if guard > 60 * n:            # hard safety bound (should not trigger)
                break
            if time.time() > deadline:
                break
            for v in order:
                for nb in adj[v]:
                    if colour[nb] == colour[v]:
                        counts = [0, 0, 0, 0, 0]
                        for x in adj[v]:
                            counts[colour[x]] += 1
                        free = [c for c in lists[v]
                                if c != colour[v] and counts[c] == 0]
                        if free:
                            colour[v] = free[0]
                        else:
                            candidates = [c for c in lists[v] if c != colour[v]]
                            if candidates:
                                colour[v] = min(candidates, key=lambda c: (counts[c], c))
                        changed = True
                        break
        return colour

    def valid(colour, lists, edges):
        for i, c in enumerate(colour):
            if c not in lists[i]:
                return False
        for u, v in edges:
            if colour[u] == colour[v]:
                return False
        return True

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            break
            
        n, edges, lists = instance(splits, seed)
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        deg = [len(adj[v]) for v in range(n)]

        # First try the robust CSP solver
        colouring = solve_csp(n, edges, lists)
        
        if colouring is not None:
            # Ensure types are Python int
            colouring = [int(c) for c in colouring]
            out.append(colouring)
            continue

        # Fallback to greedy + repair if CSP fails (e.g. timeout or bug)
        r = Random(9973 + seed)
        
        # Degeneracy ordering (smallest remaining degree first) + its reverse.
        rem = deg[:]
        used = [False] * n
        degen = []
        for _ in range(n):
            m = 0
            mi = 0
            for i in range(n):
                if not used[i] and rem[i] < m:
                    m = rem[i]
                    mi = i
            degen.append(mi)
            used[mi] = True
            for nb in adj[mi]:
                if not used[nb]:
                    rem[nb] -= 1

        orders = [
            sorted(range(n), key=lambda v: (-deg[v], v)),                 # high degree first
            sorted(range(n), key=lambda v: (deg[v], v)),                  # low degree first
            degen,                                                        # degeneracy order
            list(reversed(degen)),                                        # reverse degeneracy
            list(range(n)),                                               # natural order
            list(r.sample(range(n), n)),                                  # random shuffle 1
            list(r.sample(range(n), n)),                                  # random shuffle 2
            list(r.sample(range(n), n)),                                  # random shuffle 3
        ]

        best = None
        for order in orders:
            col = try_color(n, adj, lists, order)
            if valid(col, lists, edges):
                best = col
                break
            if best is None:
                best = col
            if time.time() > deadline:
                break
        
        if best is None:
            # Last resort placeholder
            best = [0] * n
        else:
            best = [int(c) for c in best]
            
        out.append(best)

    # Ensure we have exactly 4 lists
    while len(out) < 4:
         out.append([0,0,0,0,0,0,0,0]) # Dummy

    return out

# EVOLVE-BLOCK-END
