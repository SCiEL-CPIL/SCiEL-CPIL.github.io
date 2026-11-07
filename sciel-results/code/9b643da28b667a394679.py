"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

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

    def solve_instance(n, edges, lists, deadline):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Degree for variable ordering tie-breaking
        degree = [len(adj[i]) for i in range(n)]

        # Initialize domains as bitmasks
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
        
        assignment = [-1] * n
        sys.setrecursionlimit(10000)

        def popcount(x):
            return bin(x).count('1')

        # AC-3 with bitmasks
        def arc_consistency(initial_queue=None):
            if initial_queue is None:
                queue = deque()
                for u in range(n):
                    for v in adj[u]:
                        queue.append((v, u))
            else:
                queue = deque(initial_queue)
            
            while queue:
                if time.time() > deadline:
                    return 'timeout'
                
                xi, xj = queue.popleft()
                
                if domains[xj] == 0:
                    return False

                # If Dom(xj) is a singleton, remove that color from Dom(xi)
                if domains[xj] & (domains[xj] - 1) == 0:
                    c_bit = domains[xj]
                    if domains[xi] & c_bit:
                        new_dom = domains[xi] & ~c_bit
                        if new_dom != domains[xi]:
                            domains[xi] = new_dom
                            if domains[xi] == 0:
                                return False
                            for xk in adj[xi]:
                                if xk != xj:
                                    queue.append((xk, xi))
            
            return True

        # Backtracking with snapshot-based undo
        def backtrack():
            if time.time() > deadline:
                return 'timeout'
            
            # Find MRV
            best_v = -1
            best_count = float('inf')
            best_deg = -1
            for v in range(n):
                if assignment[v] == -1:
                    count = popcount(domains[v])
                    if count == 0:
                        return False
                    if count < best_count or (count == best_count and degree[v] > best_deg):
                        best_count = count
                        best_v = v
                        best_deg = degree[v]
            
            if best_v == -1:
                return True
            
            dom = domains[best_v]
            
            # Extract colors
            colors = []
            temp = dom
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                colors.append(c)
                temp ^= lsb
            
            for c in colors:
                if time.time() > deadline:
                    return 'timeout'
                
                lsb = 1 << c
                
                # Snapshot
                old_domains = domains[:]
                old_assignment = assignment[best_v]
                
                # Assign
                assignment[best_v] = c
                domains[best_v] = lsb
                
                # Propagate: remove color from unassigned neighbors
                valid = True
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        if domains[u] & lsb:
                            new_dom_u = domains[u] & ~lsb
                            if new_dom_u == 0:
                                valid = False
                                break
                            domains[u] = new_dom_u
                
                if valid:
                    affected = [(u, best_v) for u in adj[best_v] if assignment[u] == -1]
                    ac_res = arc_consistency(affected)
                    
                    if ac_res is True:
                        result = backtrack()
                        if result is True:
                            return True
                        if result == 'timeout':
                            domains[:] = old_domains
                            assignment[best_v] = old_assignment
                            return 'timeout'
                    elif ac_res == 'timeout':
                        domains[:] = old_domains
                        assignment[best_v] = old_assignment
                        return 'timeout'
                
                # Undo
                domains[:] = old_domains
                assignment[best_v] = old_assignment
            
            return False

        # Initial AC-3
        ac_result = arc_consistency()
        if ac_result is not True:
            return None

        result = backtrack()
        if result is True:
            return [int(a) for a in assignment]
        return None

    out = []
    start_time = time.time()
    hard_deadline = start_time + 140.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        if time.time() > hard_deadline - 5.0:
             out.append([0] * n)
             continue
        
        remaining_instances = 4 - len(out)
        time_left = hard_deadline - time.time()
        instance_deadline = time.time() + (time_left / remaining_instances) * 0.95
        
        colouring = solve_instance(n, edges, lists, instance_deadline)
        
        if colouring is None:
            colouring = [0] * n
        
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
