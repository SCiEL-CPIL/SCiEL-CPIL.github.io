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
        
        degree = [len(adj[i]) for i in range(n)]
        assignment = [-1] * n
        sys.setrecursionlimit(10000)

        def get_available_colors(v):
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
            
            # MRV heuristic
            best_v = -1
            best_count = float('inf')
            
            for v in range(n):
                if assignment[v] == -1:
                    count = 0
                    for c in lists[v]:
                        conflict = False
                        for u in adj[v]:
                            if assignment[u] == c:
                                conflict = True
                                break
                        if not conflict:
                            count += 1
                            if count == 3:
                                break
                    if count < best_count:
                        best_count = count
                        best_v = v
                        if count == 0:
                            return False
                        if count == 1:
                            pass

            if best_v == -1:
                return True
            
            avail_colors = get_available_colors(best_v)
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

    def solve_with_arc_consistency(n, edges, lists, deadline, rng):
        """Solve using AC-3 propagation combined with backtracking."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        degree = [len(adj[i]) for i in range(n)]
        
        # Domains as sets for efficient arc consistency
        domains = [set(lists[i]) for i in range(n)]
        
        def ac3(revise_queue=None):
            """Arc consistency propagation. Returns False if a domain becomes empty."""
            if revise_queue is None:
                revise_queue = deque()
                for u, v in edges:
                    revise_queue.append((u, v))
                    revise_queue.append((v, u))
            
            while revise_queue:
                if time.time() > deadline:
                    return 'timeout'
                
                xi, xj = revise_queue.popleft()
                
                # Check if xi's domain can be reduced based on xj's domain
                revised = False
                new_domain = set()
                for ci in domains[xi]:
                    # Check if ci has a support in xj's domain
                    has_support = False
                    for cj in domains[xj]:
                        if ci != cj:
                            has_support = True
                            break
                    if has_support:
                        new_domain.add(ci)
                
                if len(new_domain) < len(domains[xi]):
                    revised = True
                    domains[xi] = new_domain
                    
                    if not domains[xi]:
                        return False
                    
                    # Add arcs (xk, xi) for all neighbors xk of xi (except xj)
                    for xk in adj[xi]:
                        if xk != xj:
                            revise_queue.append((xk, xi))
            
            return True
        
        # Initial arc consistency
        ac3_result = ac3()
        if ac3_result is False:
            return None
        if ac3_result == 'timeout':
            return None
        
        # Now do backtracking with forward checking
        assignment = [-1] * n
        
        def get_available(v):
            """Get available colors for v given current assignment."""
            avail = []
            for c in domains[v]:
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
            
            # MRV
            best_v = -1
            best_count = float('inf')
            
            for v in range(n):
                if assignment[v] == -1:
                    count = len(get_available(v))
                    if count < best_count:
                        best_count = count
                        best_v = v
                        if count == 0:
                            return False
                        if count == 1:
                            pass
            
            if best_v == -1:
                return True
            
            avail = get_available(best_v)
            if len(avail) > 1:
                rng.shuffle(avail)
            
            for c in avail:
                if time.time() > deadline:
                    return 'timeout'
                
                assignment[best_v] = c
                
                # Forward checking: check if any unassigned neighbor has empty domain
                fc_failed = False
                for u in adj[best_v]:
                    if assignment[u] == -1:
                        if not get_available(u):
                            fc_failed = True
                            break
                
                if not fc_failed:
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
    hard_deadline = start_time + 140.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        remaining_instances = 4 - len(out)
        time_left = hard_deadline - time.time()
        instance_deadline = time.time() + (time_left / remaining_instances) * 0.95
        
        colouring = None
        
        # Try arc consistency + backtracking first
        for retry_seed in range(5):
            if time.time() > instance_deadline:
                break
            rng_solver = Random(retry_seed)
            colouring = solve_with_arc_consistency(n, edges, lists, instance_deadline, rng_solver)
            if colouring is not None:
                break
        
        # If that didn't work, try plain backtracking
        if colouring is None:
            for retry_seed in range(10):
                if time.time() > instance_deadline:
                    break
                rng_solver = Random(retry_seed + 100)
                colouring = solve_instance(n, edges, lists, instance_deadline, rng_solver)
                if colouring is not None:
                    break
        
        if colouring is None:
            colouring = [0] * n
        
        colouring = [int(c) for c in colouring]
        out.append(colouring)

    return out

# EVOLVE-BLOCK-END
