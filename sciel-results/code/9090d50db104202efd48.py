"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill

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
        Solves the list coloring instance using backtracking with
        MRV heuristic and forward checking with domain propagation.
        """
        if time.time() > deadline:
            return None

        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Pre-compute degrees for tie-breaking
        degrees = [len(adj[i]) for i in range(n)]

        # 1. Try greedy first (fast path)
        order = sorted(range(n), key=lambda x: -degrees[x])
        colour_greedy = [-1] * n
        success_greedy = True
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour_greedy[u] != -1:
                    taken.add(colour_greedy[u])
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
            return [int(c) for c in colour_greedy]

        # 2. Backtracking with MRV and Forward Checking
        colour = [-1] * n
        domains = [list(l) for l in lists]
        assigned = [False] * n
        
        # Count of each color in neighbors for quick domain filtering
        # We'll compute available colors on the fly for simplicity given small n
        
        def get_available(v):
            """Get available colors for vertex v given current assignments."""
            if assigned[v]:
                return list(domains[v])
            taken = set()
            for u in adj[v]:
                if assigned[u]:
                    taken.add(colour[u])
            return [c for c in domains[v] if c not in taken]
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find unassigned vertex with minimum available domain size (MRV)
            min_size = 10
            min_v = -1
            min_avail = []
            
            for v in range(n):
                if assigned[v]:
                    continue
                avail = get_available(v)
                size = len(avail)
                
                if size < min_size:
                    min_size = size
                    min_v = v
                    min_avail = avail
                    if size == 0:
                        return False
                    if size == 1:
                        break
                
            if min_v == -1:
                return True  # All assigned
            
            # Try each color in the domain of min_v
            for c in min_avail:
                colour[min_v] = c
                assigned[min_v] = True
                
                # Forward Checking: verify no neighbor has empty domain
                fc_ok = True
                for u in adj[min_v]:
                    if not assigned[u]:
                        avail_u = get_available(u)
                        if len(avail_u) == 0:
                            fc_ok = False
                            break
                
                if fc_ok and backtrack():
                    return True
                
                colour[min_v] = -1
                assigned[min_v] = False
            
            return False
        
        if backtrack():
            return [int(c) for c in colour]
        
        # 3. Fallback: try different orderings
        # Try reverse degree ordering
        order2 = sorted(range(n), key=lambda x: degrees[x])
        colour2 = [-1] * n
        success2 = True
        for v in order2:
            taken = set()
            for u in adj[v]:
                if colour2[u] != -1:
                    taken.add(colour2[u])
            valid_color = -1
            for c in lists[v]:
                if c not in taken:
                    valid_color = c
                    break
            if valid_color == -1:
                success2 = False
                break
            colour2[v] = valid_color
        
        if success2:
            return [int(c) for c in colour2]
        
        # Final fallback: return first color from each list
        return [int(lists[v][0]) for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5.0:
            break
        n, edges, lists = instance(splits, seed)
        colouring = solve_instance(n, edges, lists)
        if colouring is None:
            colouring = [int(lists[v][0]) for v in range(n)]
        colouring = [int(c) for c in colouring]
        out.append(colouring)
    
    # Pad if needed (should not happen)
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        out.append([int(lists[v][0]) for v in range(n)])
    
    return out
# EVOLVE-BLOCK-END
