"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    _start_time = time.time()
    _deadline = _start_time + 130  # safety margin before 150s kill

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

    def solve_backtrack(n, edges, lists):
        """Backtracking with MRV and forward checking."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Work with domains as sets
        domains = [set(lists[v]) for v in range(n)]
        assigned = [None] * n
        
        # MRV heuristic: pick unassigned vertex with smallest domain
        def get_mrv_vertex():
            best_v = -1
            best_size = float('inf')
            for v in range(n):
                if assigned[v] is None:
                    sz = len(domains[v])
                    if sz < best_size:
                        best_size = sz
                        best_v = v
                        if sz <= 1:
                            break
            return best_v
        
        def backtrack(depth):
            if time.time() > _deadline:
                return False
            if depth == n:
                return True
            
            v = get_mrv_vertex()
            if v == -1:
                return True
            
            if not domains[v]:
                return False
            
            # Try each colour in domain (smallest domain first already)
            for c in sorted(domains[v]):
                assigned[v] = c
                # Forward checking: remove c from neighbours' domains
                saved = []
                feasible = True
                for u in adj[v]:
                    if assigned[u] is not None:
                        if assigned[u] == c:
                            feasible = False
                            break
                    else:
                        if c in domains[u]:
                            saved.append((u, c))
                            domains[u].discard(c)
                            if not domains[u]:
                                feasible = False
                                break
                if feasible:
                    if backtrack(depth + 1):
                        return True
                # Undo
                for u, cc in saved:
                    domains[u].add(cc)
                assigned[v] = None
            
            return False
        
        # Make a copy of domains to avoid mutation issues across calls
        import copy
        domains_copy = [set(lists[v]) for v in range(n)]
        domains = domains_copy
        assigned = [None] * n
        
        if backtrack(0):
            return [int(assigned[v]) for v in range(n)]
        return None

    def solve_greedy_random(n, edges, lists, rng):
        """Greedy with random tie-breaking, multiple attempts."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        for _ in range(200):
            colour = [None] * n
            order = list(range(n))
            rng.shuffle(order)
            # Sort by degree descending as secondary
            order.sort(key=lambda u: -len(adj[u]))
            
            success = True
            for v in order:
                taken = {colour[u] for u in adj[v] if colour[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                if free:
                    colour[v] = rng.choice(free)
                else:
                    success = False
                    break
            if success:
                return [int(c) for c in colour]
        return None

    def solve_instance(n, edges, lists):
        # Try backtracking first (should be fast for these sizes)
        result = solve_backtrack(n, edges, lists)
        if result is not None:
            # Verify
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            for v in range(n):
                if result[v] not in lists[v]:
                    break
                for u in adj[v]:
                    if result[u] == result[v]:
                        break
            else:
                return result
            # If verification failed, fall through
        # Fallback: greedy with randomization
        rng = Random(12345)
        result = solve_greedy_random(n, edges, lists, rng)
        if result is not None:
            return result
        # Last resort: original greedy
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return [int(c) for c in colour]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > _deadline:
            # Emergency: return best we can
            break
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists)
        out.append(result)
    
    # If we ran out of time, fill remaining with greedy
    while len(out) < 4:
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[len(out)]
        n, edges, lists = instance(splits, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        out.append([int(c) for c in colour])
    
    return out
# EVOLVE-BLOCK-END
