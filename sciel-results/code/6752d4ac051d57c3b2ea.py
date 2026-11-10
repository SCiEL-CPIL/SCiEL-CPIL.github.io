"""Model-and-optimize: Robust list coloring via backtracking (MRV+FC) with greedy and min-conflicts fallback."""
# EVOLVE-BLOCK-START
import time
from random import Random


def construct():
    t0 = time.time()
    LIMIT = 120.0
    MARGIN = 15.0

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
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_greedy_degree(n, edges, lists, adj, rng, max_attempts=100):
        """Greedy coloring with degree-based ordering and randomized restarts."""
        degrees = [len(adj[v]) for v in range(n)]
        
        for attempt in range(max_attempts):
            if deadline():
                break
            
            if attempt < 10:
                order = sorted(range(n), key=lambda v: (-degrees[v], v))
            else:
                order = list(range(n))
                rng.shuffle(order)
            
            colour = [None] * n
            valid = True
            
            for v in order:
                taken = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        taken.add(colour[u])
                
                avail = [c for c in lists[v] if c not in taken]
                if avail:
                    colour[v] = rng.choice(avail)
                else:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify
            for u, v in edges:
                if colour[u] == colour[v]:
                    valid = False
                    break
            
            if valid:
                return colour
        
        return None

    def solve_backtracking_mrv(n, edges, lists, adj, rng, max_nodes=200000):
        """Backtracking with MRV heuristic and forward checking."""
        # Precompute degree for initial ordering
        degrees = [len(adj[v]) for v in range(n)]
        
        # Use a recursive backtracking with MRV + forward checking
        colour = [None] * n
        nodes_visited = [0]
        found = [False]
        
        def get_domains(colour_state):
            """Get remaining domain for each uncolored vertex."""
            domains = {}
            for v in range(n):
                if colour_state[v] is not None:
                    continue
                taken = set()
                for u in adj[v]:
                    if colour_state[u] is not None:
                        taken.add(colour_state[u])
                avail = [c for c in lists[v] if c not in taken]
                domains[v] = avail
            return domains
        
        def bt(colour_state, uncolored):
            if nodes_visited[0] > max_nodes:
                return False
            if not uncolored:
                return True
            nodes_visited[0] += 1
            
            # MRV: pick uncolored vertex with smallest domain
            best_v = None
            best_domain = None
            for v in uncolored:
                taken = set()
                for u in adj[v]:
                    if colour_state[u] is not None:
                        taken.add(colour_state[u])
                avail = [c for c in lists[v] if c not in taken]
                if not avail:
                    return False  # Dead end
                if best_domain is None or len(avail) < len(best_domain):
                    best_v = v
                    best_domain = avail
                    if len(avail) == 1:
                        break
            
            if best_v is None:
                return False
            
            new_uncolored = [v for v in uncolored if v != best_v]
            
            # Try colors in best_domain (shuffle for variety on ties)
            domain = list(best_domain)
            if len(domain) > 1:
                rng.shuffle(domain)
            
            for c in domain:
                colour_state[best_v] = c
                if bt(colour_state, new_uncolored):
                    return True
                colour_state[best_v] = None
            
            return False
        
        uncolored = list(range(n))
        if bt(colour, uncolored):
            return colour
        return None

    def solve_min_conflicts(n, edges, lists, adj, rng, max_iterations=10000):
        """Min-Conflicts with proper conflict tracking."""
        colour = [rng.choice(lists[v]) for v in range(n)]
        
        conflict_count = [0] * n
        for u, v in edges:
            if colour[u] == colour[v]:
                conflict_count[u] += 1
                conflict_count[v] += 1
        
        total_conflicts = sum(conflict_count) // 2
        
        if total_conflicts == 0:
            return colour
        
        for _ in range(max_iterations):
            if deadline():
                break
            if total_conflicts == 0:
                break
            
            conflicting = [v for v in range(n) if conflict_count[v] > 0]
            if not conflicting:
                break
            
            v = rng.choice(conflicting)
            
            # Find best color
            neighbor_color_count = {}
            for u in adj[v]:
                c = colour[u]
                neighbor_color_count[c] = neighbor_color_count.get(c, 0) + 1
            
            best_c = colour[v]
            best_cost = conflict_count[v]
            
            for c in lists[v]:
                if c == colour[v]:
                    continue
                new_cost = neighbor_color_count.get(c, 0)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_c = c
            
            if best_c == colour[v]:
                # Random escape
                best_c = rng.choice(lists[v])
                if best_c == colour[v]:
                    continue
            
            old_c = colour[v]
            
            # Update: remove old conflicts
            for u in adj[v]:
                if colour[u] == old_c:
                    conflict_count[u] -= 1
            
            # Set new color
            colour[v] = best_c
            
            # Add new conflicts
            new_conflicts = 0
            for u in adj[v]:
                if colour[u] == best_c:
                    conflict_count[u] += 1
                    new_conflicts += 1
            conflict_count[v] = new_conflicts
            
            # Update total: delta = (new_conflicts - old_conflict_count_v)
            # old_conflict_count_v was best_cost (the count before change)
            # But we need to be careful: total_conflicts counts each edge once
            # When we change v from old_c to best_c:
            #   - Edges (v,u) where colour[u]==old_c: these were conflicts, now not
            #   - Edges (v,u) where colour[u]==best_c: these are new conflicts
            # The number of old conflicts involving v = number of neighbors with old_c
            # The number of new conflicts involving v = new_conflicts
            # total_conflicts changes by: new_conflicts - (number of neighbors with old_c)
            # But "number of neighbors with old_c" = old conflict_count[v] BEFORE we zeroed it
            # We saved best_cost which was the old conflict_count[v]
            # Actually best_cost was the minimum cost found, not necessarily the old count
            # Let's just recompute total_conflicts periodically or track carefully
            
            # Simpler: just check solved at the end of each iteration
            pass
        
        if all(conflict_count[v] == 0 for v in range(n)):
            return colour
        return None

    def solve_instance(n, edges, lists):
        """Try multiple strategies to solve the instance."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        rng = Random(42)
        
        # Strategy 1: Backtracking with MRV (exact, fast for these sizes)
        result = solve_backtracking_mrv(n, edges, lists, adj, rng, max_nodes=200000)
        if result is not None:
            return result
        
        # Strategy 2: Greedy with degree ordering
        result = solve_greedy_degree(n, edges, lists, adj, rng, max_attempts=200)
        if result is not None:
            return result
        
        # Strategy 3: Min-Conflicts
        result = solve_min_conflicts(n, edges, lists, adj, rng, max_iterations=10000)
        if result is not None:
            return result
        
        # Strategy 4: More min-conflicts restarts
        for seed in range(20):
            if deadline():
                break
            rng2 = Random(1000 + seed)
            result = solve_min_conflicts(n, edges, lists, adj, rng2, max_iterations=5000)
            if result is not None:
                return result
        
        # Fallback
        rng3 = Random(999)
        return [rng3.choice(lists[v]) for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            n, edges, lists = instance(splits, seed)
            rng_fallback = Random(777)
            out.append([int(rng_fallback.choice(lists[v])) for v in range(n)])
            continue
        
        n, edges, lists = instance(splits, seed)
        colour = solve_instance(n, edges, lists)
        out.append([int(c) for c in colour])

    return out
# EVOLVE-BLOCK-END
