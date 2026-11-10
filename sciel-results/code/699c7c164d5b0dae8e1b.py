"""Model-and-optimize: Robust list coloring via AC-3 + Backtracking with Min-Conflicts fallback."""
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

    def ac3(n, adj, domains):
        """AC-3 arc consistency. Returns True if no domain becomes empty."""
        queue = [(u, v) for u in range(n) for v in adj[u]]
        while queue:
            (x, y) = queue.pop()
            if revise(domains, x, y):
                if not domains[x]:
                    return False
                for k in adj[x]:
                    if k != y:
                        queue.append((k, x))
        return True

    def revise(domains, x, y):
        """Revise domain of x with respect to y. Returns True if domain was revised."""
        revised = False
        for c in list(domains[x]):
            # Check if there's any value in domains[y] that differs from c
            if not any(d != c for d in domains[y]):
                domains[x].remove(c)
                revised = True
        return revised

    def backtrack(n, adj, domains, assignment, rng):
        """Backtracking search with forward checking. Returns a valid coloring or None."""
        # Find unassigned variable with smallest domain (MRV heuristic)
        unassigned = [v for v in range(n) if assignment[v] is None]
        if not unassigned:
            return assignment
        
        # MRV: pick variable with smallest domain
        v = min(unassigned, key=lambda x: len(domains[x]))
        
        for c in domains[v]:
            # Check consistency with already assigned neighbors
            consistent = True
            for u in adj[v]:
                if assignment[u] is not None and assignment[u] == c:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Forward checking: check if assigning c to v makes any neighbor's domain empty
            new_domains = [list(d) for d in domains]
            new_assignment = list(assignment)
            new_assignment[v] = c
            
            # Remove c from neighbors' domains
            fc_ok = True
            for u in adj[v]:
                if assignment[u] is None and c in new_domains[u]:
                    new_domains[u].remove(c)
                    if not new_domains[u]:
                        fc_ok = False
                        break
            
            if not fc_ok:
                continue
            
            # Recurse
            result = backtrack(n, adj, new_domains, new_assignment, rng)
            if result is not None:
                return result
        
        return None

    def solve_ac3_backtrack(n, edges, lists, rng, max_time=30.0):
        """Solve using AC-3 + backtracking with forward checking."""
        start_time = time.time()
        
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initialize domains from lists
        domains = [list(lists[v]) for v in range(n)]
        
        # Run AC-3
        if not ac3(n, adj, domains):
            return None
        
        # Check if any domain is empty after AC-3
        for v in range(n):
            if not domains[v]:
                return None
        
        # Try backtracking with forward checking
        assignment = [None] * n
        result = backtrack(n, adj, domains, assignment, rng)
        
        if result is not None:
            return result
        
        # If backtracking failed (shouldn't happen for solvable instances), try with different orderings
        for _ in range(10):
            if time.time() - start_time > max_time or deadline():
                break
            # Shuffle domains to try different orderings
            shuffled_domains = [list(d) for d in domains]
            for v in range(n):
                rng.shuffle(shuffled_domains[v])
            result = backtrack(n, adj, shuffled_domains, assignment, rng)
            if result is not None:
                return result
        
        return None

    def solve_min_conflicts(n, edges, lists, adj, rng, max_iterations=5000):
        """Min-Conflicts algorithm with proper conflict tracking."""
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
                best_c = rng.choice(lists[v])
            
            if best_c == colour[v]:
                continue
            
            old_c = colour[v]
            
            for u in adj[v]:
                if colour[u] == old_c:
                    conflict_count[u] -= 1
            
            colour[v] = best_c
            
            new_conflicts = 0
            for u in adj[v]:
                if colour[u] == best_c:
                    conflict_count[u] += 1
                    new_conflicts += 1
            conflict_count[v] = new_conflicts
            
            total_conflicts = sum(conflict_count) // 2
        
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
        
        # Strategy 1: AC-3 + Backtracking (most reliable for small solvable CSPs)
        result = solve_ac3_backtrack(n, edges, lists, rng, max_time=45.0)
        if result is not None:
            return result
        
        # Strategy 2: Min-Conflicts
        result = solve_min_conflicts(n, edges, lists, adj, rng, max_iterations=5000)
        if result is not None:
            return result
        
        # Strategy 3: More aggressive Min-Conflicts with different seeds
        for seed in range(20):
            if deadline():
                break
            rng2 = Random(1000 + seed)
            result = solve_min_conflicts(n, edges, lists, adj, rng2, max_iterations=3000)
            if result is not None:
                return result
        
        # Fallback: greedy with random restarts
        for attempt in range(50):
            if deadline():
                break
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
            if valid:
                is_valid = True
                for u, v in edges:
                    if colour[u] == colour[v]:
                        is_valid = False
                        break
                if is_valid:
                    return colour
        
        # Last resort: random assignment
        rng3 = Random(999)
        return [rng3.choice(lists[v]) for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            n, edges, lists = instance(splits, seed)
            rng_fallback = Random(777)
            out.append([rng_fallback.choice(lists[v]) for v in range(n)])
            continue
        
        n, edges, lists = instance(splits, seed)
        colour = solve_instance(n, edges, lists)
        
        out.append([int(c) for c in colour])

    return out
# EVOLVE-BLOCK-END
