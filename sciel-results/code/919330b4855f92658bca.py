"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
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

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def is_valid(colour, n, edges, lists):
        if not isinstance(colour, (list, tuple)):
            return False
        if len(colour) != n:
            return False
        for i, c in enumerate(colour):
            if not isinstance(c, int) or isinstance(c, bool):
                return False
            if c not in lists[i]:
                return False
        for u, v in edges:
            if colour[u] == colour[v]:
                return False
        return True

    def backtrack_mrv(n, adj, lists, deadline):
        """Backtracking with MRV, forward checking, and dynamic variable ordering."""
        colour = [-1] * n
        domain = [set(lists[v]) for v in range(n)]
        unassigned = set(range(n))
        # Precompute adjacency as sets for faster lookups
        adj_set = [set(a) for a in adj]

        sys.setrecursionlimit(200000)

        def solve():
            if time.time() > deadline:
                return None

            if not unassigned:
                return True

            # Find unassigned vertex with minimum remaining domain (MRV)
            best_v = -1
            best_size = float('inf')
            best_deg = -1
            for v in unassigned:
                sz = len(domain[v])
                if sz < best_size:
                    best_size = sz
                    best_v = v
                    best_deg = len(adj[v])
                    if sz == 0:
                        return False
                elif sz == best_size and len(adj[v]) > best_deg:
                    best_v = v
                    best_deg = len(adj[v])

            v = best_v
            # Try colours in order of least constraining value
            unassigned_neighbours = [u for u in adj[v] if colour[u] == -1]
            colour_order = sorted(
                domain[v],
                key=lambda c: sum(1 for u in unassigned_neighbours if c in domain[u])
            )

            for c in colour_order:
                colour[v] = c
                saved = []
                ok = True
                for u in unassigned_neighbours:
                    if c in domain[u]:
                        domain[u].discard(c)
                        saved.append((u, c))
                        if len(domain[u]) == 0:
                            ok = False
                            break

                if ok:
                    unassigned.discard(v)
                    result = solve()
                    if result is not None:
                        return result
                    unassigned.add(v)
                else:
                    pass

                # Undo
                colour[v] = -1
                for u, c2 in saved:
                    domain[u].add(c2)

            return False

        result = solve()
        if result:
            return [int(c) for c in colour]
        return None

    def greedy_with_repair(n, adj, edges, lists, deadline, rng_seed):
        """Greedy assignment with random restarts and repair."""
        rng = Random(rng_seed)
        # Order vertices by degree descending
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            colour = [-1] * n
            success = True
            
            for v in order:
                # Filter domain by assigned neighbours
                neighbour_colors = set()
                for u in adj[v]:
                    if colour[u] != -1:
                        neighbour_colors.add(colour[u])
                
                available = [c for c in lists[v] if c not in neighbour_colors]
                
                if not available:
                    success = False
                    break
                
                # Pick a colour that minimizes impact on unassigned neighbours
                best_c = available[0]
                best_impact = float('inf')
                for c in available:
                    impact = 0
                    for u in adj[v]:
                        if colour[u] == -1 and c in lists[u]:
                            impact += 1
                    if impact < best_impact:
                        best_impact = impact
                        best_c = c
                
                colour[v] = best_c
            
            if success:
                # Verify
                valid = True
                for u, v in edges:
                    if colour[u] == colour[v]:
                        valid = False
                        break
                if valid:
                    return [int(c) for c in colour]
        
        return None

    def local_search(n, adj, edges, lists, deadline, rng_seed):
        """Simulated annealing / hill climbing with random restarts."""
        rng = Random(rng_seed)
        # Start with random valid-list assignment
        colour = [rng.choice(lists[v]) for v in range(n)]
        
        # Precompute conflict count per vertex
        conflict_count = [0] * n
        total_conflicts = 0
        for u, v in edges:
            if colour[u] == colour[v]:
                conflict_count[u] += 1
                conflict_count[v] += 1
                total_conflicts += 1
        
        best_colour = list(colour)
        best_conflicts = total_conflicts
        
        if best_conflicts == 0:
            return best_colour
        
        current_conflicts = best_conflicts
        T = 1.0
        cooling = 0.9995
        max_iter = 200000
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random vertex in conflict, or random vertex
            in_conflict = [v for v in range(n) if conflict_count[v] > 0]
            if in_conflict:
                v = rng.choice(in_conflict)
            else:
                v = rng.randrange(n)
            
            old_c = colour[v]
            # Try a different colour from the list
            candidates = [c for c in lists[v] if c != old_c]
            if not candidates:
                continue
            c = rng.choice(candidates)
            
            # Compute delta in conflicts
            delta = 0
            for u in adj[v]:
                if colour[u] == old_c:
                    delta += 1  # was conflicting, now not
                if colour[u] == c:
                    delta -= 1  # was not conflicting, now is
            
            if delta <= 0 or (T > 0.001 and rng.random() < (10 ** (-current_conflicts / max(T, 0.01)))):
                # Apply flip
                colour[v] = c
                for u in adj[v]:
                    if colour[u] == old_c:
                        conflict_count[u] -= 1
                    if colour[u] == c:
                        conflict_count[u] += 1
                # Update v's own conflict count
                new_vc = 0
                for u in adj[v]:
                    if colour[u] == c:
                        new_vc += 1
                conflict_count[v] = new_vc
                current_conflicts += delta
                if current_conflicts < best_conflicts:
                    best_conflicts = current_conflicts
                    best_colour = list(colour)
                    if best_conflicts == 0:
                        return best_colour
            
            T *= cooling
            if T < 0.001:
                break
        
        return best_colour if best_conflicts == 0 else None

    def solve_instance(n, adj, edges, lists, deadline):
        """Try multiple strategies in sequence."""
        # Strategy 1: Quick greedy (fast, may succeed)
        if time.time() < deadline - 5:
            result = greedy_with_repair(n, adj, edges, lists, deadline, 42)
            if result is not None and is_valid(result, n, edges, lists):
                return result
        
        # Strategy 2: Backtracking with MRV (reliable)
        if time.time() < deadline - 2:
            result = backtrack_mrv(n, adj, lists, deadline)
            if result is not None and is_valid(result, n, edges, lists):
                return result
        
        # Strategy 3: Greedy with multiple restarts
        for seed in [123, 456, 789, 1000]:
            if time.time() > deadline:
                break
            result = greedy_with_repair(n, adj, edges, lists, deadline, seed)
            if result is not None and is_valid(result, n, edges, lists):
                return result
        
        # Strategy 4: Local search with multiple restarts
        for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
            if time.time() > deadline:
                break
            result = local_search(n, adj, edges, lists, deadline, seed)
            if result is not None and is_valid(result, n, edges, lists):
                return result
        
        # Last resort: return any list-valid assignment (may not be proper)
        return [lists[v][0] for v in range(n)]

    out = []
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)
        
        result = solve_instance(n, adj, edges, lists, deadline)
        
        # Ensure types are correct
        result = [int(c) for c in result]
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
