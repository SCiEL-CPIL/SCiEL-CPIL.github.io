"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random
from collections import deque

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

    def get_bipartition(n, adj):
        """BFS to get bipartition of the graph."""
        color = [-1] * n
        parts = [[], []]
        for start in range(n):
            if color[start] != -1:
                continue
            color[start] = 0
            parts[0].append(start)
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        parts[color[v]].append(v)
                        q.append(v)
        return parts

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

    def bipartite_solve(n, adj, edges, lists, deadline):
        """Exploit bipartite structure: assign one side, then solve the other."""
        parts = get_bipartition(n, adj)
        # Choose smaller part to enumerate/backtrack on
        if len(parts[0]) <= len(parts[1]):
            small, large = parts[0], parts[1]
        else:
            small, large = parts[1], parts[0]
        
        small_set = set(small)
        large_set = set(large)
        
        # For each large vertex, find which small vertices are neighbors
        large_neighbors = {v: [u for u in adj[v] if u in small_set] for v in large}
        small_neighbors = {v: [u for u in adj[v] if u in large_set] for v in small}
        
        colour = [-1] * n
        sys.setrecursionlimit(200000)
        
        # Backtrack on small vertices
        def solve_small(idx):
            if time.time() > deadline:
                return False
            if idx == len(small):
                # Try to assign large vertices greedily
                for v in large:
                    used_colors = set()
                    for u in large_neighbors[v]:
                        used_colors.add(colour[u])
                    available = [c for c in lists[v] if c not in used_colors]
                    if not available:
                        return False
                    colour[v] = available[0]
                return True
            
            v = small[idx]
            for c in lists[v]:
                colour[v] = c
                # Check: for each large neighbor, does c block all options?
                feasible = True
                for u in small_neighbors[v]:
                    used = set()
                    for w in large_neighbors[u]:
                        if colour[w] != -1:
                            used.add(colour[w])
                    if all(c2 in used for c2 in lists[u]):
                        feasible = False
                        break
                if feasible:
                    if solve_small(idx + 1):
                        return True
                colour[v] = -1
            return False
        
        if solve_small(0):
            return [int(c) for c in colour]
        return None

    def backtrack_mrv(n, adj, lists, deadline):
        """Backtracking with MRV, forward checking."""
        colour = [-1] * n
        domain = [set(lists[v]) for v in range(n)]
        constraint_count = [0] * n

        sys.setrecursionlimit(200000)

        def solve():
            if time.time() > deadline:
                return None

            best_v = -1
            best_size = float('inf')
            best_tiebreak = -1
            for v in range(n):
                if colour[v] == -1:
                    sz = len(domain[v])
                    if sz < best_size:
                        best_size = sz
                        best_v = v
                        best_tiebreak = constraint_count[v]
                        if sz == 0:
                            return False
                    elif sz == best_size and constraint_count[v] > best_tiebreak:
                        best_v = v
                        best_tiebreak = constraint_count[v]

            if best_v == -1:
                return True

            v = best_v
            colour_order = sorted(
                domain[v],
                key=lambda c: sum(1 for u in adj[v] if colour[u] == -1 and c in domain[u]),
                reverse=False
            )

            for c in colour_order:
                colour[v] = c
                constraint_count[v] = len(adj[v])
                saved = []
                ok = True
                for u in adj[v]:
                    if colour[u] == -1 and c in domain[u]:
                        domain[u].discard(c)
                        saved.append((u, c))
                        constraint_count[u] += 1
                        if len(domain[u]) == 0:
                            ok = False
                            break

                if ok:
                    result = solve()
                    if result is not None:
                        return result

                colour[v] = -1
                constraint_count[v] = 0
                for u, c2 in saved:
                    domain[u].add(c2)
                    constraint_count[u] -= 1

            return False

        result = solve()
        if result:
            return [int(c) for c in colour]
        return None

    def local_search(n, adj, edges, lists, deadline, rng_seed):
        """Simulated annealing with incremental conflict tracking."""
        rng = Random(rng_seed)
        colour = [rng.choice(lists[v]) for v in range(n)]
        
        # Precompute conflict count per vertex
        conflict_count = [0] * n
        total_conflicts = 0
        for u, v in edges:
            if colour[u] == colour[v]:
                conflict_count[u] += 1
                conflict_count[v] += 1
                total_conflicts += 1
        
        if total_conflicts == 0:
            return list(colour)
        
        best_colour = list(colour)
        best_conflicts = total_conflicts
        current_conflicts = total_conflicts
        T = 1.0
        cooling = 0.9995
        max_iter = 300000
        start_t = time.time()
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random vertex with conflicts, or random
            conflicted = [v for v in range(n) if conflict_count[v] > 0]
            if conflicted:
                v = rng.choice(conflicted)
            else:
                v = rng.randrange(n)
            
            old_c = colour[v]
            candidates = [c for c in lists[v] if c != old_c]
            if not candidates:
                continue
            c = rng.choice(candidates)
            
            # Compute delta
            delta = 0
            for u in adj[v]:
                if colour[u] == old_c:
                    delta += 1
                if colour[u] == c:
                    delta -= 1
            
            if delta <= 0 or rng.random() < (10 ** (-current_conflicts / max(T, 0.01))):
                colour[v] = c
                # Update conflict counts
                for u in adj[v]:
                    if colour[u] == old_c:
                        conflict_count[u] -= 1
                        conflict_count[v] -= 1
                    if colour[u] == c:
                        conflict_count[u] += 1
                        conflict_count[v] += 1
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
        # Strategy 1: Exploit bipartite structure
        result = bipartite_solve(n, adj, edges, lists, deadline)
        if result is not None and is_valid(result, n, edges, lists):
            return result
        
        # Strategy 2: Backtracking with MRV
        result = backtrack_mrv(n, adj, lists, deadline)
        if result is not None and is_valid(result, n, edges, lists):
            return result
        
        # Strategy 3: Local search with multiple restarts
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
    deadline = start_time + 110  # Safety margin before 120s

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)
        
        result = solve_instance(n, adj, edges, lists, deadline)
        
        # Ensure types are correct
        result = [int(c) for c in result]
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
