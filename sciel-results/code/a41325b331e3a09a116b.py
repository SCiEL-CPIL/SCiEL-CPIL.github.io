"""Model-and-optimize: Robust list coloring via Min-Conflicts with backtracking fallback."""
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

    def solve_with_min_conflicts(n, edges, lists, adj, rng, max_iterations=2000):
        """Min-Conflicts algorithm with proper conflict tracking."""
        # Random initial assignment
        colour = [rng.choice(lists[v]) for v in range(n)]
        
        # Count conflicts per vertex
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
            
            # Find all conflicting vertices
            conflicting = [v for v in range(n) if conflict_count[v] > 0]
            if not conflicting:
                break
            
            # Pick a random conflicting vertex
            v = rng.choice(conflicting)
            
            # Find best color for v (minimizes conflicts)
            # Count how many neighbors have each color
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
            
            # If no improvement, pick randomly to escape local minimum
            if best_c == colour[v]:
                best_c = rng.choice(lists[v])
            
            if best_c == colour[v]:
                continue
            
            # Update: remove old conflicts, add new ones
            old_c = colour[v]
            
            # Decrease conflict counts for neighbors with old_c
            for u in adj[v]:
                if colour[u] == old_c:
                    conflict_count[u] -= 1
            
            # Set new color
            colour[v] = best_c
            
            # Count new conflicts for v and update neighbors
            new_conflicts = 0
            for u in adj[v]:
                if colour[u] == best_c:
                    conflict_count[u] += 1
                    new_conflicts += 1
            conflict_count[v] = new_conflicts
            
            # Update total_conflicts
            # Old contribution: conflict_count[v] before change (which was best_cost)
            # New contribution: new_conflicts
            # But we also changed neighbors' counts
            # Simpler: just recompute or track delta carefully
            # For now, we'll check solved status by scanning
            pass
        
        # Check if solved
        if all(conflict_count[v] == 0 for v in range(n)):
            return colour
        
        return None

    def solve_with_greedy_backtrack(n, edges, lists, adj, rng, max_attempts=50):
        """Greedy coloring with randomized restarts and degree-based ordering."""
        # Precompute degree for ordering
        degrees = [len(adj[v]) for v in range(n)]
        
        for attempt in range(max_attempts):
            if deadline():
                break
            
            if attempt < 5:
                # First few attempts: use degree-based ordering (high degree first)
                order = sorted(range(n), key=lambda v: -degrees[v])
            else:
                # Remaining attempts: random order
                order = list(range(n))
                rng.shuffle(order)
            
            colour = [None] * n
            valid = True
            
            for v in order:
                # Collect colors used by already-colored neighbors
                taken = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        taken.add(colour[u])
                
                # Find available colors
                avail = [c for c in lists[v] if c not in taken]
                if avail:
                    colour[v] = rng.choice(avail)
                else:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify it's a proper coloring
            is_valid = True
            for u, v in edges:
                if colour[u] == colour[v]:
                    is_valid = False
                    break
            
            if is_valid:
                return colour
        
        return None

    def solve_with_backtracking(n, edges, lists, adj, rng, max_nodes=10000):
        """Backtracking solver for hard instances."""
        # Order vertices by degree (high first) for better pruning
        order = sorted(range(n), key=lambda v: -len(adj[v]))
        pos_in_order = {v: i for i, v in enumerate(order)}
        
        colour = [None] * n
        nodes_visited = [0]
        
        def bt(idx):
            if nodes_visited[0] > max_nodes:
                return False
            if idx == n:
                return True
            nodes_visited[0] += 1
            
            v = order[idx]
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            
            for c in lists[v]:
                if c not in taken:
                    colour[v] = c
                    if bt(idx + 1):
                        return True
                    colour[v] = None
            
            return False
        
        if bt(0):
            return colour
        return None

    def solve_instance(n, edges, lists):
        """Try multiple strategies to solve the instance."""
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        rng = Random(42)
        
        # Strategy 1: Greedy with degree ordering (fast, often sufficient)
        result = solve_with_greedy_backtrack(n, edges, lists, adj, rng, max_attempts=200)
        if result is not None:
            return result
        
        # Strategy 2: Backtracking (exact, for small/medium instances)
        if n <= 200:
            result = solve_with_backtracking(n, edges, lists, adj, rng, max_nodes=50000)
            if result is not None:
                return result
        
        # Strategy 3: Min-Conflicts
        result = solve_with_min_conflicts(n, edges, lists, adj, rng, max_iterations=5000)
        if result is not None:
            return result
        
        # Strategy 4: More aggressive Min-Conflicts with different seeds
        for seed in range(10):
            if deadline():
                break
            rng2 = Random(1000 + seed)
            result = solve_with_min_conflicts(n, edges, lists, adj, rng2, max_iterations=3000)
            if result is not None:
                return result
        
        # Fallback: return a random assignment (shouldn't happen for valid instances)
        rng3 = Random(999)
        return [rng3.choice(lists[v]) for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            # If we run out of time, we still need to return something
            # Generate a random coloring as fallback
            n, edges, lists = instance(splits, seed)
            rng_fallback = Random(777)
            out.append([rng_fallback.choice(lists[v]) for v in range(n)])
            continue
        
        n, edges, lists = instance(splits, seed)
        colour = solve_instance(n, edges, lists)
        
        # Ensure types are Python int
        out.append([int(c) for c in colour])

    return out
# EVOLVE-BLOCK-END
