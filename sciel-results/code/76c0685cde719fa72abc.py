"""Strong edge colourings via multi-restart greedy + class-elimination local search."""
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflicts_bitmask(edges, n):
        """
        Precompute conflict bitmasks for all edges.
        Two edges conflict if they share a vertex or are joined by an edge (distance 1 in line graph).
        Returns list of integers where bit j is set if edge i conflicts with edge j.
        """
        edge_index = {e: i for i, e in enumerate(edges)}
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        n_edges = len(edges)
        conflicts = [0] * n_edges
        
        for i, (u, v) in enumerate(edges):
            near_vertices = {u, v}
            near_vertices.update(adj[u])
            near_vertices.update(adj[v])
            
            for x in near_vertices:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i] |= (1 << j)
                            conflicts[j] |= (1 << i)
                            
        return conflicts

    def greedy_color(order, conflicts, n_edges):
        """Color edges in given order using first-fit with bitmask conflicts."""
        color = [-1] * n_edges
        color_to_edges = []
        
        for idx in order:
            used_mask = 0
            for c_idx, edges_in_color in enumerate(color_to_edges):
                if any(conflicts[idx] & (1 << e) for e in edges_in_color):
                    used_mask |= (1 << c_idx)
            
            c = 0
            while (used_mask >> c) & 1:
                c += 1
            
            color[idx] = c
            if c >= len(color_to_edges):
                color_to_edges.append(set())
            color_to_edges[c].add(idx)
            
        return color, len(color_to_edges)

    def try_eliminate_colors(color, conflicts, n_edges):
        """
        Try to reduce the number of colors by eliminating color classes.
        Returns (new_color, new_n_colors).
        """
        if not color:
            return color, 0
            
        n_colors = max(color) + 1
        if n_colors == 0:
            return color, n_colors
        
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)
        
        improved = True
        while improved:
            improved = False
            for c in range(n_colors - 1, -1, -1):
                if not classes[c]:
                    continue
                    
                temp_assign = {}
                success = True
                for e in classes[c]:
                    placed = False
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        # Check if e conflicts with any edge in class c2
                        if not any(conflicts[e] & (1 << x) for x in classes[c2]):
                            temp_assign[e] = c2
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                
                if success:
                    for e, c2 in temp_assign.items():
                        classes[c].discard(e)
                        classes[c2].add(e)
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    break
                    
        return color, n_colors

    def local_polish(color, conflicts, n_edges, max_iters=200):
        """
        Fine-grained local search: try to move individual edges to lower colors.
        """
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors
            
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)
            
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for c in range(n_colors - 1, -1, -1):
                if not classes[c]:
                    continue
                for e in list(classes[c]):
                    for c2 in range(c):
                        if not any(conflicts[e] & (1 << x) for x in classes[c2]):
                            classes[c].discard(e)
                            classes[c2].add(e)
                            color[e] = c2
                            improved = True
                            break
                    if improved:
                        if not classes[n_colors - 1]:
                            n_colors -= 1
                        break
                if improved:
                    break
            if not improved:
                break
                
        return color, n_colors

    def simulated_annealing(color, conflicts, n_edges, time_budget=2.0):
        """
        Simulated annealing to escape local optima.
        Moves edges between color classes, accepting worse solutions with probability exp(-delta/T).
        """
        if not color:
            return color, 0
            
        n_colors = max(color) + 1
        if n_colors == 0:
            return color, n_colors
            
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)
        
        best_color = color[:]
        best_n = n_colors
        rng = random.Random(hash(tuple(color)) & 0xFFFFFFFF)
        
        T = 1.0
        cooling = 0.995
        end_time = time.time() + time_budget
        
        while time.time() < end_time and T > 0.01:
            # Pick a random edge and try to move it
            e = rng.randrange(n_edges)
            c_old = color[e]
            c_new = rng.randrange(n_colors)
            
            if c_old == c_new:
                continue
            
            # Check if move is valid (no conflicts)
            if any(conflicts[e] & (1 << x) for x in classes[c_new]):
                continue
            
            # Compute delta: change in number of colors used
            # If we're removing the last edge from c_old, we might reduce n_colors
            old_n = n_colors
            if len(classes[c_old]) == 1 and c_old == n_colors - 1:
                new_n = n_colors - 1
            else:
                new_n = n_colors
            
            delta = new_n - old_n
            
            # Accept if better, or with probability exp(-delta/T) if worse
            if delta <= 0 or rng.random() < (2.718281828 ** (-delta / T)):
                # Apply move
                classes[c_old].discard(e)
                classes[c_new].add(e)
                color[e] = c_new
                
                # Update n_colors if needed
                if not classes[n_colors - 1]:
                    n_colors -= 1
                    if n_colors < len(classes):
                        classes = classes[:n_colors]
                elif c_new >= n_colors:
                    n_colors = c_new + 1
                    while len(classes) < n_colors:
                        classes.append(set())
                
                if n_colors < best_n:
                    best_n = n_colors
                    best_color = color[:]
            else:
                # Revert (no-op since we didn't apply)
                pass
            
            T *= cooling
        
        return best_color, best_n

    def final_polish(color, conflicts, n_edges, max_iters=500):
        """
        High-precision final polish: multiple passes of local improvement.
        """
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors
            
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)
            
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            improved = False
            
            # Pass 1: Try to eliminate highest color
            for c in range(n_colors - 1, -1, -1):
                if not classes[c]:
                    continue
                temp_assign = {}
                success = True
                for e in classes[c]:
                    placed = False
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        if not any(conflicts[e] & (1 << x) for x in classes[c2]):
                            temp_assign[e] = c2
                            placed = True
                            break
                    if not placed:
                        success = False
                        break
                if success:
                    for e, c2 in temp_assign.items():
                        classes[c].discard(e)
                        classes[c2].add(e)
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    break
            
            # Pass 2: Move individual edges to lower colors
            if not improved:
                for c in range(n_colors - 1, -1, -1):
                    if not classes[c]:
                        continue
                    for e in list(classes[c]):
                        for c2 in range(c):
                            if not any(conflicts[e] & (1 << x) for x in classes[c2]):
                                classes[c].discard(e)
                                classes[c2].add(e)
                                color[e] = c2
                                improved = True
                                break
                        if improved:
                            if not classes[n_colors - 1]:
                                n_colors -= 1
                            break
                    if improved:
                        break
            
            if not improved:
                break
                
        return color, n_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        n_edges = len(edges)
        conflicts = build_conflicts_bitmask(edges, n)
        
        best_color = None
        best_count = float('inf')
        
        # 1. Greedy with multiple random restarts
        rng = random.Random(seed + 999)
        num_restarts = 50  # Increased due to faster bitmask checks
        
        for r in range(num_restarts):
            if time.time() > deadline - 30.0:
                break
            order = list(range(n_edges))
            rng.shuffle(order)
            color, n_colors = greedy_color(order, conflicts, n_edges)
            
            # 2. Class Elimination
            color, n_colors = try_eliminate_colors(color, conflicts, n_edges)
            
            # 3. Local Polish
            color, n_colors = local_polish(color, conflicts, n_edges)
            
            # 4. Simulated Annealing for this candidate
            if n_colors < best_count and time.time() < deadline - 15.0:
                color_sa, n_sa = simulated_annealing(color, conflicts, n_edges, time_budget=1.0)
                if n_sa < best_count:
                    best_count = n_sa
                    best_color = color_sa[:]
            
            if n_colors < best_count:
                best_count = n_colors
                best_color = color[:]
                
        # Also try sorted order as a baseline
        if time.time() < deadline - 30.0:
            order = list(range(n_edges))
            color, n_colors = greedy_color(order, conflicts, n_edges)
            color, n_colors = try_eliminate_colors(color, conflicts, n_edges)
            color, n_colors = local_polish(color, conflicts, n_edges)
            if n_colors < best_count:
                best_count = n_colors
                best_color = color[:]

        # Final high-precision polish on the best solution
        if best_color is not None and time.time() < deadline - 5.0:
            best_color, best_count = final_polish(best_color, conflicts, n_edges)
            
        # Final safety check
        if best_color is None:
            order = list(range(n_edges))
            best_color, _ = greedy_color(order, conflicts, n_edges)
            
        results.append(best_color)
        
    return results
