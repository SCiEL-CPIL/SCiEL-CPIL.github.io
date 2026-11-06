"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges, n):
        """Build conflict graph: edges i,j conflict if they share a vertex or are distance-2 apart."""
        m = len(edges)
        # Map vertex -> list of edge indices
        vert_edges = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            vert_edges[u].add(i)
            vert_edges[v].add(i)
        # Conflict sets
        conflicts = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            for j in vert_edges[u]:
                if j != i:
                    conflicts[i].add(j)
            for j in vert_edges[v]:
                if j != i:
                    conflicts[i].add(j)
        return conflicts

    def dsatur_coloring(conflicts, m):
        """DSATUR heuristic for graph coloring."""
        colors = [-1] * m
        sat = [set() for _ in range(m)]
        deg = [len(conflicts[i]) for i in range(m)]
        uncolored = set(range(m))
        
        for _ in range(m):
            if not uncolored:
                break
            # Pick vertex with max saturation, tie-break by degree
            best = None
            best_key = None
            for v in uncolored:
                key = (len(sat[v]), deg[v])
                if best_key is None or key > best_key:
                    best_key = key
                    best = v
            # Assign smallest available color
            avail = 0
            while avail in sat[best]:
                avail += 1
            colors[best] = avail
            uncolored.discard(best)
            for nb in conflicts[best]:
                if nb in uncolored:
                    sat[nb].add(avail)
        return colors

    def try_improve(colors, conflicts, m, max_iters=2000):
        """Local search: try to reduce color count by reassigning colors."""
        colors = list(colors)
        current_max = max(colors) + 1
        
        for _ in range(max_iters):
            improved = False
            # Try to eliminate highest color
            target_color = current_max - 1
            vertices_with_target = [v for v in range(m) if colors[v] == target_color]
            if not vertices_with_target:
                current_max -= 1
                continue
            
            # Try reassigning each vertex with target color to a lower color
            for v in vertices_with_target:
                # Find available color < target_color
                for c in range(target_color):
                    if c not in {colors[nb] for nb in conflicts[v] if colors[nb] != -1}:
                        colors[v] = c
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try to reduce by moving a vertex to free a color
                # Try removing a color entirely
                for c in range(current_max - 1, -1, -1):
                    verts_c = [v for v in range(m) if colors[v] == c]
                    if not verts_c:
                        current_max = c
                        break
                    # Try to move all vertices with color c to other colors
                    moved = True
                    for v in verts_c:
                        placed = False
                        for c2 in range(current_max):
                            if c2 == c:
                                continue
                            if c2 not in {colors[nb] for nb in conflicts[v] if colors[nb] != -1}:
                                colors[v] = c2
                                placed = True
                                break
                        if not placed:
                            moved = False
                            break
                    if moved:
                        current_max = c
                        improved = True
                        break
            
            if not improved:
                break
        
        return colors, current_max

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n = spec[0]
        m = len(edges)
        conflicts = build_conflict(edges, n)
        
        # DSATUR initial coloring
        colors = dsatur_coloring(conflicts, m)
        num_colors = max(colors) + 1
        
        # Local search improvement
        colors, num_colors = try_improve(colors, conflicts, m)
        
        # Check time and try more rounds if time permits
        if time.time() < deadline - 5:
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                colors2, nc2 = try_improve(colors, conflicts, m, max_iters=1000)
                if nc2 < num_colors:
                    colors, num_colors = colors2, nc2
        
        out.append(colors)
    
    return out
# EVOLVE-BLOCK-END
