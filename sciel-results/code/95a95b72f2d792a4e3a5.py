"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock, with safety margin we'll check against
    
    N = 20
    C = 20  # number of colours
    MAX_EDGES = N * (N - 1) // 2  # 190
    
    # We'll use a greedy + local search approach.
    # State: for each pair (u,v) with u<v, either no edge or an edge with a colour.
    # edges[u][v] = colour or -1 (no edge)
    
    def is_valid_coloring(edges):
        """Check if every 4-cycle has 4 distinct colours."""
        # For each 4-cycle (a,b,c,d) in order, check that edges (a,b),(b,c),(c,d),(d,a)
        # all exist and have distinct colours.
        # A 4-cycle is a set of 4 vertices {a,b,c,d} and a cyclic ordering.
        # We need to check all cyclic orderings of 4 distinct vertices.
        # For vertices a<b<c<d, the possible 4-cycles are:
        #   a-b-c-d-a, a-b-d-c-a, a-c-b-d-a
        # These correspond to the 3 perfect matchings of K4 being the "diagonals".
        # Actually, a 4-cycle on 4 vertices is determined by which 2 pairs are opposite.
        # For 4 vertices {a,b,c,d}, there are 3 ways to form a 4-cycle:
        #   (a,b,c,d), (a,b,d,c), (a,c,b,d)
        # where (a,b,c,d) means edges (a,b),(b,c),(c,d),(d,a).
        
        for a in range(N):
            for b in range(a+1, N):
                for c in range(b+1, N):
                    for d in range(c+1, N):
                        # Three possible 4-cycles on {a,b,c,d}
                        # Cycle 1: a-b-c-d-a
                        if (edges[a][b] >= 0 and edges[b][c] >= 0 and 
                            edges[c][d] >= 0 and edges[d][a] >= 0):
                            cols = {edges[a][b], edges[b][c], edges[c][d], edges[d][a]}
                            if len(cols) < 4:
                                return False
                        # Cycle 2: a-b-d-c-a
                        if (edges[a][b] >= 0 and edges[b][d] >= 0 and 
                            edges[d][c] >= 0 and edges[c][a] >= 0):
                            cols = {edges[a][b], edges[b][d], edges[d][c], edges[c][a]}
                            if len(cols) < 4:
                                return False
                        # Cycle 3: a-c-b-d-a
                        if (edges[a][c] >= 0 and edges[c][b] >= 0 and 
                            edges[b][d] >= 0 and edges[d][a] >= 0):
                            cols = {edges[a][c], edges[c][b], edges[b][d], edges[d][a]}
                            if len(cols) < 4:
                                return False
        return True
    
    def count_edges(edges):
        cnt = 0
        for u in range(N):
            for v in range(u+1, N):
                if edges[u][v] >= 0:
                    cnt += 1
        return cnt
    
    def get_conflicts_for_edge(edges, u, v, color):
        """
        If we add edge (u,v) with given color, check if it creates any invalid 4-cycle.
        A new 4-cycle involving (u,v) must use (u,v) as one of its 4 edges.
        The other 3 edges must form a path from u to v of length 3: u-x-y-v.
        So for each pair (x,y) with x!=v, y!=u, x!=y, x,y distinct from u,v:
          if edges(u,x), edges(x,y), edges(y,v) all exist, then (u,v,x,y) or (u,v,y,x) 
          forms a 4-cycle. We need to check if adding (u,v) with 'color' makes any
          such 4-cycle non-rainbow.
        """
        # For each x != u,v:
        for x in range(N):
            if x == u or x == v:
                continue
            if edges[u][x] < 0:
                continue
            # For each y != u,v,x:
            for y in range(N):
                if y == u or y == v or y == x:
                    continue
                if edges[x][y] < 0 or edges[y][v] < 0:
                    continue
                # We have path u-x-y-v. Adding (u,v) creates 4-cycle u-x-y-v-u.
                # The colours are: edges[u][x], edges[x][y], edges[y][v], color
                cols = {edges[u][x], edges[x][y], edges[y][v], color}
                if len(cols) < 4:
                    return True
        return False
    
    def can_add_edge(edges, u, v, color):
        """Check if adding edge (u,v) with 'color' keeps the coloring valid."""
        if u == v:
            return False
        if edges[u][v] >= 0:
            return False
        return not get_conflicts_for_edge(edges, u, v, color)
    
    def try_add_edge(edges, u, v):
        """Try to add edge (u,v) with some valid colour. Returns True if added."""
        for color in range(C):
            if can_add_edge(edges, u, v, color):
                edges[u][v] = color
                edges[v][u] = color
                return True
        return False
    
    def remove_edge(edges, u, v):
        edges[u][v] = -1
        edges[v][u] = -1
    
    def recolor_edge(edges, u, v, new_color):
        """Recolor edge (u,v) to new_color, checking validity."""
        old_color = edges[u][v]
        if old_color == new_color:
            return True
        # Temporarily remove
        edges[u][v] = -1
        edges[v][u] = -1
        if can_add_edge(edges, u, v, new_color):
            edges[u][v] = new_color
            edges[v][u] = new_color
            return True
        else:
            edges[u][v] = old_color
            edges[v][u] = old_color
            return False
    
    # Initialize with no edges
    edges = [[-1]*N for _ in range(N)]
    
    # Generate all possible edges sorted by some heuristic
    # Strategy: use spectral/Hadamard-inspired initialization
    # For a 20x20 Hadamard matrix, we can use the Sylvester construction or just a random orthogonal-like matrix.
    # Let's use a simple approach: sort edges by the number of common neighbors (ascending) to minimize C4 pressure.
    
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    # Compute a heuristic: for each edge (u,v), count how many 4-cycles it would be part of
    # if all other edges existed. This is proportional to the number of common neighbors of u and v... 
    # Actually, for a new edge (u,v), the number of 4-cycles it participates in is the number of paths u-x-y-v
    # which is sum over x,y of [edge(u,x)]*[edge(x,y)]*[edge(y,v)].
    # Initially with no edges, this is 0 for all. Let's use a different heuristic.
    
    # Let's use a structured ordering: bipartite-like. Split vertices into two sets of 10.
    # Add cross-part edges first (they form C4s only with other cross-part edges in a specific way),
    # then within-part edges.
    
    part1 = list(range(10))
    part2 = list(range(10, 20))
    
    cross_edges = [(u, v) for u in part1 for v in part2]
    within1_edges = [(u, v) for u in range(10) for v in range(u+1, 10)]
    within2_edges = [(u, v) for u in range(10, 20) for v in range(u+1, 20)]
    
    # Shuffle within groups for variety
    random.seed(42)
    random.shuffle(cross_edges)
    random.shuffle(within1_edges)
    random.shuffle(within2_edges)
    
    # Greedy phase: try to add edges in order
    for (u, v) in cross_edges:
        if time.time() > deadline - 30:
            break
        try_add_edge(edges, u, v)
    
    for (u, v) in within1_edges:
        if time.time() > deadline - 30:
            break
        try_add_edge(edges, u, v)
    
    for (u, v) in within2_edges:
        if time.time() > deadline - 30:
            break
        try_add_edge(edges, u, v)
    
    # Try remaining edges
    for (u, v) in all_edges:
        if time.time() > deadline - 30:
            break
        if edges[u][v] < 0:
            try_add_edge(edges, u, v)
    
    best_edges = [row[:] for row in edges]
    best_count = count_edges(best_edges)
    
    # Local search phase
    # Try to add missing edges and recolor existing edges to enable more additions
    missing_edges = [(u, v) for u in range(N) for v in range(u+1, N) if edges[u][v] < 0]
    
    while time.time() < deadline - 10:
        # Try to add each missing edge
        progress = False
        for (u, v) in missing_edges:
            if time.time() > deadline - 10:
                break
            if edges[u][v] >= 0:
                continue
            for color in range(C):
                if can_add_edge(edges, u, v, color):
                    edges[u][v] = color
                    edges[v][u] = color
                    progress = True
                    break
        if not progress:
            break
        
        # Update missing edges
        missing_edges = [(u, v) for u in range(N) for v in range(u+1, N) if edges[u][v] < 0]
        
        # Try recoloring to enable more edges
        for (u, v) in missing_edges:
            if time.time() > deadline - 10:
                break
            # For each existing edge that conflicts with adding (u,v), try to recolor it
            # First, find which colors would work for (u,v)
            valid_colors_for_uv = [c for c in range(C) if can_add_edge(edges, u, v, c)]
            if not valid_colors_for_uv:
                # Try to recolor some edges to free up a color
                # Find all 4-cycles that (u,v) would participate in
                conflict_edges = set()
                for x in range(N):
                    if x == u or x == v:
                        continue
                    if edges[u][x] < 0:
                        continue
                    for y in range(N):
                        if y == u or y == v or y == x:
                            continue
                        if edges[x][y] < 0 or edges[y][v] < 0:
                            continue
                        # This 4-cycle u-x-y-v blocks (u,v)
                        # The colors involved: edges[u][x], edges[x][y], edges[y][v]
                        # We need one of these to change so that for some color c, all 4 are distinct
                        conflict_edges.add((u, x))
                        conflict_edges.add((x, y))
                        conflict_edges.add((y, v))
                
                # Try recoloring each conflict edge
                for (cu, cv) in conflict_edges:
                    if time.time() > deadline - 10:
                        break
                    if cu > cv:
                        cu, cv = cv, cu
                    for new_color in range(C):
                        if recolor_edge(edges, cu, cv, new_color):
                            # Check if (u,v) can now be added
                            for color in range(C):
                                if can_add_edge(edges, u, v, color):
                                    edges[u][v] = color
                                    edges[v][u] = color
                                    progress = True
                                    break
                            if progress:
                                break
                    if progress:
                        break
                if progress:
                    break
        
        if not progress:
            # Try a random restart of local search with random recoloring
            # Pick a random existing edge and try to recolor it
            existing = [(u, v) for u in range(N) for v in range(u+1, N) if edges[u][v] >= 0]
            if not existing:
                break
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                u, v = random.choice(existing)
                old_color = edges[u][v]
                for new_color in random.sample(range(C), min(C, 5)):
                    if new_color != old_color and recolor_edge(edges, u, v, new_color):
                        # Try to add some missing edges
                        for (mu, mv) in random.sample(missing_edges, min(len(missing_edges), 20)):
                            if edges[mu][mv] >= 0:
                                continue
                            for color in range(C):
                                if can_add_edge(edges, mu, mv, color):
                                    edges[mu][mv] = color
                                    edges[mv][mu] = color
                                    break
                        break
            # Update
            missing_edges = [(u, v) for u in range(N) for v in range(u+1, N) if edges[u][v] < 0]
            current_count = count_edges(edges)
            if current_count > best_count:
                best_count = current_count
                best_edges = [row[:] for row in edges]
            if not missing_edges:
                break
            # If no progress after random recoloring, break
            # Check if we made progress
            if count_edges(edges) <= best_count:
                # Revert to best
                edges = [row[:] for row in best_edges]
                missing_edges = [(u, v) for u in range(N) for v in range(u+1, N) if edges[u][v] < 0]
                break
    
    # Final check
    if not is_valid_coloring(best_edges):
        # Fallback: use a star
        return [[0, v, 0] for v in range(1, 20)]
    
    # Convert to output format
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if best_edges[u][v] >= 0:
                result.append([u, v, best_edges[u][v]])
    
    return result
# EVOLVE-BLOCK-END
