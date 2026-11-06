"""Strong edge colourings via maximum induced matching extraction."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    DEADLINE = 125  # safety margin before 150s kill

    def instance(n, d, seed):
        from random import Random
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

    def solve_instance(edges, n, d, deadline):
        if not edges:
            return []
        m = len(edges)
        # Build conflict graph: two edges conflict if they share a vertex or are joined by an edge
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # edge_to_idx
        eidx = {e: i for i, e in enumerate(edges)}
        
        # Conflict: edges i and j conflict if they share a vertex or are joined by an edge
        # i.e., there exist u in edge_i, v in edge_j with (u,v) in edges
        # Precompute: for each edge i, the set of edges that conflict with it
        # Two edges (a,b) and (c,d) conflict if:
        #   a==c or a==d or b==c or b==d (share vertex)
        #   or (a,c) in edges or (a,d) in edges or (b,c) in edges or (b,d) in edges (joined)
        
        conflict = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                # Check shared vertex
                if a == c or a == dd or b == c or b == dd:
                    conflict[i].add(j)
                    conflict[j].add(i)
                    continue
                # Check if joined by an edge
                if (a, c) in eidx or (a, dd) in eidx or (b, c) in eidx or (b, dd) in eidx:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        # Now we need to find a minimum number of independent sets (induced matchings)
        # covering all edges. An independent set in the conflict graph = induced matching.
        # Strategy: iteratively extract a maximum independent set, assign it a color,
        # remove it, repeat.
        
        # Maximum independent set via branch and bound (Tomita-style)
        # We work on the conflict graph; an independent set in conflict graph is a set
        # of edges no two of which conflict = induced matching.
        
        # Use a greedy + local search approach for MIS to keep it fast
        # Actually, let's use a proper branch-and-bound for maximum independent set.
        
        # For the conflict graph, we want maximum independent set.
        # Equivalent: maximum clique in the complement of the conflict graph.
        # But building the complement is expensive. Let's do MIS directly.
        
        # Greedy heuristic for a good initial independent set
        def greedy_mis(candidates, conflict_sets):
            """Greedy: pick vertices with fewest conflicts among remaining."""
            if not candidates:
                return set()
            result = set()
            remaining = set(candidates)
            while remaining:
                # Pick vertex with minimum degree in induced subgraph on remaining
                best = None
                best_deg = float('inf')
                for v in remaining:
                    deg = len(conflict_sets[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best = v
                        if deg == 0:
                            break
                result.add(best)
                remaining.discard(best)
                # Remove all neighbors of best from remaining
                remaining -= conflict_sets[best]
            return result
        
        # Branch and bound for maximum independent set
        def max_independent_set(vertices, conflict_sets):
            """Find maximum independent set in the subgraph induced by vertices."""
            if not vertices:
                return set()
            if len(vertices) == 1:
                return set(vertices)
            
            # Quick upper bound: |vertices|
            # Use a simple branch and bound with ordering
            
            vertices = list(vertices)
            # Order by degree (ascending) for better pruning
            vertices.sort(key=lambda v: len(conflict_sets[v] & set(vertices)))
            
            best_set = set()
            best_size = 0
            
            def bb(remaining, current):
                nonlocal best_set, best_size
                if not remaining:
                    if len(current) > best_size:
                        best_size = len(current)
                        best_set = set(current)
                    return
                # Upper bound
                if len(current) + len(remaining) <= best_size:
                    return
                
                v = remaining[0]
                rest = remaining[1:]
                
                # Branch 1: include v
                current.add(v)
                bb([x for x in rest if x not in conflict_sets[v]], current)
                current.discard(v)
                
                # Branch 2: exclude v
                bb(rest, current)
            
            bb(vertices, set())
            return best_set
        
        # Iteratively extract maximum independent sets
        colour = [None] * m
        remaining = list(range(m))
        color_count = 0
        
        while remaining:
            if time.time() > deadline:
                # Fallback: assign each remaining edge a unique color
                for v in remaining:
                    colour[v] = color_count
                    color_count += 1
                break
            
            # Find a good independent set (induced matching) among remaining
            # Use greedy first for a lower bound, then try to improve
            candidates = set(remaining)
            
            # Greedy MIS
            mis = greedy_mis(candidates, conflict)
            
            # Try to improve with a simple local search
            # (For small instances, branch and bound on the candidate set)
            if len(candidates) <= 30:
                # Try exact MIS
                exact_mis = max_independent_set(list(candidates), conflict)
                if len(exact_mis) > len(mis):
                    mis = exact_mis
            
            # Assign this color
            for v in mis:
                colour[v] = color_count
            remaining = [v for v in remaining if v not in mis]
            color_count += 1
        
        return colour
    
    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        colour = solve_instance(edges, n, d, DEADLINE)
        out.append(colour)
    
    return out
# EVOLVE-BLOCK-END
