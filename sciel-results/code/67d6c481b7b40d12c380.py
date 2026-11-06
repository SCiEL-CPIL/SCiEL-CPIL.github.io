"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random

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

    def build_conflict_graph(edges, edge_set):
        m = len(edges)
        conflict = [set() for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, d = edges[j]
                if a == c or a == d or b == c or b == d:
                    conflict[i].add(j)
                    conflict[j].add(i)
                else:
                    # Check if endpoints are joined by an edge in the graph
                    # edge_set contains tuples (min(u, v), max(u, v))
                    if (min(a, c), max(a, c)) in edge_set or \
                       (min(a, d), max(a, d)) in edge_set or \
                       (min(b, c), max(b, c)) in edge_set or \
                       (min(b, d), max(b, d)) in edge_set:
                        conflict[i].add(j)
                        conflict[j].add(i)
        return conflict

    def dsatur_color(conflict, m, rng=None):
        color = [-1] * m
        color_edges = {i: [] for i in range(m)} # Pre-allocate? No, dynamic is better for memory if m is large, but m is small.
        # Actually, we just need to know which colors are used in the neighborhood.
        # Let's use a set of colors for each vertex to track saturation.
        sat = [set() for _ in range(m)]
        
        # Initial order can be arbitrary or by degree, but DSATUR is dynamic.
        # We pick the uncolored vertex with highest saturation, then highest degree.
        uncolored = set(range(m))
        max_color_used = -1
        
        while uncolored:
            # Select next vertex
            best_v = -1
            best_sat = -1
            best_deg = -1
            for v in uncolored:
                s = len(sat[v])
                d = len(conflict[v])
                if s > best_sat or (s == best_sat and d > best_deg):
                    best_v = v
                    best_sat = s
                    best_deg = d
            
            # Assign color
            # Find smallest color not in sat[best_v]
            c = 0
            while c in sat[best_v]:
                c += 1
            
            color[best_v] = c
            if c > max_color_used:
                max_color_used = c
            
            # Update saturation of neighbors
            for u in conflict[best_v]:
                if u in uncolored:
                    sat[u].add(c)
            
            uncolored.remove(best_v)
            
        return color, max_color_used + 1

    out = []
    start_time = time.time()
    deadline = start_time + 140

    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline:
            while len(out) < 3:
                out.append([])
            break

        edges = instance(*spec)
        m = len(edges)
        edge_set = set(edges)
        conflict = build_conflict_graph(edges, edge_set)

        best_color = None
        best_count = m + 1

        # Run DSATUR multiple times with different random tie-breakers?
        # Standard DSATUR is deterministic if ties are broken by ID.
        # To add randomness, we can shuffle the initial vertex list or break ties randomly.
        # Let's try a few variations.
        
        rng = Random(42)
        
        # Variation 1: Standard DSATUR
        color1, count1 = dsatur_color(conflict, m)
        if count1 < best_count:
            best_count = count1
            best_color = color1

        # Variation 2: Randomized DSATUR (shuffle ties)
        # We can modify dsatur to accept an rng and use it for tie-breaking.
        # For simplicity, let's just run standard DSATUR first, then try a few random restarts if we have time.
        # But DSATUR is fast. Let's try to implement a randomized version properly.
        
        def dsatur_randomized(conflict, m, rng):
            color = [-1] * m
            sat = [set() for _ in range(m)]
            uncolored = list(range(m))
            rng.shuffle(uncolored) # Randomize initial order to break ties in first step? 
            # Actually, DSATUR picks based on current saturation. 
            # To randomize, we can pick randomly among vertices with max saturation.
            
            max_color_used = -1
            
            while uncolored:
                # Find max saturation
                max_sat = -1
                candidates = []
                for v in uncolored:
                    s = len(sat[v])
                    if s > max_sat:
                        max_sat = s
                        candidates = [v]
                    elif s == max_sat:
                        candidates.append(v)
                
                # Pick one of the candidates
                # Tie break by degree? Or random?
                # Let's pick random from candidates to explore different colorings
                v = rng.choice(candidates)
                
                # Assign color
                c = 0
                while c in sat[v]:
                    c += 1
                
                color[v] = c
                if c > max_color_used:
                    max_color_used = c
                
                for u in conflict[v]:
                    if u in uncolored:
                        sat[u].add(c)
                
                uncolored.remove(v)
                
            return color, max_color_used + 1

        # Run randomized DSATUR a few times
        for _ in range(10):
            if time.time() > deadline:
                break
            color_r, count_r = dsatur_randomized(conflict, m, rng)
            if count_r < best_count:
                best_count = count_r
                best_color = color_r

        if best_color is None:
            best_color = color1

        out.append([best_color[e] for e in range(m)])

    while len(out) < 3:
        out.append([])

    return out
# EVOLVE-BLOCK-END
