"""Strong edge colourings via DSATUR + color-elimination local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

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

    def build_conflict_bitsets(edges, adj):
        """Build conflict graph as list of frozensets (or use integer bitmasks for small graphs)."""
        n_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        # Use list of sets for conflicts
        conflicts = [set() for _ in range(n_edges)]

        for i, (u, v) in enumerate(edges):
            # Edges sharing a vertex
            for nb in adj[u]:
                e2 = (min(u, nb), max(u, nb))
                if e2 in edge_to_idx:
                    j = edge_to_idx[e2]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            for nb in adj[v]:
                e2 = (min(v, nb), max(v, nb))
                if e2 in edge_to_idx:
                    j = edge_to_idx[e2]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            # Edges joined by an edge: (u,w) where w~u and w!=v, and (w,x) is an edge
            # Actually "joined by an edge" means there exists a vertex w such that (u,w) in E or (v,w) in E
            # and (w, something) is another edge... 
            # More precisely: edge e1=(a,b) conflicts with e2=(c,d) if:
            #   {a,b} ∩ {c,d} ≠ ∅ (share vertex)
            #   OR (a,c)∈E, (a,d)∈E, (b,c)∈E, (b,d)∈E (joined by an edge)
            # The "joined by an edge" part: for each endpoint of e1, look at its neighbors,
            # and check if any edge has both endpoints among those neighbors... 
            # Actually no: "joined by an edge" means there's a single edge connecting an endpoint of e1 to an endpoint of e2.
            # So for e1=(u,v): for each w in adj[u] (w≠v), check if there's an edge (w,x) where x is... 
            # Wait, re-reading: "no two edges share a vertex or are joined by an edge"
            # "joined by an edge" = there exists an edge in the graph that has one endpoint in e1 and one in e2.
            # So for e1=(u,v), e2=(c,d): conflict if (u,c)∈E or (u,d)∈E or (v,c)∈E or (v,d)∈E
            for w in adj[u]:
                if w != v:
                    # Check all edges incident to w that don't involve u or v
                    for x in adj[w]:
                        if x != u and x != v:
                            e2 = (min(w, x), max(w, x))
                            if e2 in edge_to_idx:
                                j = edge_to_idx[e2]
                                if j != i:
                                    conflicts[i].add(j)
                                    conflicts[j].add(i)
            for w in adj[v]:
                if w != u:
                    for x in adj[w]:
                        if x != u and x != v:
                            e2 = (min(w, x), max(w, x))
                            if e2 in edge_to_idx:
                                j = edge_to_idx[e2]
                                if j != i:
                                    conflicts[i].add(j)
                                    conflicts[j].add(i)

        return conflicts

    def dsatur_color(conflicts, n_edges, time_budget):
        """DSATUR greedy coloring with bitset tracking."""
        colour = [-1] * n_edges
        # For each color, track which edges have that color
        color_edges = defaultdict(set)
        # For fast "is color c blocked for edge i" check, track per-edge blocked colors
        # blocked[i] = set of colors used by neighbors of i
        
        # DSATUR: pick uncolored edge with max saturation (distinct colors among colored neighbors),
        # tiebreak by degree in conflict graph
        uncolored = set(range(n_edges))
        # Precompute degree for tiebreaking
        deg = [len(conflicts[i]) for i in range(n_edges)]
        
        for _ in range(n_edges):
            if time.time() > time_budget:
                break
            best_i = -1
            best_sat = -1
            best_deg = -1
            for i in uncolored:
                # Compute saturation: number of distinct colors among colored neighbors
                sat_colors = set()
                for j in conflicts[i]:
                    if colour[j] != -1:
                        sat_colors.add(colour[j])
                sat = len(sat_colors)
                if sat > best_sat or (sat == best_sat and deg[i] > best_deg):
                    best_sat = sat
                    best_deg = deg[i]
                    best_i = i
            if best_i == -1:
                break
            uncolored.remove(best_i)
            # Assign smallest available color
            blocked = set()
            for j in conflicts[best_i]:
                if colour[j] != -1:
                    blocked.add(colour[j])
            c = 0
            while c in blocked:
                c += 1
            colour[best_i] = c
            color_edges[c].add(best_i)
        
        # If we ran out of time, assign remaining edges greedily
        if uncolored:
            for i in sorted(uncolored):
                blocked = set()
                for j in conflicts[i]:
                    if colour[j] != -1:
                        blocked.add(colour[j])
                c = 0
                while c in blocked:
                    c += 1
                colour[i] = c
                color_edges[c].add(i)
        
        max_color = max(colour) if n_edges > 0 else 0
        return colour, max_color + 1

    def try_eliminate_color(colour, conflicts, n_edges, target_color, time_budget):
        """Try to eliminate target_color by reassigning its edges to lower colors.
        Returns (success, new_colour) or (False, None)."""
        edges_in_color = [i for i in range(n_edges) if colour[i] == target_color]
        if not edges_in_color:
            return True, colour[:]
        
        # Try to reassign each edge in target_color to a lower color
        # Use a simple approach: for each edge, try colors 0..target_color-1
        # If we can reassign all, we've eliminated the color
        new_colour = colour[:]
        
        for idx in edges_in_color:
            placed = False
            for c in range(target_color):
                # Check if color c is available for idx (no neighbor has color c)
                valid = True
                for j in conflicts[idx]:
                    if new_colour[j] == c:
                        valid = False
                        break
                if valid:
                    new_colour[idx] = c
                    placed = True
                    break
            if not placed:
                return False, None
        return True, new_colour

    def local_search_eliminate(colour, conflicts, n_edges, time_budget):
        """Iteratively try to eliminate the highest color, then second highest, etc."""
        max_c = max(colour) if n_edges > 0 else 0
        # Work from highest color down
        for target in range(max_c, -1, -1):
            if time.time() > time_budget:
                break
            if not any(colour[i] == target for i in range(n_edges)):
                continue
            success, new_colour = try_eliminate_color(colour, conflicts, n_edges, target, time_budget)
            if success:
                colour = new_colour
        return colour

    def color_instance(n, d, seed, time_budget):
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflicts = build_conflict_bitsets(edges, adj)
        n_edges = len(edges)
        
        if n_edges == 0:
            return []
        
        best_colour = None
        best_count = n_edges
        
        # DSATUR coloring
        colour, count = dsatur_color(conflicts, n_edges, time_budget)
        if count < best_count:
            best_count = count
            best_colour = colour[:]
        
        # Local search: eliminate colors
        if time.time() < time_budget - 2:
            improved_colour = local_search_eliminate(best_colour, conflicts, n_edges, time_budget)
            if improved_colour is not None:
                max_c = max(improved_colour) if n_edges > 0 else 0
                if max_c + 1 < best_count:
                    best_count = max_c + 1
                    best_colour = improved_colour
        
        # Try a few random perturbations + re-color if time permits
        if time.time() < time_budget - 5 and n_edges > 0:
            rng = random.Random(seed + 999)
            for attempt in range(50):
                if time.time() > time_budget - 2:
                    break
                # Perturb: randomly reassign a few edges
                perturbed = best_colour[:]
                for _ in range(min(5, n_edges)):
                    idx = rng.randrange(n_edges)
                    # Find valid colors for idx
                    blocked = set()
                    for j in conflicts[idx]:
                        if perturbed[j] != -1:
                            blocked.add(perturbed[j])
                    avail = [c for c in range(best_count) if c not in blocked]
                    if avail:
                        perturbed[idx] = rng.choice(avail)
                
                # Try to eliminate colors from perturbed
                perturbed = local_search_eliminate(perturbed, conflicts, n_edges, time_budget)
                if perturbed is not None:
                    max_c = max(perturbed) if n_edges > 0 else 0
                    if max_c + 1 < best_count:
                        best_count = max_c + 1
                        best_colour = perturbed[:]
        
        return [best_colour[i] for i in range(n_edges)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining < 10:
            # Fallback: simple greedy
            edges = instance(*spec)
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = set()
                for f, c in colour.items():
                    if set(f) & near:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                colour[e] = c
            out.append([colour[e] for e in edges])
        else:
            # Allocate time proportional to graph size
            budget = min(remaining * 0.6, 40.0)
            out.append(color_instance(*spec, budget))
    
    return out
# EVOLVE-BLOCK-END
