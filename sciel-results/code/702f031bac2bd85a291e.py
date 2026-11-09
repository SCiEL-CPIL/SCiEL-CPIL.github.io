import time
import random
import math

DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    t0 = time.time()

    def make_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # Try more attempts to get a denser graph
        for _ in range(8 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_data(edges, n):
        """Build adjacency and edge-index structures."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        m = len(edges)
        # edge_index: maps (u,v) -> index
        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i
        return adj, edge_idx, m

    def is_induced_matching(colour_class, edges, adj, edge_idx):
        """Check if a set of edge indices forms an induced matching."""
        if len(colour_class) <= 1:
            return True
        # Check no shared vertices
        used_verts = set()
        for ei in colour_class:
            u, v = edges[ei]
            if u in used_verts or v in used_verts:
                return False
            used_verts.add(u)
            used_verts.add(v)
        # Check no edges joining them
        for ei in colour_class:
            u, v = edges[ei]
            for w in adj[u]:
                if w != v and (w, min(u,v)) in edge_idx or (w, max(u,v)) in edge_idx:
                    # w is adjacent to u but not v, so edge (w,?) might join
                    # Actually check: is there an edge from w to any vertex in the matching?
                    pass
        # More careful: for each edge (u,v) in matching, check no edge from u to any other matching vertex
        # and no edge from v to any other matching vertex
        matching_verts = set()
        for ei in colour_class:
            u, v = edges[ei]
            matching_verts.add((u, v))
        for ei in colour_class:
            u, v = edges[ei]
            for w in adj[u]:
                if w != v and (w, u) in matching_verts or (w, v) in matching_verts:
                    # Check if w is connected to v (the other endpoint of same edge) - that's fine
                    # Check if w is connected to any OTHER matching edge's endpoint
                    for ej in colour_class:
                        if ej == ei:
                            continue
                        a, b = edges[ej]
                        if w == a or w == b:
                            return False
            for w in adj[v]:
                if w != u:
                    for ej in colour_class:
                        if ej == ei:
                            continue
                        a, b = edges[ej]
                        if w == a or w == b:
                            return False
        return True

    def dsatur_colouring(edges, adj, n, edge_idx, max_colours):
        """DSATUR-based initial colouring for induced matching."""
        m = len(edges)
        colour = [-1] * m
        # For each colour, track which edges are assigned
        colour_classes = []  # list of lists of edge indices
        # Precompute for each edge the set of edges it conflicts with
        # Conflict: share a vertex OR are joined by an edge
        conflict = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            for j, (a, b) in enumerate(edges):
                if i == j:
                    continue
                # Share vertex?
                if u in (a, b) or v in (a, b):
                    conflict[i].add(j)
                    continue
                # Joined by an edge?
                if a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                    conflict[i].add(j)
        
        # DSATUR: pick edge with most differently-coloured neighbours first
        # Track saturation (number of different colours in conflict set)
        for step in range(m):
            best_e = -1
            best_sat = -1
            best_deg = -1
            for i in range(m):
                if colour[i] != -1:
                    continue
                # Count distinct colours among conflicts
                col_set = set()
                for j in conflict[i]:
                    if colour[j] != -1:
                        col_set.add(colour[j])
                sat = len(col_set)
                deg = len(conflict[i])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_e = i
            if best_e == -1:
                break
            # Assign smallest available colour
            used = set()
            for j in conflict[best_e]:
                if colour[j] != -1:
                    used.add(colour[j])
            c = 0
            while c in used:
                c += 1
            if c >= max_colours:
                return None  # Failed
            colour[best_e] = c
            while len(colour_classes) <= c:
                colour_classes.append([])
            colour_classes[c].append(best_e)
        
        return colour, colour_classes

    def verify_colouring(colour, edges, adj, edge_idx):
        """Verify the colouring is valid (each colour class is an induced matching)."""
        m = len(edges)
        classes = {}
        for i, c in enumerate(colour):
            if c < 0:
                return False
            classes.setdefault(c, []).append(i)
        for c, cls in classes.items():
            if not is_induced_matching(cls, edges, adj, edge_idx):
                return False
        return True

    def local_search(colour, edges, adj, edge_idx, n, max_colours, time_limit=5.0):
        """Local search: try to reduce number of colours by moving edges."""
        m = len(edges)
        t_start = time.time()
        # Build conflict sets
        conflict = [set() for _ in range(m)]
        for i, (u, v) in enumerate(edges):
            for j, (a, b) in enumerate(edges):
                if i == j:
                    continue
                if u in (a, b) or v in (a, b):
                    conflict[i].add(j)
                    continue
                if a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                    conflict[i].add(j)
        
        num_colours = max(colour) + 1 if m > 0 else 0
        best_colour = list(colour)
        best_nc = num_colours
        
        for iteration in range(2000):
            if time.time() - t_start > time_limit:
                break
            # Try to eliminate the highest colour
            target_c = num_colours - 1
            if target_c < 0:
                break
            # Find edges with this colour
            edges_in_class = [i for i in range(m) if colour[i] == target_c]
            if not edges_in_class:
                num_colours -= 1
                continue
            # Try to move each edge in this class to a lower colour
            moved = False
            for ei in edges_in_class:
                for c in range(target_c):
                    # Check if ei can be moved to colour c
                    ok = True
                    for j in conflict[ei]:
                        if colour[j] == c:
                            ok = False
                            break
                    if ok:
                        colour[ei] = c
                        moved = True
                        break
                if not moved:
                    # Try swapping
                    for ej in edges_in_class:
                        if ej == ei:
                            continue
                        # Try swapping ei and ej's colours
                        old_ei = colour[ei]
                        old_ej = colour[ej]
                        # Can we swap?
                        ok_ei = True
                        for j in conflict[ei]:
                            if colour[j] == old_ej and j != ej:
                                ok_ei = False
                                break
                        ok_ej = True
                        for j in conflict[ej]:
                            if colour[j] == old_ei and j != ei:
                                ok_ej = False
                                break
                        if ok_ei and ok_ej:
                            colour[ei] = old_ej
                            colour[ej] = old_ei
                            moved = True
                            break
            if moved:
                # Recompute num_colours
                num_colours = max(colour) + 1 if m > 0 else 0
                if num_colours < best_nc:
                    best_nc = num_colours
                    best_colour = list(colour)
            else:
                # Perturbation: randomly reassign a few edges
                for _ in range(3):
                    ei = random.randint(0, m - 1)
                    c = random.randint(0, max_colours - 1)
                    colour[ei] = c
                num_colours = max(colour) + 1 if m > 0 else 0
        
        return best_colour, best_nc

    def solve_instance(n, d, seed):
        edges = make_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        adj, edge_idx, m = build_data(edges, n)
        
        # Upper bound from Erdős-Nešetřil: 5d^2/4
        upper_bound = int(math.ceil(5 * d * d / 4.0))
        # Add some margin
        max_colours = upper_bound + 5
        
        best_colour = None
        best_nc = m  # worst case: all different
        
        # Multi-restart DSATUR + local search
        t_start = time.time()
        for restart in range(30):
            if time.time() > DEADLINE:
                break
            # DSATUR initialisation
            result = dsatur_colouring(edges, adj, n, edge_idx, max_colours)
            if result is None:
                continue
            colour, classes = result
            nc = max(colour) + 1
            if nc < best_nc:
                best_nc = nc
                best_colour = list(colour)
            # Local search
            ls_time = min(3.0, max(0.1, DEADLINE - time.time() - 1.0))
            colour2, nc2 = local_search(colour, edges, adj, edge_idx, n, max_colours, ls_time)
            if nc2 < best_nc:
                best_nc = nc2
                best_colour = list(colour2)
            if time.time() > DEADLINE - 5.0:
                break
        
        if best_colour is None:
            # Fallback: greedy
            colour = [-1] * m
            for i, (u, v) in enumerate(edges):
                used = set()
                for j in range(m):
                    if colour[j] != -1:
                        a, b = edges[j]
                        if u in (a, b) or v in (a, b):
                            used.add(colour[j])
                        elif a in adj[u] or a in adj[v] or b in adj[u] or b in adj[v]:
                            used.add(colour[j])
                c = 0
                while c in used:
                    c += 1
                colour[i] = c
            best_colour = colour
            best_nc = max(colour) + 1
        
        return best_colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE:
            # Return best effort
            out.append([])
            continue
        result = solve_instance(*spec)
        out.append(result)
    
    return out

