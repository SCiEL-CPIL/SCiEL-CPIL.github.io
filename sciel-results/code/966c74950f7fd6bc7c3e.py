"""Strong edge colouring: randomized greedy with vertex-forbidden sets + local compression via swaps."""
import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # safety margin under the 150 s kill

    def gen_instance(n, d, seed):
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

    def greedy_first_fit(edges, adj, rng):
        """
        Greedy edge colouring with vertex-forbidden sets.
        For each edge in random order, assign the smallest available colour.
        An edge (u,v) conflicts with any edge sharing u or v, or any edge
        whose endpoints are adjacent to u or v (i.e., distance-2 edges).
        """
        m = len(edges)
        # For each vertex, track which colours are "used" at that vertex
        # and at distance-1 neighbours (for induced matching constraint)
        # colour_at_vertex[v] = set of colours used on edges incident to v
        # colour_at_dist1[v] = set of colours used on edges that are at distance 1 from v
        #   i.e., edges (a,b) where a is adjacent to v (and a != v) or b is adjacent to v
        
        # Actually, let's think about this more carefully.
        # Two edges e1=(u1,v1) and e2=(u2,v2) conflict if:
        #   - They share a vertex: {u1,v1} ∩ {u2,v2} ≠ ∅
        #   - They are joined by an edge: there exists an edge between {u1,v1} and {u2,v2}
        #     i.e., u1~u2 or u1~v2 or v1~u2 or v1~v2 (and they don't share a vertex)
        
        # For greedy: when placing edge e=(u,v), it conflicts with any previously placed edge
        # that shares u or v, or is adjacent to u or v.
        
        # vertex_forbidden[c] won't work directly. Let's track:
        # For each colour c, the set of vertices that are "covered" by colour c
        # A vertex is covered if it's an endpoint of an edge of colour c, OR it's adjacent
        # to an endpoint of an edge of colour c.
        # Then a new edge (u,v) can use colour c iff neither u nor v is covered by colour c.
        
        # But "covered" depends on the graph structure, not just the colour assignment.
        # Let's precompute: for each vertex v, the set of vertices that conflict with v
        # (i.e., v itself and all neighbours of v).
        
        conflict_verts = {}
        for v in range(len(adj)):
            conflict_verts[v] = {v} | adj.get(v, set())
        
        # Now for each colour c, maintain the set of "blocked" vertices
        # blocked[c] = union of conflict_verts[x] for all x that is an endpoint of an edge with colour c
        # Edge (u,v) can use colour c iff u not in blocked[c] and v not in blocked[c]
        
        # To make this efficient, we track for each vertex v, which colours block it.
        # vertex_colour_blocks[v] = set of colours c such that v is in blocked[c]
        
        # When we assign colour c to edge (u,v):
        #   For each x in conflict_verts[u] ∪ conflict_verts[v], add c to vertex_colour_blocks[x]
        #   And blocked[c] gets conflict_verts[u] ∪ conflict_verts[v]
        
        # When checking if edge (u,v) can use colour c:
        #   Check if c in vertex_colour_blocks[u] or c in vertex_colour_blocks[v]
        
        # This is O(deg(u)+deg(v)) per colour check, which is good.
        
        order = list(range(m))
        rng.shuffle(order)
        
        colour = [None] * m
        # vertex_colour_blocks[v] = set of colours that block vertex v
        vertex_colour_blocks = [set() for _ in range(len(adj))]
        # Also track which edges have each colour for potential local search
        colour_edges = {}  # colour -> list of edge indices
        
        for idx in order:
            u, v = edges[idx]
            # Find smallest available colour
            # Try colours 0, 1, 2, ...
            c = 0
            while c in vertex_colour_blocks[u] or c in vertex_colour_blocks[v]:
                c += 1
            # Assign colour c
            colour[idx] = c
            if c not in colour_edges:
                colour_edges[c] = []
            colour_edges[c].append(idx)
            # Update vertex_colour_blocks
            cu = conflict_verts[u]
            cv = conflict_verts[v]
            for x in cu:
                vertex_colour_blocks[x].add(c)
            for x in cv:
                vertex_colour_blocks[x].add(c)
        
        return colour, colour_edges

    def local_compress(edges, adj, colour, colour_edges, rng, max_iters=500):
        """
        Try to reduce the number of colours by moving edges between colour classes.
        We try to merge colour classes or move individual edges to reduce total colours.
        """
        m = len(edges)
        conflict_verts = {}
        for v in range(len(adj)):
            conflict_verts[v] = {v} | adj.get(v, set())
        
        ncolours = max(colour) + 1 if colour else 0
        if ncolours == 0:
            return colour, colour_edges
        
        # Rebuild vertex_colour_blocks from current assignment
        vertex_colour_blocks = [set() for _ in range(len(adj))]
        for c, elist in colour_edges.items():
            for idx in elist:
                u, v = edges[idx]
                for x in conflict_verts[u]:
                    vertex_colour_blocks[x].add(c)
                for x in conflict_verts[v]:
                    vertex_colour_blocks[x].add(c)
        
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Try to eliminate the highest colour class
            if ncolours == 0:
                break
            target_c = ncolours - 1
            if target_c not in colour_edges or len(colour_edges[target_c]) == 0:
                ncolours -= 1
                continue
            
            # Try to move each edge in colour_edges[target_c] to a lower colour
            for idx in list(colour_edges[target_c]):
                u, v = edges[idx]
                moved = False
                for c in range(target_c):
                    if c not in vertex_colour_blocks[u] and c not in vertex_colour_blocks[v]:
                        # Can move edge idx to colour c
                        # Remove from colour target_c
                        colour_edges[target_c].remove(idx)
                        if not colour_edges[target_c]:
                            del colour_edges[target_c]
                        # Add to colour c
                        if c not in colour_edges:
                            colour_edges[c] = []
                        colour_edges[c].append(idx)
                        # Update vertex_colour_blocks
                        for x in conflict_verts[u]:
                            vertex_colour_blocks[x].discard(target_c)
                            vertex_colour_blocks[x].add(c)
                        for x in conflict_verts[v]:
                            vertex_colour_blocks[x].discard(target_c)
                            vertex_colour_blocks[x].add(c)
                        colour[idx] = c
                        moved = True
                        improved = True
                        break
                if not moved:
                    # Try swap: find an edge in colour c that can swap with idx
                    # This is more complex; skip for now
                    pass
            
            # Update ncolours
            if colour_edges:
                ncolours = max(colour_edges.keys()) + 1
            else:
                ncolours = 0
        
        return colour, colour_edges

    def solve_instance(n, d, seed):
        edges = gen_instance(n, d, seed)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        best_colour = None
        best_count = float('inf')
        
        rng = Random(seed + 1000)
        num_restarts = 200
        
        for r in range(num_restarts):
            if time.time() > DEADLINE - 5:
                break
            colour, colour_edges = greedy_first_fit(edges, adj, rng)
            ncolours = max(colour) + 1 if colour else 0
            
            if ncolours < best_count:
                best_colour = colour[:]
                best_count = ncolours
            
            # Local compression
            colour, colour_edges = local_compress(edges, adj, colour, colour_edges, rng, max_iters=100)
            ncolours = max(colour) + 1 if colour else 0
            if ncolours < best_count:
                best_colour = colour[:]
                best_count = ncolours
        
        return best_colour

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 10:
            # Fallback: simple greedy
            edges = gen_instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            rng = Random(spec[2])
            colour, _ = greedy_first_fit(edges, adj, rng)
            out.append(colour)
        else:
            colour = solve_instance(*spec)
            out.append(colour)
    
    return out
