"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

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

    def get_conflicts(edge, adj, n):
        """Get all vertices that conflict with this edge (endpoints + their neighbors)."""
        u, v = edge
        near = {u, v}
        near.update(adj[u])
        near.update(adj[v])
        return near

    def greedy_color(edges, adj, n, color_order=None):
        """Greedy edge coloring with induced matching constraint."""
        if color_order is None:
            color_order = list(range(len(edges)))
        
        colour = {}
        # For each color, track which vertices are "used" by edges of that color
        # and which edges are in that color (to check if new edge is joined by an edge)
        color_vertices = defaultdict(set)  # color -> set of vertices used
        color_edges = defaultdict(set)  # color -> set of edge indices
        
        for idx in color_order:
            u, v = edges[idx]
            near = get_conflicts((u, v), adj, n)
            
            best_color = None
            for c in range(len(edges)):
                # Check if any vertex in near is already used by color c
                if color_vertices[c] & near:
                    continue
                # Check if edge (u,v) is joined by an edge in color c
                # i.e., if there's an edge (a,b) in color c where a is neighbor of u or v,
                # or b is neighbor of u or v, but (a,b) != (u,v) and they don't share a vertex
                joined = False
                for (a, b) in color_edges[c]:
                    if (a == u or a == v or b == u or b == v):
                        continue  # shares a vertex, already handled above
                    # Check if (a,b) is joined by (u,v)
                    # They are joined if there's an edge between {a,b} and {u,v}
                    # i.e., a-u, a-v, b-u, or b-v is an edge
                    if a in adj.get(u, set()) or a in adj.get(v, set()) or \
                       b in adj.get(u, set()) or b in adj.get(v, set()):
                        joined = True
                        break
                if not joined:
                    best_color = c
                    break
            
            if best_color is None:
                best_color = len(color_vertices)
            
            colour[idx] = best_color
            color_vertices[best_color].add(u)
            color_vertices[best_color].add(v)
            color_edges[best_color].add((u, v))
        
        return colour

    def local_improve(colour, edges, adj, n, max_iters=50):
        """Try to reduce number of colors by recoloring edges."""
        num_colors = max(colour.values()) + 1 if colour else 0
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try to recolor each edge to a lower color if possible
            for idx in random.sample(range(len(edges)), len(edges)):
                u, v = edges[idx]
                current_color = colour[idx]
                near = get_conflicts((u, v), adj, n)
                
                # Try all colors from 0 to current_color - 1
                for c in range(current_color):
                    # Temporarily remove edge from its color class
                    # Check if we can assign color c
                    can_assign = True
                    
                    # Check vertex conflicts
                    for other_idx, other_color in colour.items():
                        if other_idx == idx or other_color != c:
                            continue
                        ou, ov = edges[other_idx]
                        if ou in near or ov in near:
                            can_assign = False
                            break
                    if not can_assign:
                        continue
                    
                    # Check edge-joining conflicts
                    for other_idx, other_color in colour.items():
                        if other_idx == idx or other_color != c:
                            continue
                        ou, ov = edges[other_idx]
                        if ou == u or ou == v or ov == u or ov == v:
                            continue  # shares vertex, already handled
                        if ou in adj.get(u, set()) or ou in adj.get(v, set()) or \
                           ov in adj.get(u, set()) or ov in adj.get(v, set()):
                            can_assign = False
                            break
                    
                    if can_assign:
                        colour[idx] = c
                        improved = True
                        break
            
            if not improved:
                break
        
        return num_colors

    def color_instance(edges, adj, n):
        """Color a single instance with multiple random orders and local improvement."""
        m = len(edges)
        best_colour = None
        best_num_colors = float('inf')
        
        # Try multiple random orderings
        num_tries = 20
        rng = random.Random(42)
        
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            
            # Random order
            order = list(range(m))
            rng.shuffle(order)
            
            colour = greedy_color(edges, adj, n, order)
            num_colors = max(colour.values()) + 1 if colour else 0
            
            # Local improvement
            num_colors = local_improve(colour, edges, adj, n)
            
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_colour = dict(colour)
        
        # Return colors in sorted edge order
        if best_colour is None:
            best_colour = greedy_color(edges, adj, n)
            best_num_colors = max(best_colour.values()) + 1 if best_colour else 0
        
        return [best_colour[i] for i in range(m)]

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        colours = color_instance(edges, adj, n)
        out.append(colours)
    
    return out
# EVOLVE-BLOCK-END
