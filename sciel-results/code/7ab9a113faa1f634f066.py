"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    def build_conflict_bitmasks(edges, adj):
        """Build bitmask conflict representation for fast color checking."""
        n_edges = len(edges)
        conflict = [0] * n_edges
        for i in range(n_edges):
            ui, vi = edges[i]
            for j in range(i + 1, n_edges):
                uj, vj = edges[j]
                # Share vertex
                if ui in (uj, vj) or vi in (uj, vj):
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
                # Joined by edge
                elif ui in adj[uj] or ui in adj[vj] or vi in adj[uj] or vi in adj[vj]:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        return conflict

    def greedy_color_bitmask(edges, conflict, order=None):
        """Greedy coloring using bitmask conflict representation."""
        n_edges = len(edges)
        if order is None:
            order = list(range(n_edges))
        
        colour = [0] * n_edges
        # For each color, track which edges use it as a bitmask
        color_edges = {}  # color -> bitmask of edges with that color
        max_color = -1
        
        for idx in order:
            # Find which colors are taken by conflicting edges
            taken_mask = 0
            for j in range(n_edges):
                if (conflict[idx] >> j) & 1:
                    taken_mask |= (1 << colour[j])
            
            # Assign smallest available color
            c = 0
            while (taken_mask >> c) & 1:
                c += 1
            
            colour[idx] = c
            if c not in color_edges:
                color_edges[c] = 0
            color_edges[c] |= (1 << idx)
            
            if c > max_color:
                max_color = c
        
        return colour, max_color + 1

    def local_improve_bitmask(edges, conflict, colour, max_colors, max_iter=10000, rng=None):
        """Try to reduce max color by reassigning edges using bitmasks."""
        if rng is None:
            rng = random.Random()
        
        n_edges = len(edges)
        current_max = max_colors - 1
        improved = True
        iterations = 0
        
        while improved and iterations < max_iter and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            if current_max <= 0:
                break
            
            # Find edges with the highest color
            high_edges = [i for i in range(n_edges) if colour[i] == current_max]
            if not high_edges:
                current_max -= 1
                continue
            
            rng.shuffle(high_edges)
            
            for e_idx in high_edges:
                # Try to reassign this edge to a lower color
                # Check which colors are taken by conflicting edges
                taken_mask = 0
                for j in range(n_edges):
                    if j != e_idx and ((conflict[e_idx] >> j) & 1):
                        taken_mask |= (1 << colour[j])
                
                # Find smallest available color
                new_c = 0
                while (taken_mask >> new_c) & 1:
                    new_c += 1
                
                if new_c < colour[e_idx]:
                    colour[e_idx] = new_c
                    # Update max if needed
                    if current_max == max(colour):
                        current_max = max(colour)
                    improved = True
                    break
        
        return colour, current_max + 1

    def try_merge_colors(edges, conflict, colour, max_colors, rng=None):
        """Try to merge two color classes by reassigning edges."""
        if rng is None:
            rng = random.Random()
        
        n_edges = len(edges)
        if max_colors <= 1:
            return colour, max_colors
        
        # Try to merge the highest color into a lower one
        high_color = max_colors - 1
        high_edges = [i for i in range(n_edges) if colour[i] == high_color]
        
        if not high_edges:
            return colour, max_colors - 1
        
        # Try each lower color as a target
        for target_color in range(high_color):
            # Check if we can reassign all high_color edges to target_color
            # or to other lower colors
            success = True
            new_colour = colour[:]
            
            for e_idx in high_edges:
                # Find available colors (excluding high_color since we're removing it)
                taken_mask = 0
                for j in range(n_edges):
                    if j != e_idx and ((conflict[e_idx] >> j) & 1):
                        # If j currently has high_color, it will be reassigned too
                        # For now, just check current colors
                        taken_mask |= (1 << new_colour[j])
                
                # Find smallest available color below high_color
                new_c = 0
                while new_c < high_color and (taken_mask >> new_c) & 1:
                    new_c += 1
                
                if new_c < high_color:
                    new_colour[e_idx] = new_c
                else:
                    success = False
                    break
            
            if success:
                return new_colour, high_color
        
        return colour, max_colors

    def multi_order_greedy(edges, adj, n_trials=20, time_budget=10.0):
        """Try multiple random orderings and keep the best result."""
        n_edges = len(edges)
        conflict = build_conflict_bitmasks(edges, adj)
        
        # Compute constraint degree for initial ordering
        edge_constraints = [0] * n_edges
        for i in range(n_edges):
            edge_constraints[i] = bin(conflict[i]).count('1')
        
        best_colour = None
        best_num_colors = float('inf')
        
        # Try constrained ordering first
        order = sorted(range(n_edges), key=lambda i: -edge_constraints[i])
        colour, num_colors = greedy_color_bitmask(edges, conflict, order)
        best_colour = colour[:]
        best_num_colors = num_colors
        
        # Try random orderings
        rng = random.Random(42)
        trial_time = time.time()
        
        for trial in range(n_trials):
            if time.time() - trial_time > time_budget:
                break
            
            # Random ordering with some bias toward constrained edges
            order = list(range(n_edges))
            for i in range(len(order) - 1, 0, -1):
                # Probabilistically swap to put more constrained edges earlier
                j = rng.randint(0, i)
                if edge_constraints[order[i]] > edge_constraints[order[j]] and rng.random() < 0.7:
                    order[i], order[j] = order[j], order[i]
            
            colour, num_colors = greedy_color_bitmask(edges, conflict, order)
            
            if num_colors < best_num_colors:
                best_colour = colour[:]
                best_num_colors = num_colors
        
        return best_colour, best_num_colors, conflict

    def polish(edges, conflict, colour, max_colors, time_budget=20.0):
        """Polish the coloring with local improvement and color merging."""
        rng = random.Random(123)
        start_polish = time.time()
        
        # First, do local improvement to reduce max color
        colour, max_colors = local_improve_bitmask(edges, conflict, colour, max_colors, 
                                                    max_iter=5000, rng=rng)
        
        # Then try to merge colors
        while time.time() - start_polish < time_budget and max_colors > 1:
            old_max = max_colors
            colour, max_colors = try_merge_colors(edges, conflict, colour, max_colors, rng=rng)
            
            # After merging, do another round of local improvement
            if max_colors < old_max:
                colour, max_colors = local_improve_bitmask(edges, conflict, colour, max_colors,
                                                            max_iter=2000, rng=rng)
        
        return colour, max_colors

    out = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() > deadline - 15:
            break
        
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        n_edges = len(edges)
        
        # Multi-order greedy to get a good initial coloring
        colour, num_colors, conflict = multi_order_greedy(edges, adj, n_trials=30, time_budget=15.0)
        
        # Polish with local improvement and color merging
        if time.time() < deadline - 20:
            colour, num_colors = polish(edges, conflict, colour, num_colors, time_budget=25.0)
        
        # Convert to list in original edge order
        result = [colour[i] for i in range(n_edges)]
        out.append(result)
    
    # If we ran out of time, fill remaining with simple greedy
    while len(out) < 3:
        spec = specs[len(out)]
        edges = instance(*spec)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        conflict = build_conflict_bitmasks(edges, adj)
        colour, _ = greedy_color_bitmask(edges, conflict)
        result = [colour[i] for i in range(len(edges))]
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
