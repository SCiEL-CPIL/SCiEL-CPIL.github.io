import random
import time

def construct():
    n = 20
    num_colors = 20
    # List of all potential edges in K20
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(all_edges)
    
    # Precompute all 4-cycles
    # A 4-cycle is defined by 4 distinct vertices. There are 3 ways to form a cycle from 4 vertices.
    # Vertices: a, b, c, d. Cycles: (a,b,c,d), (a,b,d,c), (a,c,b,d)
    # Edges for (a,b,c,d): (a,b), (b,c), (c,d), (d,a)
    cycles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    # Cycle 1: i-j-k-l-i
                    cycles.append([(i, j), (j, k), (k, l), (l, i)])
                    # Cycle 2: i-j-l-k-i
                    cycles.append([(i, j), (j, l), (l, k), (k, i)])
                    # Cycle 3: i-k-j-l-i
                    cycles.append([(i, k), (k, j), (j, l), (l, i)])
                    
    # Map edge to index for quick lookup
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    
    # State: colors for each edge. -1 means edge is removed.
    # Start with all edges present, random colors
    colors = [random.randint(0, num_colors - 1) for _ in range(num_edges)]
    
    def get_penalty():
        """Calculate total penalty.
        Penalty = (num improper adjacent pairs) + (num non-rainbow 4-cycles)
        """
        penalty = 0
        
        # Check proper edge coloring (adjacent edges)
        # Build adjacency list of edges
        # For each vertex, collect incident edges
        incident_edges = [[] for _ in range(n)]
        for i, (u, v) in enumerate(all_edges):
            if colors[i] != -1:
                incident_edges[u].append(i)
                incident_edges[v].append(i)
        
        for u in range(n):
            edges_at_u = incident_edges[u]
            len_e = len(edges_at_u)
            for i in range(len_e):
                c1 = colors[edges_at_u[i]]
                for j in range(i + 1, len_e):
                    c2 = colors[edges_at_u[j]]
                    if c1 == c2:
                        penalty += 1
                        
        # Check 4-cycles
        for cycle in cycles:
            c_list = []
            valid = True
            for e in cycle:
                idx = edge_to_idx[e]
                if colors[idx] == -1:
                    valid = False
                    break
                c_list.append(colors[idx])
            if valid:
                if len(set(c_list)) < 4:
                    penalty += 1
                    
        return penalty

    def get_edge_conflicts(edge_idx):
        """Return number of conflicts involving this specific edge."""
        if colors[edge_idx] == -1:
            return 0
            
        u, v = all_edges[edge_idx]
        conflicts = 0
        my_color = colors[edge_idx]
        
        # Adjacent edges
        for other_idx in range(num_edges):
            if other_idx == edge_idx:
                continue
            if colors[other_idx] == -1:
                continue
            ou, ov = all_edges[other_idx]
            if (ou == u or ou == v or ov == u or ov == v) and (ou != ov):
                # Check if they share a vertex
                if (ou == u or ou == v or ov == u or ov == v):
                    if colors[other_idx] == my_color:
                        conflicts += 1

        # 4-cycles containing this edge
        # An edge (u,v) is in a 4-cycle (u,v,w,x) or (u,v,x,w) etc.
        # It's faster to just iterate all cycles and check if edge is in it.
        # With ~14k cycles, this is fast enough for local updates if not too many iterations.
        for cycle in cycles:
            if edge_idx in [edge_to_idx[e] for e in cycle]:
                c_list = []
                valid = True
                for e in cycle:
                    idx = edge_to_idx[e]
                    if colors[idx] == -1:
                        valid = False
                        break
                    c_list.append(colors[idx])
                if valid and len(set(c_list)) < 4:
                    conflicts += 1
                    
        return conflicts

    # Simulated Annealing
    start_time = time.time()
    time_limit = 110.0 # Safety margin
    
    best_colors = colors[:]
    best_penalty = get_penalty()
    
    current_penalty = best_penalty
    temperature = 10.0
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        if temperature < 0.01:
            break
            
        # Pick a random edge to modify
        edge_idx = random.randint(0, num_edges - 1)
        old_color = colors[edge_idx]
        
        # Option 1: Change color
        # Option 2: Remove edge (set to -1)
        # Option 3: Add edge (if -1)
        
        if colors[edge_idx] == -1:
            # Try adding it with a random color
            new_color = random.randint(0, num_colors - 1)
            colors[edge_idx] = new_color
            new_penalty = get_penalty()
            
            delta = new_penalty - current_penalty
            if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                current_penalty = new_penalty
                if current_penalty < best_penalty:
                    best_penalty = current_penalty
                    best_colors = colors[:]
            else:
                colors[edge_idx] = old_color
        else:
            # Try changing color
            if random.random() < 0.9: # 90% recolor, 10% remove
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    new_color = (new_color + 1) % num_colors
                colors[edge_idx] = new_color
                new_penalty = get_penalty()
                
                delta = new_penalty - current_penalty
                if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                    current_penalty = new_penalty
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_colors = colors[:]
                else:
                    colors[edge_idx] = old_color
            else:
                # Remove edge
                colors[edge_idx] = -1
                new_penalty = get_penalty()
                
                delta = new_penalty - current_penalty
                if delta < 0 or random.random() < pow(2.71828, -delta / temperature):
                    current_penalty = new_penalty
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_colors = colors[:]
                else:
                    colors[edge_idx] = old_color
                    
        temperature *= cooling
        
    # Restore best colors
    colors = best_colors[:]
    
    # Construct output
    result = []
    for i, (u, v) in enumerate(all_edges):
        if colors[i] != -1:
            result.append([u, v, colors[i]])
            
    return result

