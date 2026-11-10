import random
import itertools

N = 16

def construct():
    """
    Construct edge coloring of K_16 with no 4-cycle having 4 distinct colors,
    maximizing the number of distinct colors used.
    """
    # Number of edges
    num_edges = N * (N - 1) // 2
    
    # Helper: get edge index for (i, j)
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * N - i * (i + 1) // 2 + j - i - 1
    
    # Helper: get all 4-cycles and their edges
    # A 4-cycle on vertices (a,b,c,d) in order has edges (a,b),(b,c),(c,d),(d,a)
    # We need to check all 3 ways to form a 4-cycle on 4 vertices:
    # For 4 vertices {v0,v1,v2,v3}, the 3 cycles are:
    # (v0,v1,v2,v3), (v0,v1,v3,v2), (v0,v2,v1,v3)
    # Actually, for 4 vertices, there are 3 distinct 4-cycles (up to rotation/reflection)
    
    def check_valid(colors):
        """Check if no 4-cycle has 4 distinct colors"""
        for quad in itertools.combinations(range(N), 4):
            v0, v1, v2, v3 = quad
            # Three 4-cycles on these 4 vertices
            cycles = [
                [(v0,v1), (v1,v2), (v2,v3), (v3,v0)],
                [(v0,v1), (v1,v3), (v3,v2), (v2,v0)],
                [(v0,v2), (v2,v1), (v1,v3), (v3,v0)]
            ]
            for cycle in cycles:
                cols = set()
                for (a, b) in cycle:
                    cols.add(colors[edge_idx(a, b)])
                if len(cols) == 4:
                    return False
        return True
    
    # Start with a structured coloring: partition into 4 groups of 4
    # Color inter-group edges based on group pair, intra-group with fixed colors
    colors = [0] * num_edges
    
    # Assign group to each vertex
    group = [i // 4 for i in range(N)]
    
    # Intra-group: color 0
    # Inter-group: use colors 1,2,3,4,5,6,7,8,9 for the 6 group pairs
    group_pairs = {}
    gp_idx = 1
    for g1 in range(4):
        for g2 in range(g1+1, 4):
            group_pairs[(g1, g2)] = gp_idx
            gp_idx += 1
    
    for i in range(N):
        for j in range(i+1, N):
            idx = edge_idx(i, j)
            if group[i] == group[j]:
                colors[idx] = 0
            else:
                g1, g2 = sorted([group[i], group[j]])
                colors[idx] = group_pairs[(g1, g2)]
    
    # This gives 1 + 6 = 7 colors
    # Now try to improve by splitting colors
    
    # Local search: try to recolor edges to introduce new colors while maintaining validity
    best_colors = colors[:]
    best_num_colors = len(set(best_colors))
    
    # Try multiple restarts with different initial structures
    for restart in range(50):
        if restart == 0:
            current = best_colors[:]
        else:
            # Random perturbation
            current = best_colors[:]
            for _ in range(10):
                e = random.randint(0, num_edges-1)
                current[e] = random.randint(0, 119)
        
        improved = True
        while improved and len(set(current)) < 120:
            improved = False
            # Try to change random edges to new colors
            edges_to_try = list(range(num_edges))
            random.shuffle(edges_to_try)
            
            for e in edges_to_try:
                old_color = current[e]
                # Try a new color not currently used
                used_colors = set(current)
                new_color = None
                for c in range(120):
                    if c not in used_colors:
                        new_color = c
                        break
                if new_color is None:
                    break
                
                current[e] = new_color
                if check_valid(current):
                    improved = True
                    best_colors = current[:]
                    best_num_colors = len(set(best_colors))
                else:
                    current[e] = old_color
        
        if len(set(current)) > best_num_colors:
            best_colors = current[:]
            best_num_colors = len(set(best_colors))
    
    return best_colors
