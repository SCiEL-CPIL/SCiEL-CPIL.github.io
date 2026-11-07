import math

def construct():
    N = 256
    # 0-indexed: vertex i represents number i+1
    weights = [i + 1 for i in range(N)]
    
    # Precompute conflict neighbors (gcd > 1)
    # neighbors[i] is a bitmask of j such that gcd(i+1, j+1) > 1
    neighbors = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i != j:
                vj = j + 1
                if math.gcd(vi, vj) > 1:
                    mask |= (1 << j)
        neighbors[i] = mask
        
    # For the coloring bound, we work on the complement graph (coprime graph).
    # In the complement graph, an edge exists if gcd == 1.
    # We want a Max Weight Independent Set in G (conflict graph).
    # This is a Max Weight Clique in G_bar (complement).
    # We color G_bar.
    # Two vertices have the same color if they are NOT connected in G_bar.
    # Not connected in G_bar means they ARE connected in G (conflict).
    # So same color => conflict => cannot be in independent set together.
    # Thus, at most one vertex of each color can be in the solution.
    # Upper bound = sum of max weights in each color class.

    best_sum = 0
    best_set = []

    # Greedy coloring of the subgraph induced by 'mask' in the complement graph.
    # Returns the upper bound on the weight of a clique in the complement (independent set in G).
    def get_bound(mask):
        # We color the complement graph.
        # Adjacency in complement: j is neighbor of i in complement if NOT (j in neighbors[i]) and i != j.
        # To speed up, we can just iterate through available vertices.
        
        # Order vertices for coloring.
        # A good heuristic is to order by degree in the complement graph descending, or weight descending.
        # Let's use weight descending for simplicity.
        
        available_indices = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            available_indices.append(idx)
            m ^= lsb
            
        # Sort by weight descending
        available_indices.sort(key=lambda x: weights[x], reverse=True)
        
        colors = {} # index -> color
        color_max_weights = [] # list of max weights for each color
        
        for idx in available_indices:
            # Find the smallest color that can be assigned to idx
            # idx cannot share a color with any vertex that is connected to it in the complement graph.
            # Connected in complement => gcd(idx, other) == 1.
            # So idx CANNOT share a color with any 'other' in available_indices such that gcd(idx+1, other+1) == 1.
            # This means idx CAN share a color with any 'other' such that gcd(idx+1, other+1) > 1.
            
            # Let's check used colors.
            # For each color c, check if there is any vertex v with color c such that gcd(idx, v) == 1.
            # If yes, color c is forbidden.
            
            forbidden_colors = set()
            # Iterate through all vertices that already have a color
            for other_idx, c in colors.items():
                # Check if idx and other_idx are connected in complement (gcd == 1)
                if math.gcd(idx + 1, other_idx + 1) == 1:
                    forbidden_colors.add(c)
                    
            # Find smallest non-forbidden color
            c = 0
            while c in forbidden_colors:
                c += 1
                
            colors[idx] = c
            if c >= len(color_max_weights):
                color_max_weights.append(weights[idx])
            else:
                if weights[idx] > color_max_weights[c]:
                    color_max_weights[c] = weights[idx]
                    
        return sum(color_max_weights)

    def solve(mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        if mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
            
        # Calculate bound
        bound = get_bound(mask)
        if current_sum + bound <= best_sum:
            return
            
        # Select a vertex to branch on.
        # Heuristic: Pick vertex with highest weight.
        m = mask
        v = -1
        max_w = -1
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                v = idx
            m ^= lsb
            
        # Branch 1: Include v
        # Remove v and all its neighbors in G (conflict graph)
        remove_mask = neighbors[v] | (1 << v)
        new_mask = mask & ~remove_mask
        solve(new_mask, current_sum + weights[v], current_set + [v])
        
        # Branch 2: Exclude v
        new_mask_ex = mask & ~(1 << v)
        solve(new_mask_ex, current_sum, current_set)

    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    result = [i + 1 for i in best_set]
    return [int(x) for x in result]
