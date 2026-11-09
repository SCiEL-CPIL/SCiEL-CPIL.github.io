import time

def construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q)
    # For k=3, q=2.
    # Girth >= d+5 for odd d.
    # Target g=16 => d=11.
    # Vertices = 2 * 2^11 = 4096.
    
    q = k - 1
    if q != 2:
        # Fallback for other k if needed, but task is k=3
        # For now, assume k=3
        pass
        
    # Determine d
    # Need girth >= g
    # Girth(CD(d, q)) >= d + 5 (for odd d)
    # So d >= g - 5
    # d must be odd
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
        if d % 2 == 0: d += 1
        
    # Precompute rule
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute powers
    powers = [1] * (d + 1)
    for i in range(1, d + 1):
        powers[i] = powers[i-1] * q
    
    total_ids = q ** d
    
    # Precompute digits
    # digits[val * d + pos]
    digits_flat = [0] * (total_ids * d)
    for vid in range(total_ids):
        tmp = vid
        for pos in range(1, d + 1):
            digits_flat[vid * d + (pos - 1)] = tmp % q
            tmp //= q
            
    def get_digit(val, pos):
        return digits_flat[val * d + (pos - 1)]
        
    # Precompute neighbors
    # point_neighbors[p_id] -> list of l_ids
    # line_neighbors[l_id] -> list of p_ids
    point_neighbors = [[] for _ in range(total_ids)]
    line_neighbors = [[] for _ in range(total_ids)]
    
    for p_id in range(total_ids):
        for l1 in range(q):
            current_l_id = l1
            for c in range(2, d + 1):
                a, b = rule[c]
                p_c = get_digit(p_id, c)
                l_a = get_digit(current_l_id, a)
                p_b = get_digit(p_id, b)
                l_c = (p_c + l_a * p_b) % q
                current_l_id += l_c * powers[c-1]
            point_neighbors[p_id].append(current_l_id)
            
    for l_id in range(total_ids):
        for p1 in range(q):
            current_p_id = p1
            for c in range(2, d + 1):
                a, b = rule[c]
                l_c = get_digit(l_id, c)
                l_a = get_digit(l_id, a)
                p_b = get_digit(current_p_id, b)
                p_c = (l_c - l_a * p_b) % q
                current_p_id += p_c * powers[c-1]
            line_neighbors[l_id].append(current_p_id)
            
    # Build graph
    n_vertices = 2 * total_ids
    index = [-1] * n_vertices
    order = []
    edges = []
    
    start_vertex = 0
    index[start_vertex] = 0
    order.append(start_vertex)
    
    for vertex in order:
        side = vertex // total_ids
        vid = vertex % total_ids
        
        if side == 0:
            # Point side
            for l_id in point_neighbors[vid]:
                n_vertex = total_ids + l_id
                if index[n_vertex] == -1:
                    index[n_vertex] = len(order)
                    order.append(n_vertex)
                edges.append((index[vertex], index[n_vertex]))
        else:
            # Line side
            for p_id in line_neighbors[vid]:
                n_vertex = p_id
                if index[n_vertex] == -1:
                    index[n_vertex] = len(order)
                    order.append(n_vertex)
                edges.append((index[vertex], index[n_vertex]))
                
    n = len(order)
    return n, edges
