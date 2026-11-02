"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph with girth at least g using Lazebnik-Ustimenko-Woldar construction."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    q = k
    # For odd d: girth >= d+5, for even d: girth >= d+4
    # We want girth >= g, so:
    # If we use odd d: d >= g-5
    # If we use even d: d >= g-4
    # Choose the smallest valid d
    d_candidates = []
    for d in range(2, 20):
        if d % 2 == 1:  # odd
            if d + 5 >= g:
                d_candidates.append(d)
        else:  # even
            if d + 4 >= g:
                d_candidates.append(d)
    if not d_candidates:
        d = max(3, g - 5)
    else:
        d = min(d_candidates)
    
    # Setup GF(q) arithmetic
    if q == 4:
        # GF(4) = {0, 1, a, a+1} where a^2 = a+1
        # Represent as {0, 1, 2, 3} where 2=a, 3=a+1
        # Addition: XOR
        # Multiplication table
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def gf_add(x, y):
            return x ^ y
        def gf_sub(x, y):
            return x ^ y  # In characteristic 2, subtraction = addition
        def gf_mul(x, y):
            return mul_table[x][y]
    else:
        # q is prime (for small k values like 3, 5, 7)
        def gf_add(x, y):
            return (x + y) % q
        def gf_sub(x, y):
            return (x - y) % q
        def gf_mul(x, y):
            return (x * y) % q
    
    # Build the rule for the LWU construction
    # rule[c] = (a, b) means: y[c] = y[a] * x[b] (with appropriate side handling)
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:  # r == 3
            rule[c] = (1, c0 + 1)
    
    # BFS from the zero point to find the connected component
    # Vertex representation: (side, tuple of d+1 elements, first element is None for side 0, or a value for side 1)
    # Actually, let's use: side 0 vertices are (0, y_1, y_2, ..., y_d) where y_i are in GF(q)
    # side 1 vertices are (1, x_1, x_2, ..., x_d) where x_i are in GF(q)
    
    # Encode vertex as integer for efficiency
    # side (1 bit) + d elements each needing log2(q) bits
    if q == 4:
        bits_per_elem = 2
    elif q == 3:
        bits_per_elem = 2  # 0,1,2 need 2 bits
    elif q == 5:
        bits_per_elem = 3
    else:
        bits_per_elem = (q - 1).bit_length()
    
    def encode_vertex(side, coords):
        """Encode (side, coords) to an integer."""
        code = side
        for i in range(d):
            code = (code << bits_per_elem) | coords[i]
        return code
    
    def decode_vertex(code):
        """Decode integer to (side, coords)."""
        coords = [0] * d
        for i in range(d - 1, -1, -1):
            coords[i] = code & ((1 << bits_per_elem) - 1)
            code >>= bits_per_elem
        side = code
        return side, tuple(coords)
    
    # Neighbour generation
    def get_neighbours(side, coords):
        """Generate all neighbours of a vertex."""
        neighbours = []
        if side == 0:
            # Point vertex: coords = (y_1, ..., y_d)
            # For each x_1 in GF(q), determine x_2, ..., x_d
            # Relation: for c=2..d, x[c] - y[c] = x[a] * y[b] where (a,b) = rule[c]
            # So x[c] = y[c] + x[a] * y[b]
            for x1 in range(q):
                x = [0] * (d + 1)  # 1-indexed: x[1]..x[d]
                x[1] = x1
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # x[c] = y[c] + x[a] * y[b]
                    x[c] = gf_add(coords[c-1], gf_mul(x[a], coords[b-1]))
                neighbours.append((1, tuple(x[1:])))
        else:
            # Line vertex: coords = (x_1, ..., x_d)
            # For each y_1 in GF(q), determine y_2, ..., y_d
            # Relation: for c=2..d, x[c] - y[c] = x[a] * y[b]
            # So y[c] = x[c] - x[a] * y[b]
            for y1 in range(q):
                y = [0] * (d + 1)  # 1-indexed: y[1]..y[d]
                y[1] = y1
                for c in range(2, d + 1):
                    a, b = rule[c]
                    # y[c] = x[c] - x[a] * y[b]
                    y[c] = gf_sub(coords[c-1], gf_mul(coords[a-1], y[b]))
                neighbours.append((0, tuple(y[1:])))
        return neighbours
    
    # BFS
    start_side = 0
    start_coords = tuple([0] * d)
    start_code = encode_vertex(start_side, start_coords)
    
    visited = {start_code}
    queue = [start_code]
    vertex_list = [start_code]
    
    edge_set = set()
    edges = []
    
    idx = 0
    while idx < len(queue):
        code = queue[idx]
        idx += 1
        
        if time.time() > deadline:
            break
        
        side, coords = decode_vertex(code)
        for n_side, n_coords in get_neighbours(side, coords):
            n_code = encode_vertex(n_side, n_coords)
            
            # Add edge (only from side 0 to side 1 to avoid duplicates)
            if side == 0:
                if n_code not in edge_set:
                    edge_set.add(n_code)
                    edges.append((len(vertex_list), len(vertex_list) if n_code not in visited else -1))
                    # We'll fix edge indices later
            
            if n_code not in visited:
                visited.add(n_code)
                queue.append(n_code)
                vertex_list.append(n_code)
    
    # Now we have vertex_list, need to build proper edges
    # Rebuild edges with correct indices
    code_to_idx = {code: i for i, code in enumerate(vertex_list)}
    
    final_edges = []
    edge_set2 = set()
    for code in vertex_list:
        side, coords = decode_vertex(code)
        if side == 0:  # Only from point side to avoid duplicates
            for n_side, n_coords in get_neighbours(side, coords):
                n_code = encode_vertex(n_side, n_coords)
                if n_code in code_to_idx:
                    u = code_to_idx[code]
                    v = code_to_idx[n_code]
                    if u < v:
                        key = (u, v)
                    else:
                        key = (v, u)
                    if key not in edge_set2:
                        edge_set2.add(key)
                        final_edges.append(key)
    
    n = len(vertex_list)
    return n, final_edges
# EVOLVE-BLOCK-END
