"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque

    # Construct a 3-regular graph using a voltage lift of K_{3,3} over Z_m.
    # The base graph is K_{3,3} with vertices A={0,1,2} and B={0,1,2}.
    # Edges in base: (i, j) for all i in 0..2, j in 0..2.
    # Each edge gets a voltage in Z_m. The lifted graph has 6*m vertices.
    # Vertex encoding: side * 3 * m + base * m + shift
    # side 0 = A, side 1 = B
    # Edge from (0, i, s) to (1, j, (s + v[i][j]) % m)
    
    # For girth g, we need no cycles shorter than g.
    # With K_{3,3} lift, n = 6*m.
    # Moore bound for (3, g) cage: for g even, n >= 3*(2^(g/2-1)-1)
    
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    def check_girth_nb(m, voltages, required_girth):
        """Check girth using non-backtracking walks from each vertex.
        voltages: 3x3 matrix, voltages[i][j] is voltage on edge from A_i to B_j.
        Returns True if girth >= required_girth.
        """
        # Precompute neighbor lists in a flat array for speed
        # Vertex id = side * 3 * m + base * m + shift
        # For vertex (side, base, shift), neighbors are:
        #   if side==0: (1, j, (shift + voltages[base][j]) % m) for j in 0,1,2
        #   if side==1: (0, i, (shift - voltages[i][base]) % m) for i in 0,1,2
        
        n = 6 * m
        half = required_girth // 2
        
        # For each starting vertex, do non-backtracking BFS up to depth half
        # A non-backtracking walk is a sequence v0, v1, ..., vd where vi != vi-2
        # We track (current_vertex, previous_vertex) states
        
        for start in range(n):
            # State: (vertex, prev_vertex) where prev_vertex is the vertex we came from
            # Start: from 'start', we can go to any neighbor with prev = -1
            # But to avoid counting the trivial back-and-forth, we start with prev=-1
            
            # Use a dict/set for visited states: (vertex, prev) -> depth
            visited = {}
            queue = deque()
            
            # Initial: from start, go to each neighbor
            side_s = start // (3 * m)
            base_s = (start // m) % 3
            shift_s = start % m
            
            for j in range(3):
                if side_s == 0:
                    nb = 1 * 3 * m + j * m + (shift_s + voltages[base_s][j]) % m
                else:
                    nb = 0 * 3 * m + j * m + (shift_s - voltages[j][base_s]) % m
                state = (nb, start)
                if state not in visited:
                    visited[state] = 1
                    queue.append(state)
            
            while queue:
                cur, prev = queue.popleft()
                depth = visited[(cur, prev)]
                
                if depth >= half:
                    continue
                
                # Get neighbors of cur, excluding prev (non-backtracking)
                side_c = cur // (3 * m)
                base_c = (cur // m) % 3
                shift_c = cur % m
                
                for j in range(3):
                    if side_c == 0:
                        nb = 1 * 3 * m + j * m + (shift_c + voltages[base_c][j]) % m
                    else:
                        nb = 0 * 3 * m + j * m + (shift_c - voltages[j][base_c]) % m
                    
                    if nb == prev:
                        continue
                    
                    new_state = (nb, cur)
                    if new_state in visited:
                        # Cycle found of length depth + 1
                        cycle_len = depth + 1
                        if cycle_len < required_girth:
                            return False
                    else:
                        visited[new_state] = depth + 1
                        queue.append(new_state)
        
        return True
    
    def build_graph(m, voltages):
        n = 6 * m
        edges = []
        for i in range(3):
            for j in range(3):
                v = voltages[i][j]
                for s in range(m):
                    u = i * m + s
                    w = 3 * m + j * m + (s + v) % m
                    edges.append((u, w))
        return n, edges
    
    random.seed(42)
    
    # Try m values from m_min upward
    # For each m, try structured then random voltage assignments
    max_m = min(m_min + 300, 500)  # Cap m to keep n <= 3000
    
    for m in range(m_min, max_m + 1):
        if 6 * m > 3000:
            break
        
        # Try structured assignments first
        # Strategy 1: voltages[i][j] = (a*i + b*j) % m for various a, b
        found = False
        for a in range(1, min(m, 10)):
            for b in range(1, min(m, 10)):
                voltages = [[(a * i + b * j) % m for j in range(3)] for i in range(3)]
                if check_girth_nb(m, voltages, g):
                    n, edges = build_graph(m, voltages)
                    return n, edges
                found = True
        
        # Strategy 2: voltages[i][j] = (a*i*j + c) % m
        for a in range(1, min(m, 10)):
            for c in range(0, min(m, 5)):
                voltages = [[(a * i * j + c) % m for j in range(3)] for i in range(3)]
                if check_girth_nb(m, voltages, g):
                    n, edges = build_graph(m, voltages)
                    return n, edges
        
        # Random trials - fewer for larger m to save time
        num_trials = 30 if m < m_min + 50 else 15
        for trial in range(num_trials):
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            if check_girth_nb(m, voltages, g):
                n, edges = build_graph(m, voltages)
                return n, edges
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
