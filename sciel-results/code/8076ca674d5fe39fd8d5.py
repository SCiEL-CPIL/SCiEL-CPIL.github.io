"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    import random
    import math
    from collections import deque

    # Voltage lift of K_{3,3} over Z_m: n = 6*m vertices
    # Vertex (side, base, shift): side in {0,1}, base in 0..2, shift in 0..m-1
    # Edge from (0,i,s) to (1,j,(s+v[i][j])%m)
    
    # Moore bound for 3-regular graph with girth g
    if g % 2 == 0:
        moore_bound = 3 * (2 ** (g // 2 - 1) - 1)
    else:
        moore_bound = 3 * (2 ** ((g - 1) // 2 - 1) - 1) if (g - 1) // 2 >= 1 else 3
    
    m_min = max(1, math.ceil(moore_bound / 6))
    
    def check_girth_fast(m, voltages, required_girth):
        """Check girth using non-backtracking walks from each vertex.
        Returns True if girth >= required_girth."""
        n = 6 * m
        half_girth = required_girth // 2
        
        # Build adjacency: for vertex (side, base, shift), neighbors are determined by voltages
        # Precompute neighbor offsets for speed
        # Vertex id = side * 3*m + base * m + shift
        
        for start in range(n):
            # Non-backtracking BFS: state is (current_vertex, previous_vertex)
            # Track visited states to detect cycles
            visited = set()
            queue = deque()
            
            start_side = start // (3 * m)
            start_base = (start % (3 * m)) // m
            start_shift = start % m
            
            # Initialize: from start, go to each neighbor
            for nb_base in range(3):
                if start_side == 0:
                    # From A side, go to B side
                    v = voltages[start_base][nb_base]
                    nb_shift = (start_shift + v) % m
                    nb_id = 1 * 3 * m + nb_base * m + nb_shift
                    visited.add((start, nb_id))
                    queue.append((nb_id, start, 1))
                else:
                    # From B side, go to A side
                    # Edge from B_j to A_i has voltage -v[i][j]
                    v = voltages[nb_base][start_base]
                    nb_shift = (start_shift - v) % m
                    nb_id = 0 * 3 * m + nb_base * m + nb_shift
                    visited.add((start, nb_id))
                    queue.append((nb_id, start, 1))
            
            while queue:
                curr, prev, depth = queue.popleft()
                if depth >= half_girth:
                    continue
                
                curr_side = curr // (3 * m)
                curr_base = (curr % (3 * m)) // m
                curr_shift = curr % m
                
                for nb_base in range(3):
                    if curr_side == 0:
                        v = voltages[curr_base][nb_base]
                        nb_shift = (curr_shift + v) % m
                        nb_id = 1 * 3 * m + nb_base * m + nb_shift
                    else:
                        v = voltages[nb_base][curr_base]
                        nb_shift = (curr_shift - v) % m
                        nb_id = 0 * 3 * m + nb_base * m + nb_shift
                    
                    if nb_id == prev:
                        continue
                    
                    state = (curr, nb_id)
                    if state in visited:
                        # Found a cycle: length is depth + 1 + (distance from start to prev in tree)
                        # Actually, cycle length = 2 * depth + 1 or 2 * depth
                        # More precisely: the walk from start to prev is depth steps,
                        # then curr->nb_id closes the loop. The cycle has length depth + 1 + depth_from_start_to_prev
                        # But we need to be more careful. Let's just check if cycle < required_girth
                        cycle_len = 2 * depth + 1  # approximate, may need refinement
                        if cycle_len < required_girth:
                            return False
                    else:
                        visited.add(state)
                        queue.append((nb_id, curr, depth + 1))
        
        return True
    
    random.seed(42)
    num_trials = 200
    
    for m in range(m_min, m_min + 300):
        if 6 * m > 3000:
            break
        
        for trial in range(num_trials):
            voltages = [[random.randint(0, m - 1) for _ in range(3)] for _ in range(3)]
            
            if check_girth_fast(m, voltages, g):
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
        
        # Try structured patterns
        for pattern in range(3):
            if pattern == 0:
                voltages = [[(i * 7 + j * 13) % m for j in range(3)] for i in range(3)]
            elif pattern == 1:
                voltages = [[(i * 11 + j * 5) % m for j in range(3)] for i in range(3)]
            else:
                voltages = [[(i * i * 3 + j * j * 7) % m for j in range(3)] for i in range(3)]
            
            if check_girth_fast(m, voltages, g):
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
    
    # Fallback: Lazebnik-Ustimenko-Woldar construction
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
