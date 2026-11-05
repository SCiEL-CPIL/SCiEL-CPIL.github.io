import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin

    # 1. Try known specific cages for small k, g
    # The 4-cage of girth 10 has 320 vertices.
    # It is conjectured/known to be the circulant C_320(1, 79) or similar.
    # Let's verify this and other candidates.
    
    if k == 4 and g == 10:
        # Try a broader set of circulant candidates
        candidates = []
        # Known good candidates
        candidates.append((320, [1, 79]))
        candidates.append((320, [1, 81]))
        candidates.append((320, [1, 77]))
        candidates.append((320, [1, 83]))
        candidates.append((320, [1, 101]))
        candidates.append((320, [1, 119]))
        
        # Try more candidates systematically
        for s in range(2, 160):
            if s == 1 or s == 79 or s == 81 or s == 77 or s == 83 or s == 101 or s == 119:
                continue
            candidates.append((320, [1, s]))
        
        random.shuffle(candidates)
        
        for n, steps in candidates:
            if time.time() > deadline:
                break
            if _check_circulant_girth_fast(n, steps, k, g):
                return n, _get_circulant_edges(n, steps)
        
        # If no circulant worked, fall back to LUW
        return _luw_construct(k, g)

    # General fallback: LUW construction
    return _luw_construct(k, g)

def _check_circulant_girth_fast(n, steps, k, g):
    """
    Fast girth check for circulant graph C_n(steps).
    Uses BFS with state = (position, last_step_direction) to avoid backtracking.
    """
    if len(steps) * 2 != k:
        return False
    
    # Each vertex has 2*len(steps) neighbors: +s and -s for each s in steps
    # State: (pos, last_step_idx, last_sign) where last_step_idx is index in steps, last_sign is +1 or -1
    # We start from vertex 0, with no previous step
    
    # BFS to find shortest non-backtracking closed walk
    # If shortest closed walk < g, girth < g
    
    # visited[pos][step_idx][sign] = minimum depth to reach this state
    # step_idx in range(len(steps)), sign in {0, 1} for +1, -1
    # Use -1 for "no previous step"
    
    INF = float('inf')
    # visited[pos][step_idx][sign] where step_idx is -1 for start
    # Let's use a dict or 3D array
    # pos in 0..n-1, step_idx in -1..len(steps)-1, sign in 0..1
    
    num_steps = len(steps)
    # Map step_idx to 0..num_steps, where 0 means "no previous step"
    # sign: 0 for +1, 1 for -1
    
    visited = [[INF] * (num_steps + 1) * 2 for _ in range(n)]
    
    # Queue: (pos, step_idx, sign, depth)
    # step_idx = -1 for start, sign = -1 for start
    queue = [(0, -1, -1, 0)]
    visited[0][(-1) * 2 + 0][0] = 0  # Hmm, this indexing is getting complex
    
    # Let me use a different approach: visited as a set of (pos, step_idx, sign) with min depth
    visited = {}
    queue = [(0, -1, -1, 0)]
    visited[(0, -1, -1)] = 0
    
    while queue:
        pos, prev_idx, prev_sign, depth = queue.pop(0)
        
        if depth > 0 and pos == 0:
            if depth < g:
                return False
            continue
        
        if depth >= g - 1:
            continue
        
        for i, s in enumerate(steps):
            for sign_val in [1, -1]:
                # Backtracking check: can't go back the way we came
                if prev_idx is not None and prev_idx == i and prev_sign == -sign_val:
                    continue
                
                new_pos = (pos + sign_val * s) % n
                new_idx = i
                new_sign = sign_val
                
                state = (new_pos, new_idx, new_sign)
                new_depth = depth + 1
                
                if state not in visited or visited[state] > new_depth:
                    visited[state] = new_depth
                    queue.append((new_pos, new_idx, new_sign, new_depth))
    
    return True

def _get_circulant_edges(n, steps):
    edges = []
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges

def _luw_construct(k, g):
    # Original seed logic
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
