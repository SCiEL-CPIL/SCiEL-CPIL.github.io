import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s kill

    if k == 4 and g == 10:
        # Try known circulant candidates first (fast check)
        # These are sorted by likelihood of being valid or found in previous runs
        candidates = [
            (320, [1, 79]),
            (320, [1, 81]),
            (320, [1, 77]),
            (320, [1, 83]),
            (320, [1, 101]),
            (320, [1, 119]),
            (320, [1, 39]),
            (320, [1, 41]),
            (320, [1, 59]),
            (320, [1, 61]),
            (320, [1, 99]),
            (320, [1, 121]),
            (320, [1, 139]),
            (320, [1, 141]),
            (320, [1, 159]),
            (320, [1, 161]),
            # Add more potential candidates
            (320, [1, 49]),
            (320, [1, 51]),
            (320, [1, 69]),
            (320, [1, 71]),
            (320, [1, 89]),
            (320, [1, 91]),
            (320, [1, 109]),
            (320, [1, 111]),
            (320, [1, 129]),
            (320, [1, 131]),
            (320, [1, 149]),
            (320, [1, 151]),
            (320, [1, 179]),
            (320, [1, 181]),
        ]
        
        for n, steps in candidates:
            if time.time() > deadline:
                break
            if _check_circulant_girth_fast(n, steps, k, g):
                return n, _get_circulant_edges(n, steps)
        
        # Systematic search over more circulant candidates
        if time.time() < deadline - 5.0:
            result = _search_circulants(4, 10, 320, deadline)
            if result is not None:
                return result
        
        # Try a few more n values if time permits
        for n_try in [322, 324, 326, 328, 330]:
            if time.time() > deadline - 3.0:
                break
            if n_try % 2 == 0:
                result = _search_circulants(4, 10, n_try, deadline)
                if result is not None:
                    return result
        
        # Fallback: corrected LUW construction with d=6 for girth >= 10
        # This is a last resort and will result in a large n (score < 1.0)
        return _luw_construct(k, g)

    # General fallback: LUW construction
    return _luw_construct(k, g)

def _check_circulant_girth_fast(n, steps, k, g):
    """
    Fast girth check for circulant graph C_n(steps) using vectorized non-backtracking BFS.
    Returns True iff no cycle of length < g exists.
    """
    if len(steps) * 2 != k:
        return False
    
    s1, s2 = steps[0], steps[1]
    s1 = s1 % n
    s2 = s2 % n
    
    # Precompute step offsets for each direction
    # Directions: 0:+s1, 1:-s1, 2:+s2, 3:-s2
    step_offsets = np.array([s1, -s1, s2, -s2], dtype=np.int64)
    
    max_depth = g - 1  # We need to check depths up to g-1 for cycles of length < g
    
    # visited[pos][dir] stores the depth at which we first reached this state
    visited = np.full((n, 4), -1, dtype=np.int32)
    
    # Initialize BFS: from vertex 0, take each of the 4 directions
    for d in range(4):
        new_pos = step_offsets[d] % n
        if visited[new_pos, d] == -1:
            visited[new_pos, d] = 1
    
    # Current frontier: list of (pos, prev_dir) at current depth
    frontier = [(step_offsets[d] % n, d) for d in range(4)]
    
    for depth in range(1, max_depth):
        if not frontier:
            break
        
        new_depth = depth + 1
        # Build next frontier
        next_frontier = []
        
        # Vectorize: for each state in frontier, generate 3 valid next states (excluding backtrack)
        # To keep it simple and fast, we process in batches
        for pos, prev_dir in frontier:
            # Valid next directions: all except the backtrack of prev_dir
            if prev_dir == 0:
                valid_dirs = [2, 3]  # exclude 1 (backtrack of 0)
                # Actually we also exclude 0? No, we can go same direction
                # Wait: non-backtracking means we can't immediately reverse.
                # From direction 0 (+s1), we can go 0 (+s1), 2 (+s2), 3 (-s2)
                # We cannot go 1 (-s1)
                valid_dirs = [0, 2, 3]
            elif prev_dir == 1:
                valid_dirs = [0, 2, 3]  # cannot go 0? No, cannot go 0 is wrong.
                # From direction 1 (-s1), we can go 1 (-s1), 2 (+s2), 3 (-s2)
                # We cannot go 0 (+s1)
                valid_dirs = [1, 2, 3]
            elif prev_dir == 2:
                valid_dirs = [0, 1, 2]  # cannot go 3
            else:  # prev_dir == 3
                valid_dirs = [0, 1, 3]  # cannot go 2
            
            for nd in valid_dirs:
                new_pos = (pos + step_offsets[nd]) % n
                
                if new_pos == 0:
                    # Found a cycle of length new_depth
                    if new_depth < g:
                        return False
                    continue
                
                if visited[new_pos, nd] == -1:
                    visited[new_pos, nd] = new_depth
                    next_frontier.append((new_pos, nd))
        
        frontier = next_frontier
    
    return True

def _search_circulants(k, g, n_target, deadline):
    """Search for valid circulant graphs with n_target vertices."""
    n = n_target
    s1 = 1
    candidates = []
    for s2 in range(2, n // 2 + 1):
        if s2 == n - s2:
            continue
        candidates.append((s1, s2))
    
    random.shuffle(candidates)
    
    max_tries = 500  # Increased tries with faster check
    for i, (a, b) in enumerate(candidates[:max_tries]):
        if time.time() > deadline:
            break
        if _check_circulant_girth_fast(n, [a, b], k, g):
            return n, _get_circulant_edges(n, [a, b])
    
    return None

def _get_circulant_edges(n, steps):
    edges = []
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular girth-g graphs.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
