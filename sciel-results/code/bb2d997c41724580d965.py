import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Try to find a circulant graph C_320({s1, s2}) with girth >= 10
        # This is a long shot as simple circulants of size 320 typically don't have girth 10,
        # but it's fast to check a subset or use the existing search.
        n_target = 320
        result = _search_circulant_optimized(n_target, k, g, deadline)
        if result is not None:
            return result
        
        # Fallback to LUW construction with minimal d for girth >= g
        # For LUW, girth >= 2d + 2.
        # We need 2d + 2 >= g => 2d >= g - 2 => d >= (g - 2) / 2.
        # For g = 10, d >= 4.
        # Seed used d = 5 (n=2048). We use d = 4 (n=1024).
        return _luw_construct_optimized(k, g)

    # General fallback
    return _luw_construct_optimized(k, g)

def _search_circulant_optimized(n, k, g, deadline):
    """
    Systematically search for a circulant graph C_n({s1, s2}) with girth >= g.
    """
    max_s = n // 2
    found = False
    
    # Iterate over possible s1
    for s1 in range(1, max_s + 1):
        if 2 * s1 % n == 0:
            continue
        for s2 in range(s1 + 1, max_s + 1):
            if 2 * s2 % n == 0:
                continue
            if s1 + s2 == n: # s1 = -s2 mod n
                continue
            
            # Check girth
            if _check_girth_circulant(n, s1, s2, g):
                # Found a valid circulant
                edges = _get_circulant_edges(n, [s1, s2])
                return n, edges
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return None

def _check_girth_circulant(n, s1, s2, g):
    """
    Check if C_n({s1, s2}) has girth >= g.
    Uses BFS on non-backtracking walks.
    """
    visited = [[-1] * 4 for _ in range(n)]
    
    queue = []
    for d in range(4):
        if d == 0:
            np_ = s1 % n
        elif d == 1:
            np_ = (-s1) % n
        elif d == 2:
            np_ = s2 % n
        else:
            np_ = (-s2) % n
            
        visited[np_][d] = 1
        queue.append((np_, d, 1))
        
    qi = 0
    max_depth = g - 1
    
    while qi < len(queue):
        pos, prev_dir, depth = queue[qi]
        qi += 1
        
        if depth >= max_depth:
            continue
            
        for next_dir in range(4):
            if (prev_dir == 0 and next_dir == 1) or (prev_dir == 1 and next_dir == 0) or \
               (prev_dir == 2 and next_dir == 3) or (prev_dir == 3 and next_dir == 2):
                continue
                
            if next_dir == 0:
                np_ = (pos + s1) % n
            elif next_dir == 1:
                np_ = (pos - s1) % n
            elif next_dir == 2:
                np_ = (pos + s2) % n
            else:
                np_ = (pos - s2) % n
                
            new_depth = depth + 1
            
            if np_ == 0:
                if new_depth < g:
                    return False
                else:
                    continue
                    
            if visited[np_][next_dir] != -1:
                if visited[np_][next_dir] <= new_depth:
                    continue
            else:
                visited[np_][next_dir] = new_depth
                queue.append((np_, next_dir, new_depth))
                
    return True

def _get_circulant_edges(n, steps):
    edges = []
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges

def _luw_construct_optimized(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular girth-g graphs.
    Optimized to use the minimal d such that girth >= g.
    Girth of LUW(q, d) is at least 2d + 2.
    """
    q = k
    
    # Determine minimal d
    # 2d + 2 >= g
    # d >= (g - 2) / 2
    # Since d must be integer, d = ceil((g - 2) / 2)
    # For g=10, d = ceil(8/2) = 4.
    # For g=8, d = ceil(6/2) = 3.
    d = max(3, (g - 2 + 1) // 2) # Ceil division for positive ints
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for the LUW construction
    # The rule depends on d.
    # For d=4, we need rules for c=2, 3, 4.
    # The standard LUW rules for q=4 are fixed.
    
    # The rule array is indexed by c (column index in the matrix representation).
    # rule[c] = (a, b) such that y_c = x_c +/- y_a * x_b
    # For q=4, the rules are:
    # c=2: (1, 1)
    # c=3: (2, 1)
    # c=4: (1, 2)
    # c=5: (1, 3) ... etc.
    
    rule = {}
    # Pre-compute rules up to d
    for c in range(2, d + 1):
        if c == 2:
            rule[c] = (1, 1)
        elif c == 3:
            rule[c] = (2, 1)
        elif c == 4:
            rule[c] = (1, 2)
        else:
            # General pattern for larger c
            # The sequence of rules for q=4 is periodic or follows a pattern.
            # For d=4, we only need up to 4.
            # If d > 4, we need the general formula.
            # The seed code had:
            # r, c0 = (c - 5) % 4, c - (c - 5) % 4
            # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            # Let's use this general formula to be safe for any g.
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
