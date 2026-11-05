import time
import random
import math
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s

    if k == 4 and g == 10:
        # Search for the smallest 4-regular graph with girth >= 10
        # that is a circulant C_n({a, b}).
        
        # Search for circulants with n < 320
        for n in range(243, 320):
            if time.time() > deadline - 5.0:
                break
            
            # Try C_n({1, s}) first
            for s in range(2, n//2 + 1):
                if math.gcd(s, n) != 1:
                    continue
                if _check_girth_circulant_fast(n, 1, s, g):
                    edges = _get_circulant_edges(n, [1, s])
                    return n, edges
            
            # Try C_n({a, b}) with a > 1 for odd n
            if n % 2 == 1:
                for a in range(2, min(8, n//3 + 1)):
                    if math.gcd(a, n) != 1:
                        continue
                    for b in range(a + 1, min(a + 15, n//2 + 1)):
                        if math.gcd(b, n) != 1:
                            continue
                        if math.gcd(b - a, n) == 1:
                            if _check_girth_circulant_fast(n, a, b, g):
                                edges = _get_circulant_edges(n, [a, b])
                                return n, edges
        
        # Search for the best circulant on n=320
        n_target = 320
        
        # First try the known good one
        if _check_girth_circulant_fast(n_target, 1, 79, g):
            edges = _get_circulant_edges(n_target, [1, 79])
            return n_target, edges
        
        # Systematic search over (a, b) pairs for n=320
        # 320 = 2^6 * 5, so a,b must be odd and not divisible by 5
        candidates = []
        for a in range(1, 161):
            if math.gcd(a, 320) != 1:
                continue
            for b in range(a + 1, 161):
                if math.gcd(b, 320) != 1:
                    continue
                if math.gcd(b - a, 320) == 1:
                    candidates.append((a, b))
        
        # Try a subset of candidates
        if time.time() < deadline - 2.0:
            for a, b in candidates[:300]:
                if time.time() > deadline - 2.0:
                    break
                if _check_girth_circulant_fast(n_target, a, b, g):
                    edges = _get_circulant_edges(n_target, [a, b])
                    return n_target, edges
        
        # Final fallback: LUW construction
        return _luw_construct(k, g)

    # General fallback for other k, g values
    return _luw_construct(k, g)

def _check_girth_circulant_fast(n, s1, s2, g):
    """
    Check if C_n({s1, s2}) has girth >= g.
    Uses BFS on non-backtracking walks.
    """
    steps = [s1, -s1, s2, -s2]
    opposite = [1, 0, 3, 2]
    
    visited = [[-1] * 4 for _ in range(n)]
    queue = []
    
    for d in range(4):
        np_ = steps[d] % n
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
            if next_dir == opposite[prev_dir]:
                continue
                
            np_ = (pos + steps[next_dir]) % n
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

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular girth-g graphs.
    """
    q = k
    d = g // 2
    
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
