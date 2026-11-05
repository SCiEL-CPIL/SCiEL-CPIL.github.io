import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s kill

    if k == 4 and g == 10:
        # Try known circulant candidates first (fast check)
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
        
        # Fallback: corrected LUW construction with d=6 for girth >= 10
        return _luw_construct(k, g)

    # General fallback: LUW construction
    return _luw_construct(k, g)

def _check_circulant_girth_fast(n, steps, k, g):
    """
    Fast girth check for circulant graph C_n(steps) using vectorized non-backtracking BFS.
    """
    if len(steps) * 2 != k:
        return False
    
    s1, s2 = steps[0], steps[1]
    
    # Directions: 0=+s1, 1=-s1, 2=+s2, 3=-s2
    # Opposite (backtracking) pairs: 0<->1, 2<->3
    opposite = np.array([1, 0, 3, 2], dtype=np.int32)
    # Step deltas for each direction
    deltas = np.array([s1, -s1, s2, -s2], dtype=np.int64)
    
    max_depth = g - 1  # check cycles of length < g
    
    # visited[pos, dir] = depth at which first reached, -1 if unvisited
    visited = np.full((n, 4), -1, dtype=np.int32)
    
    # Current frontier: set of (pos, dir) at current depth
    # We'll track as a boolean mask frontier[pos, dir] = True if in current level
    
    # Initialize depth 1
    frontier = np.zeros((n, 4), dtype=bool)
    pos0 = 0
    for d in range(4):
        p = (pos0 + deltas[d]) % n
        frontier[p, d] = True
        visited[p, d] = 1
    
    # BFS level by level
    for depth in range(1, max_depth):
        if not frontier.any():
            break
        
        next_frontier = np.zeros((n, 4), dtype=bool)
        
        # For each direction, compute where we go from current frontier positions
        # But we must exclude backtracking (previous dir is opposite of next dir)
        # A state (pos, prev_dir) in frontier can go to next_dir if next_dir != opposite[prev_dir]
        
        # For each next_dir nd:
        #   Positions reachable: pos + deltas[nd] for all (pos, pd) in frontier where pd != opposite[nd]
        #   i.e., exclude pd == opposite[nd]
        
        for nd in range(4):
            # All positions in frontier that are NOT in direction opposite[nd]
            valid_prev = np.ones(4, dtype=bool)
            valid_prev[opposite[nd]] = False
            # Get positions from frontier where prev_dir is valid
            # frontier is (n,4), we want positions where any valid_prev dir is True
            # For each valid prev dir pd, positions are frontier[:, pd]
            # Combine: pos is reachable if any valid pd has frontier[pos, pd] = True
            pos_mask = np.zeros(n, dtype=bool)
            for pd in range(4):
                if valid_prev[pd]:
                    pos_mask |= frontier[:, pd]
            
            if not pos_mask.any():
                continue
            
            # New positions
            new_pos = (np.where(pos_mask)[0] + deltas[nd]) % n
            
            # Check for cycle: new_pos == 0 means we returned to start
            # Cycle length = depth + 1
            if depth + 1 < g:
                if 0 in new_pos:
                    return False
            
            # Filter out position 0 (don't need to enqueue)
            non_zero = new_pos != 0
            new_pos = new_pos[non_zero]
            
            # Check visited
            for p in new_pos:
                if visited[p, nd] == -1:
                    visited[p, nd] = depth + 1
                    next_frontier[p, nd] = True
        
        frontier = next_frontier
    
    return True

def _search_circulants(k, g, n_target, deadline):
    """Search for valid circulant graphs with n_target vertices."""
    n = n_target
    
    # Try s1 = 1 (most common in known circulants)
    # s2 ranges over 2..n//2, avoiding duplicates
    s1 = 1
    candidates = []
    for s2 in range(2, n // 2 + 1):
        if s2 == n - s2:  # s2 == n/2, self-inverse, skip
            continue
        candidates.append((s1, s2))
    
    # Also try other small s1 values for broader coverage
    for s1 in [2, 3, 4, 5]:
        for s2 in range(s1 + 1, n // 2 + 1):
            if s2 == n - s2:
                continue
            # Avoid s1 + s2 == n (would create duplicate edge)
            if (s1 + s2) % n == 0 or (s1 + s2) % n == n - s2 or (s1 + s2) % n == n - s1:
                continue
            # Avoid s1 - s2 == 0 mod n or s1 - s2 == n - something that creates dup
            if (s1 - s2) % n == 0:
                continue
            candidates.append((s1, s2))
    
    # Shuffle to try diverse candidates
    random.shuffle(candidates)
    
    # Try as many as possible within time budget
    for a, b in candidates:
        if time.time() > deadline - 2.0:
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
    For k=4, g=10: use d=6 (girth >= d+4 = 10)
    """
    q = k
    # For girth g, we need d such that d + 4 >= g (for even d) or d + 5 >= g (for odd d)
    # Actually the LUW graph CD(d, q) has girth at least d + 5 for odd d, d + 4 for even d
    # For g=10: d=6 gives girth >= 10, d=5 gives girth >= 10 (5+5=10)
    # Let's use d = g - 5 if (g-5) is odd, else g - 4
    # g=10: g-5=5 (odd), so d=5 -> girth >= 5+5=10
    # Wait, let me re-check: the standard result is girth >= d+5 for odd d, d+4 for even d
    # So for g=10: d=5 (odd) -> girth >= 10. Good.
    # But the original code had d = max(3, g-5 if (g-5)%2 else g-4)
    # For g=10: g-5=5, 5%2=1 (odd), so d=5. That gives girth >= 5+5=10.
    
    # Actually, I need to double-check the LUW girth formula.
    # The LUW graph has girth at least d + 5 for odd d.
    # So d=5 -> girth >= 10. 
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
    
    # Build the rule
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
