import time
import numpy as np
from scipy.sparse import csr_matrix
from collections import deque

_T0 = time.time()
_DEADLINE = _T0 + 110.0  # 10 s safety margin before the 120 s evaluator kill

def _deadline():
    return time.time() >= _DEADLINE

def _field_ops(q):
    """Return (add, sub, mul) over GF(q) for prime q, or GF(4) when q == 4."""
    if q == 4:
        table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])
        return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x, y])
    return ((lambda x, y: (x + y) % q),
            (lambda x, y: (x - y) % q),
            (lambda x, y: (x * y) % q))

def _luw_rule(d):
    """LUW (d, q) rule: rule[c] = (a, b) for c = 2..d.  p ~ l iff l[c]-p[c] = l[a]*p[b]."""
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule

def _luw_graph(d, q):
    """Build the connected LUW component from the zero point.  Returns (n, edges, adj)."""
    add, sub, mul = _field_ops(q)
    rule = _luw_rule(d)
    rc = [None] * (d + 1)
    for c in range(2, d + 1):
        rc[c] = rule[c]

    # We represent vertices as tuples of length d+1.
    # Side 0: Points. Side 1: Lines.
    # Vertex is (side, coords)
    
    # To speed up, we store coords as numpy arrays in a list, but use tuples for dict keys.
    # This is a trade-off. For q=5, d=4, size is 5. Tuples are fine.
    
    index = {}
    order = []
    adj = []
    
    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)
    adj.append([])
    
    # BFS
    # We will process level by level to allow vectorization if possible, 
    # but for q=5, simple loop might be faster due to overhead of numpy for small batches.
    # However, for larger q or d, vectorization helps.
    # Let's stick to a hybrid: try to vectorize the neighbor generation for the current frontier.
    
    current_level = [0]
    
    while current_level:
        if _deadline():
            break
        next_level = []
        
        # Separate current level into points and lines
        points = []
        lines = []
        for idx in current_level:
            side, coords = order[idx]
            if side == 0:
                points.append(idx)
            else:
                lines.append(idx)
        
        # Process Points -> Lines
        if points:
            # Vectorized neighbor generation for points
            # P: list of indices
            # For each point p, generate q lines.
            # We can do this in a loop, but use numpy for the calculation.
            for p_idx in points:
                p_side, p_coords = order[p_idx]
                # p_coords is a tuple
                p_arr = np.array(p_coords, dtype=int)
                
                # Generate q lines
                for first in range(q):
                    l_arr = np.zeros(d+1, dtype=int)
                    l_arr[0] = first # Placeholder, will be ignored in key? 
                    # Wait, the key is the tuple. 
                    # The LUW construction defines lines by their coords.
                    # The "first" coord is part of the line definition.
                    
                    # y[c] = p[c] + y[a] * p[b]
                    # Here y is the line.
                    # y[1] = first.
                    l_arr[1] = first
                    for c in range(2, d + 1):
                        a, b = rc[c]
                        # l[c] = p[c] + l[a] * p[b]
                        # Note: l[a] is known if a < c? 
                        # In LUW, a < c always? 
                        # rule[c] = (a,b). 
                        # For c=2, a=1. l[1] is known.
                        # For c=3, a=2. l[2] is known.
                        # So yes, we can compute sequentially.
                        l_arr[c] = add(p_arr[c], mul(l_arr[a], p_arr[b]))
                    
                    l_key = (1, tuple(l_arr))
                    if l_key not in index:
                        index[l_key] = len(order)
                        order.append(l_key)
                        adj.append([])
                        next_level.append(index[l_key])
                    
                    l_idx = index[l_key]
                    adj[p_idx].append(l_idx)
                    adj[l_idx].append(p_idx)
        
        # Process Lines -> Points
        if lines:
            for l_idx in lines:
                l_side, l_coords = order[l_idx]
                l_arr = np.array(l_coords, dtype=int)
                
                for first in range(q):
                    p_arr = np.zeros(d+1, dtype=int)
                    p_arr[0] = first
                    p_arr[1] = first # Wait, point coords?
                    # Point p ~ Line l iff l[c] - p[c] = l[a] p[b]
                    # p[c] = l[c] - l[a] p[b]
                    # p[1] = first?
                    # In the seed code:
                    # y = [None, first] + ...
                    # For points, y is the point.
                    # So p[1] = first.
                    
                    p_arr[1] = first
                    for c in range(2, d + 1):
                        a, b = rc[c]
                        # p[c] = l[c] - l[a] p[b]
                        p_arr[c] = sub(l_arr[c], mul(l_arr[a], p_arr[b]))
                    
                    p_key = (0, tuple(p_arr))
                    if p_key not in index:
                        index[p_key] = len(order)
                        order.append(p_key)
                        adj.append([])
                        next_level.append(index[p_key])
                    
                    p_idx = index[p_key]
                    adj[l_idx].append(p_idx)
                    adj[p_idx].append(l_idx)
        
        current_level = next_level

    n = len(order)
    edges = []
    seen = set()
    for i in range(n):
        for j in adj[i]:
            if i < j:
                if (i, j) not in seen:
                    seen.add((i, j))
                    edges.append((i, j))
    
    return n, edges, adj

def _verify_girth(adj, g):
    """Exact certificate: no cycle shorter than g. Uses scipy.sparse for matrix powers."""
    n = len(adj)
    if n == 0:
        return True
    
    # Build adjacency matrix
    rows = []
    cols = []
    for u in range(n):
        for v in adj[u]:
            rows.append(u)
            cols.append(v)
    
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    
    # Check for cycles of length 2k < g
    # Since graph is bipartite, girth is even.
    # We need to check 4, 6, 8, ... up to g-1 (if g is odd) or g-2 (if g is even? No, girth >= g).
    # If g=9, we check 4, 6, 8.
    # If g=10, we check 4, 6, 8.
    
    max_len = g - 1
    if max_len % 2 != 0:
        max_len -= 1
        
    A_pow = A.copy()
    for length in range(4, max_len + 1, 2):
        # A_pow is A^(length-2)
        A_pow = A_pow @ A
        # Check diagonal
        diag = A_pow.diagonal()
        if np.any(diag > 0):
            return False
    return True

def _check_simple_and_regular(n, edges, k):
    seen = set()
    deg = [0] * n
    for (i, j) in edges:
        if i < 0 or i >= n or j < 0 or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    return all(x == k for x in deg)

def construct(k, g):
    """Return (n, edges) for a k-regular graph of girth >= g on as few vertices as possible."""
    candidates = []
    if k == 4:
        qs = [4]
    else:
        qs = [k]
    for q in qs:
        d0 = g - 5
        if d0 % 2 == 0:
            d0 += 1
        d0 = max(3, d0)
        for d in (d0, d0 + 2, d0 + 4):
            candidates.append((d, q))
            
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)

    best = None
    best_key = None
    for (d, q) in ordered:
        if _deadline():
            break
        try:
            n, edges, adj = _luw_graph(d, q)
        except Exception:
            continue
        if not _check_simple_and_regular(n, edges, k):
            continue
        if _deadline():
            break
        if _verify_girth(adj, g):
            # Valid graph
            key = n
            if best_key is None or key < best_key:
                best_key = key
                best = (n, edges)
            if _deadline():
                break

    if best is None:
        # Fallback
        for (d, q) in ordered:
            try:
                n, edges, adj = _luw_graph(d, q)
                if _check_simple_and_regular(n, edges, k):
                    return n, edges
            except Exception:
                continue
        return 0, []

    return best
