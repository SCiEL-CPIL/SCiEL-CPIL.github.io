import random
import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin

    if k != 5:
        return _luw_construct(k, g, deadline)

    # Try SL(2,11) Cayley graph construction first
    # SL(2,11) has 1320 elements; with 5 involutions as generators,
    # we get a 5-regular Cayley graph. Verify girth >= 9.
    try:
        sl211_result = _sl211_cayley_graph(deadline)
        if sl211_result is not None:
            n_sl, edges_sl = sl211_result
            # Verify girth >= 9 using BFS from every vertex
            if _verify_girth(n_sl, edges_sl, 9, deadline):
                return sl211_result
    except Exception:
        pass

    # Fall back to LUW construction
    return _luw_construct(k, g, deadline)


def _sl211_cayley_graph(deadline):
    """Construct a 5-regular Cayley graph on SL(2,11) using 5 involutions."""
    # SL(2,11): 2x2 matrices with determinant 1 over GF(11)
    # |SL(2,11)| = 11 * (11^2 - 1) = 11 * 120 = 1320
    
    p = 11
    
    # Generate all elements of SL(2,11)
    # Element is (a, b, c, d) representing matrix [[a,b],[c,d]] with ad-bc=1
    elements = []
    elem_to_idx = {}
    
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # Need ad - bc = 1, so ad = 1 + bc
                # If a != 0: d = (1 + bc) * a^(-1)
                # If a == 0: -bc = 1, so bc = -1 = 10 mod 11
                if a != 0:
                    a_inv = pow(a, p - 2, p)  # Fermat's little theorem
                    d = (1 + b * c) % p * a_inv % p
                    # Check determinant
                    if (a * d - b * c) % p == 1:
                        elem = (a, b, c, d)
                        elements.append(elem)
                        elem_to_idx[elem] = len(elements) - 1
                else:
                    # a = 0, need -bc = 1, so bc = 10
                    if b == 0:
                        continue  # bc = 0 != 10
                    # c = 10 * b^(-1)
                    b_inv = pow(b, p - 2, p)
                    c_val = 10 * b_inv % p
                    d = 0
                    elem = (0, b, c_val, d)
                    if (a * d - b * c) % p == 1:
                        if elem not in elem_to_idx:
                            elements.append(elem)
                            elem_to_idx[elem] = len(elements) - 1
    
    n = len(elements)
    if n != 1320:
        return None
    
    # Define 5 involutions (elements of order 2) in SL(2,11)
    # An involution satisfies M^2 = I and M != I
    # Trace = 0 for involutions in SL(2,p) (since eigenvalues are ±1, trace = 0)
    # Actually for order 2: M^2 = I, so (M-I)(M+I) = 0
    # In SL(2,p), involutions have trace 0 (if p is odd and M != I)
    
    # Let's find 5 involutions that generate SL(2,11)
    # Common involutions: matrices with trace 0 and determinant 1
    
    # Try specific involutions
    # S = [[0, -1], [1, 0]] = [[0, 10], [1, 0]] in GF(11)
    # T = [[1, 1], [0, 1]] is not an involution (order 11)
    # We need elements of order exactly 2
    
    # Involutions in SL(2,11): trace = 0, det = 1
    # [[a, b], [c, -a]] with -a^2 - bc = 1, i.e., a^2 + bc = -1 = 10
    
    involutions = []
    # S = [[0, 10], [1, 0]]: trace=0, det = 0*0 - 10*1 = -10 = 1 mod 11. Good!
    involutions.append((0, 10, 1, 0))
    
    # [[1, 0], [0, 10]]: trace = 11 = 0, det = 10. Not in SL(2,11).
    # [[a, b], [c, -a]] with a^2 + bc = 10
    
    # Try a=1: 1 + bc = 10, bc = 9
    # b=1, c=9: [[1, 1], [9, 10]]
    involutions.append((1, 1, 9, 10))
    
    # a=2: 4 + bc = 10, bc = 6
    # b=1, c=6: [[2, 1], [6, 9]]
    involutions.append((2, 1, 6, 9))
    
    # a=3: 9 + bc = 10, bc = 1
    # b=1, c=1: [[3, 1], [1, 8]]
    involutions.append((3, 1, 1, 8))
    
    # a=4: 16 + bc = 10, bc = 10-16 = -6 = 5
    # b=1, c=5: [[4, 1], [5, 7]]
    involutions.append((4, 1, 5, 7))
    
    # Verify each is an involution: M^2 = I
    def mat_mult(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p,
                (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
    
    for inv in involutions:
        sq = mat_mult(inv, inv)
        if sq != (1, 0, 0, 1):
            return None  # Not an involution
    
    # Build the Cayley graph
    # For each element g, neighbors are g * s_i for each involution s_i
    # Since s_i are involutions, the graph is undirected
    
    edges = []
    seen_edges = set()
    
    for i, g in enumerate(elements):
        if time.time() > deadline - 10.0:
            break
        for s in involutions:
            h = mat_mult(g, s)
            if h in elem_to_idx:
                j = elem_to_idx[h]
                if i < j:
                    edge_key = (i, j)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append(edge_key)
    
    return n, edges


def _verify_girth(n, edges, min_girth, deadline):
    """Verify that the graph has no cycle shorter than min_girth using BFS."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 5:
            return False
    
    # Check for self-loops and duplicate edges
    edge_set = set()
    for i, j in edges:
        if i == j:
            return False
        key = (min(i,j), max(i,j))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # BFS from each vertex to find shortest cycle
    for start in range(n):
        if time.time() > deadline - 5.0:
            return False  # Timeout
        
        dist = [-1] * n
        parent = [-1] * n
        queue = deque()
        dist[start] = 0
        queue.append(start)
        
        min_cycle = float('inf')
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle < min_girth:
                            return False
        
        if min_cycle < min_girth:
            return False
    
    return True


def _luw_construct(k, g, deadline=None):
    """Vectorized L-U-W construction using NumPy for speed."""
    # This is the original LUW construction, kept as fallback
    if deadline is None:
        deadline = time.time() + 110.0
    
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)

    if q == 4:
        mul_table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int64)
    else:
        mul_table = np.outer(np.arange(q, dtype=np.int64), np.arange(q, dtype=np.int64)) % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    rule_a = np.array([rule[c][0] for c in range(2, d + 1)], dtype=np.int64)
    rule_b = np.array([rule[c][1] for c in range(2, d + 1)], dtype=np.int64)

    free_vals = np.arange(q, dtype=np.int64)

    def get_neighbours(side, x_arr):
        y_coords = np.empty((q, d + 1), dtype=np.int64)
        y_coords[:, 0] = 0
        y_coords[:, 1] = free_vals
        if side == 0:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                prod = mul_table[y_coords[:, a], x_arr[b]]
                y_coords[:, c] = (x_arr[c] + prod) % q
        else:
            for idx, c in enumerate(range(2, d + 1)):
                a = rule_a[idx]
                b = rule_b[idx]
                prod = mul_table[x_arr[a], y_coords[:, b]]
                y_coords[:, c] = (x_arr[c] - prod) % q
        return y_coords

    start_key = (0, (0,) * (d + 1))
    index = {start_key: 0}
    order = [start_key]
    edges = []

    for i, vertex in enumerate(order):
        if time.time() > deadline - 5.0:
            break
        side, x_tuple = vertex
        x_arr = np.array(x_tuple, dtype=np.int64)
        y_coords = get_neighbours(side, x_arr)
        for j in range(q):
            other_key = (1 - side, tuple(y_coords[j]))
            if other_key not in index:
                index[other_key] = len(order)
                order.append(other_key)
            if side == 0:
                edges.append((i, index[other_key]))

    return len(order), edges
