# EVOLVE-BLOCK-START
import time
import random
import math
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Combines LUW construction with Cayley graph search on SL(2, q).
    """
    start_time = time.time()
    deadline = start_time + 110

    if k != 3:
        return _lnw_construct(k, g)

    # 1. Try Cayley Graph Construction on SL(2, q) for small q
    # We look for a 3-regular Cayley graph with girth >= g
    # n = |SL(2, q)| = q(q^2 - 1)
    # For q=13, n=2184. For q=17, n=4896.
    
    best_n = float('inf')
    best_edges = None
    
    # Try q=13 first (smaller n)
    for q in [13, 17]:
        if time.time() > deadline - 20:
            break
            
        n_cayley = q * (q*q - 1)
        
        # If this graph is already larger than the best LUW we know (8748), skip?
        # LUW(3,7) is 8748.
        if n_cayley > 8748:
            continue
            
        # Search for 3 involutions in SL(2, q) that generate the group
        # and have girth >= g
        
        edges = _find_cayley_graph_3reg(q, g, deadline)
        if edges is not None:
            # Verify girth and regularity
            adj = _edges_to_adj(edges, n_cayley)
            girth = _calculate_girth(adj, n_cayley)
            if girth >= g:
                if n_cayley < best_n:
                    best_n = n_cayley
                    best_edges = edges
                # If we found a very small one, we can break
                if n_cayley <= 2184: # Smallest possible
                    break

    if best_edges is not None:
        return best_n, best_edges

    # 2. Fallback to LUW Construction
    # Iterate d from 6 upwards
    for d in range(6, 16):
        if time.time() > deadline - 10:
            break
            
        try:
            n, edges = _lnw_construct_params(3, d)
        except Exception:
            continue
            
        if n >= best_n:
            continue
            
        # Check girth
        adj = _edges_to_adj(edges, n)
        girth = _calculate_girth(adj, n)
        
        if girth >= g:
            best_n = n
            best_edges = edges
            return n, edges

    if best_edges is not None:
        return best_n, best_edges
        
    # Final fallback
    return _lnw_construct(3, g)

def _find_cayley_graph_3reg(q, g, deadline):
    """
    Search for 3 involutions in SL(2, q) that form a generating set
    with girth >= g.
    """
    # Generate the group SL(2, q)
    # Elements are (a, b, c, d) with ad - bc = 1
    group = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                # ad - bc = 1 => d = (1 + bc) * a^-1
                # if a == 0, then -bc = 1 => bc = -1
                if a != 0:
                    # a_inv = pow(a, q-2, q)
                    a_inv = pow(a, q-2, q)
                    d = (1 + b*c) * a_inv % q
                    group.append((a, b, c, d))
                else:
                    # a=0, -bc=1 => bc = q-1
                    if b != 0:
                        c = (q-1) * pow(b, q-2, q) % q
                        d = 0 # ad - bc = 0 - b*c = -1 != 1? 
                        # Wait. If a=0, ad-bc = -bc. We need -bc=1 => bc=q-1.
                        # d can be anything? No, the matrix is [[0,b],[c,d]].
                        # Det = -bc. We need -bc=1.
                        # So d is free? No, the matrix must be in SL(2,q).
                        # The set of matrices with det 1 is a group.
                        # If a=0, then d is determined by the condition?
                        # No, a=0, d is free?
                        # Let's re-evaluate.
                        # Matrix [[a,b],[c,d]]. Det = ad-bc.
                        # If a=0, Det = -bc. We need -bc=1.
                        # d is not constrained by the determinant equation alone?
                        # But the matrix must be invertible.
                        # Actually, for any a,b,c, there is a unique d?
                        # No, if a=0, d can be anything?
                        # No, the group SL(2,q) has size q(q^2-1).
                        # Number of triples (a,b,c) is q^3.
                        # For each (a,b,c), is there a unique d?
                        # ad - bc = 1.
                        # If a != 0, d = (1+bc)/a. Unique.
                        # If a == 0, -bc = 1. If this holds, d is free?
                        # No, the matrix [[0,b],[c,d]] with det 1.
                        # The number of such matrices is q-1 (choices for b!=0, c determined, d free? No).
                        # Actually, the map (a,b,c) -> d is not a function if a=0.
                        # Let's just iterate all a,b,c,d and check det.
                        # q^4 iterations. For q=13, 13^4 = 28561. Fast.
                        pass
    
    # Simpler: Iterate all 4x4 matrices? No, 2x2.
    group = []
    for a in range(q):
        for b in range(q):
            for c in range(q):
                for d in range(q):
                    if (a*d - b*c) % q == 1:
                        group.append((a, b, c, d))
    
    n = len(group)
    # Map group elements to indices
    index = {g: i for i, g in enumerate(group)}
    
    # Identify involutions: g^2 = I
    identity = (1, 0, 0, 1)
    involutions = []
    for g in group:
        if g == identity:
            continue
        # Compute g^2
        a, b, c, d = g
        a2 = (a*a + b*c) % q
        b2 = (a*b + b*d) % q
        c2 = (c*a + d*c) % q
        d2 = (c*b + d*d) % q
        if (a2, b2, c2, d2) == identity:
            involutions.append(g)
    
    if len(involutions) < 3:
        return None
        
    # Randomly sample triples of involutions
    # Check if they generate the group and have girth >= g
    
    # Pre-compute multiplication table for speed?
    # n is up to 4896. n^2 is 24M. Might be too slow/memory heavy.
    # We can compute on the fly.
    
    def mul(g1, g2):
        a1, b1, c1, d1 = g1
        a2, b2, c2, d2 = g2
        return ((a1*a2 + b1*c2) % q,
                (a1*b2 + b1*d2) % q,
                (c1*a2 + d1*c2) % q,
                (c1*b2 + d1*d2) % q)
        
    def get_neighbours(vertex):
        # Vertex is index
        g_v = group[vertex]
        neigh = []
        for s in generators:
            g_new = mul(g_v, s)
            neigh.append(index[g_new])
        return neigh

    # We need to pick 3 generators.
    # To ensure the graph is 3-regular and simple, the generators must be distinct
    # and not identity. Also, if s1*s2 = s3, we might have issues?
    # No, Cayley graph is simple if s_i != s_j and s_i != s_j^-1.
    # Since they are involutions, s_i^-1 = s_i.
    # So we just need 3 distinct involutions.
    
    random.shuffle(involutions)
    
    # Limit the number of trials
    max_trials = 500
    for _ in range(max_trials):
        if time.time() > deadline - 5:
            break
            
        # Pick 3 random involutions
        # Ensure they are distinct
        s1 = involutions[0]
        s2 = involutions[1]
        s3 = involutions[2]
        
        generators = [s1, s2, s3]
        
        # Check if they generate the group
        # BFS from identity
        visited = {0: 0} # index: dist
        queue = deque([0])
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for s in generators:
                g_u = group[u]
                g_v = mul(g_u, s)
                v = index[g_v]
                if v not in visited:
                    visited[v] = visited[u] + 1
                    queue.append(v)
                    
        if len(visited) < n:
            continue # Not connected
            
        # Check girth
        # BFS from each vertex? Too slow.
        # BFS from identity is enough for Cayley graphs?
        # The girth is the length of the shortest cycle.
        # In a Cayley graph, the girth is the minimum length of a word in S that equals identity.
        # We can find this by BFS from identity, looking for the shortest path back to identity
        # that doesn't use the reverse edge immediately?
        # Actually, standard BFS girth check works.
        # Since it's a Cayley graph, the shortest cycle is the shortest relation.
        
        # We already did BFS to check connectivity.
        # We can modify the BFS to find the shortest cycle.
        
        # Re-do BFS with parent tracking
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        queue = deque([0])
        head = 0
        min_cycle = n + 1
        
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
            for s in generators:
                g_u = group[u]
                g_v = mul(g_u, s)
                v = index[g_v]
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            
        if min_cycle >= g:
            # Build edges
            edges = []
            for u in range(n):
                for s in generators:
                    g_u = group[u]
                    g_v = mul(g_u, s)
                    v = index[g_v]
                    if u < v:
                        edges.append((u, v))
            return edges
            
    return None

def _edges_to_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calculate_girth(adj, n):
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
    return min_cycle if min_cycle <= n else n + 1

def _lnw_construct_params(k, d):
    q = k
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

def _lnw_construct(k, g):
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)
# EVOLVE-BLOCK-END
