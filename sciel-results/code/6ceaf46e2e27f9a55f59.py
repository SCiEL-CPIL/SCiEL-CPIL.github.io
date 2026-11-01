import time
import random
import sys
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k != 5 or g != 9:
        return _luw_construct(k, g, deadline)

    # Try Cayley graph of SL(2, 11)
    # Order 1320
    # If we find a valid generating set, return it.
    result = _try_sl2_11(k, g, deadline)
    if result is not None:
        return result

    # Fallback to LUW
    return _luw_construct(k, g, deadline)

def _try_sl2_11(k, g, deadline):
    # Construct SL(2, 11)
    # Elements are 2x2 matrices over Z_11 with det 1
    p = 11
    elements = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                # d = (1 - bc) * a^{-1} if a != 0, else handle separately
                # Actually, iterate all a,b,c,d and check det
                # 11^4 = 14641, small enough to filter
                pass
    
    # Better enumeration
    elems = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    det = (a*d - b*c) % p
                    if det == 1:
                        elems.append(((a, b), (c, d)))
    
    # Map elements to indices
    elem_to_idx = {e: i for i, e in enumerate(elems)}
    n = len(elems)
    
    # Define multiplication
    def mat_mul(m1, m2):
        a1, b1 = m1[0]
        c1, d1 = m1[1]
        a2, b2 = m2[0]
        c2, d2 = m2[1]
        return (
            ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p),
            ((c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
        )
    
    # Precompute multiplication table? 1320^2 = 1.7M entries.
    # Each entry is an index.
    # Memory: 1.7M * 4 bytes ~ 7MB. OK.
    # Time: 1.7M multiplications. Fast.
    mult_table = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            prod = mat_mul(elems[i], elems[j])
            mult_table[i][j] = elem_to_idx[prod]
    
    # Find identity
    identity = ((1, 0), (0, 1))
    id_idx = elem_to_idx[identity]
    
    # Find inverses
    inv_idx = [0]*n
    for i in range(n):
        # Inverse of [[a,b],[c,d]] is [[d,-b],[-c,a]]
        a, b = elems[i][0]
        c, d = elems[i][1]
        inv_mat = ((d, (-b)%p), ((-c)%p, a))
        inv_idx[i] = elem_to_idx[inv_mat]
    
    # Candidate Generating Sets
    # We need 5 elements.
    # Standard generators U, V
    U = ((1, 1), (0, 1))
    V = ((1, 0), (1, 1))
    W = ((0, 1), (10, 0)) # -1 = 10
    neg_I = ((10, 0), (0, 10))
    
    u_idx = elem_to_idx[U]
    v_idx = elem_to_idx[V]
    w_idx = elem_to_idx[W]
    neg_i_idx = elem_to_idx[neg_I]
    
    # Set 1: {U, U^-1, V, V^-1, -I}
    # Check if -I is an involution (yes, (-I)^2 = I)
    # Check if U, V are not involutions (Order 11)
    
    candidates = []
    
    # Set 1
    s1 = [u_idx, inv_idx[u_idx], v_idx, inv_idx[v_idx], neg_i_idx]
    # Ensure no duplicates and no identity
    if len(set(s1)) == 5 and id_idx not in s1:
        candidates.append(s1)
        
    # Try other involutions
    # Find all involutions
    involutions = []
    for i in range(n):
        if i == id_idx: continue
        if mult_table[i][i] == id_idx:
            involutions.append(i)
            
    # Try a few random involutions combined with U, V
    random.shuffle(involutions)
    for inv in involutions[:20]:
        s = [u_idx, inv_idx[u_idx], v_idx, inv_idx[v_idx], inv]
        if len(set(s)) == 5 and id_idx not in s:
            candidates.append(s)
            
    # Also try U, U^-1, W, W^-1, -I?
    # W has order 4. W^-1 = W^3 = -W.
    # Set: {U, U^-1, W, W^-1, -I}
    w_inv_idx = inv_idx[w_idx]
    s2 = [u_idx, inv_idx[u_idx], w_idx, w_inv_idx, neg_i_idx]
    if len(set(s2)) == 5 and id_idx not in s2:
        candidates.append(s2)
        
    # Check each candidate
    for S in candidates:
        if time.time() > deadline - 10:
            break
            
        # Build adjacency
        adj = [[] for _ in range(n)]
        for i in range(n):
            for s in S:
                j = mult_table[i][s]
                adj[i].append(j)
                
        # Check girth
        if _check_girth_cayley(adj, n, g):
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append((i, j))
            return n, edges
            
    return None

def _check_girth_cayley(adj, n, min_girth):
    # Check for cycles of length < min_girth
    # Using BFS on state space (current, previous)
    # For each start vertex s
    for s in range(n):
        # Start states: (v, s) for v in adj[s]
        # We want to reach (s, u) for u in adj[s], u != v
        # Distance in state space corresponds to walk length - 1
        
        # BFS
        visited = {} # (curr, prev) -> dist
        queue = []
        for v in adj[s]:
            queue.append((v, s, 1)) # curr=v, prev=s, dist=1 (step 1)
            visited[(v, s)] = 1
            
        while queue:
            curr, prev, dist = queue.pop(0)
            # If curr is s, we have a cycle
            # The walk length is dist + 1 (step to s)
            # But we must ensure the last step is not backtracking?
            # The state (s, u) means we are at s, came from u.
            # The walk was ... -> u -> s.
            # The previous step was to u.
            # If the walk started s -> v -> ... -> u -> s.
            # Length is dist + 1.
            # We need Length >= min_girth.
            if curr == s:
                cycle_len = dist + 1
                if cycle_len < min_girth:
                    return False
                # If cycle_len >= min_girth, we can stop expanding from this state?
                # No, there might be shorter cycles through other paths?
                # Actually, BFS finds shortest path.
                # So the first time we hit (s, u) is the shortest cycle through s starting with edge s-u?
                # No, we started with s-v.
                # So this is the shortest cycle containing edge s-v and returning to s?
                # Yes.
                # If this shortest cycle is >= min_girth, are we safe?
                # We need ALL cycles through s to be >= min_girth.
                # So we must continue BFS to find if there's a SHORTER cycle via a different path?
                # No, BFS explores in order of distance.
                # So if we hit (s, u) at dist D, any other path to (s, u) is longer.
                # But there might be a path to (s, w) with w != u that is shorter?
                # Yes. So we must check ALL (s, w) states.
                # We can mark (s, w) as "visited" and continue?
                # Actually, once we hit a state (s, w), we know the shortest cycle through s via w.
                # We can prune further expansion from (s, w) because any cycle going through s and continuing is longer.
                # So we don't add neighbors of (s, w) to the queue.
                continue
                
            # Expand
            for next_v in adj[curr]:
                if next_v == prev:
                    continue
                new_state = (next_v, curr)
                if new_state not in visited:
                    visited[new_state] = dist + 1
                    queue.append((next_v, curr, dist + 1))
                    
    return True

def _luw_construct(k, g, deadline):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = g - 5
    if d % 2 == 0:
        d += 1
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def get_neighbours(vertex):
        side, x = vertex
        neighbors = []
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbors.append((1 - side, tuple(y)))
        return neighbors
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    i = 0
    while i < len(order):
        if time.time() > deadline:
            n = len(order)
            valid_edges = [(u, v) for u, v in edges if u < n and v < n]
            return n, valid_edges
        vertex = order[i]
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
        i += 1
    
    return len(order), edges
