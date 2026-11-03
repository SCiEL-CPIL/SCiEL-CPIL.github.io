import time
import random
from collections import deque

_DEADLINE = 115.0  # safety margin vs 120s wall-clock


def _sl2_11():
    """Construct SL(2,11) as a list of (a,b,c,d) tuples with det=1 mod 11."""
    p = 11
    elements = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    det = (a * d - b * c) % p
                    if det == 1:
                        elements.append((a, b, c, d))
    return elements


def _mat_mul(x, y):
    a1, b1, c1, d1 = x
    a2, b2, c2, d2 = y
    p = 11
    return ((a1 * a2 + b1 * c2) % p,
            (a1 * b2 + b1 * d2) % p,
            (c1 * a2 + d1 * c2) % p,
            (c1 * b2 + d1 * d2) % p)


def _construct_sl2_cayley(k, g, deadline):
    """Construct a 5-regular Cayley graph on SL(2,11) with girth >= g."""
    if k != 5 or g != 9:
        return None
    
    t0 = time.time()
    elements = _sl2_11()
    n = len(elements)
    idx = {e: i for i, e in enumerate(elements)}
    ident = (1, 0, 0, 1)
    
    # Find all involutions
    involutions = []
    for e in elements:
        if e == ident:
            continue
        e2 = _mat_mul(e, e)
        if e2 == ident:
            involutions.append(e)
    
    if len(involutions) < 5:
        return None
    
    inv_set = set(involutions)
    
    # Precompute: for each pair of involutions, check if their product is an involution
    # and if product equals another involution
    # inv_prod_inv[i][j] = True if involutions[i] * involutions[j] is an involution
    # inv_prod_eq[i][j] = involutions[k] if involutions[i]*involutions[j] == involutions[k], else -1
    
    m = len(involutions)
    inv_prod_is_inv = [[False] * m for _ in range(m)]
    inv_prod_eq = [[-1] * m for _ in range(m)]
    
    for i in range(m):
        for j in range(m):
            prod = _mat_mul(involutions[i], involutions[j])
            if prod in inv_set:
                inv_prod_is_inv[i][j] = True
                # Find which involution it equals
                for k in range(m):
                    if involutions[k] == prod:
                        inv_prod_eq[i][j] = k
                        break
    
    # We need to find 5 involutions s0..s4 such that:
    # 1. For all distinct i,j: inv_prod_is_inv[i][j] == False (no 4-cycles)
    # 2. For all distinct i,j,k: inv_prod_eq[i][j] != k (no 3-cycles)
    # 3. No non-backtracking word of length 5,6,7,8 equals identity
    
    # Condition 1 and 2 can be checked quickly with precomputed tables.
    # Condition 3 requires checking words.
    
    # Strategy: use backtracking with pruning based on conditions 1 and 2
    # Then verify condition 3 with BFS
    
    def check_short_word_5_8(S_list):
        """Check if there's a non-backtracking word of length 5-8 in S that equals identity."""
        # A non-backtracking word means no consecutive same generators
        # (since generators are involutions, s_i * s_i = 1 is backtracking)
        # We need: s_{i1} * s_{i2} * ... * s_{il} = 1 where consecutive indices differ
        # and the word is not a palindrome that cancels (but since involutions, 
        # s*a*s*b*s*a*s*b... patterns can create cycles)
        
        # For length 5: s1*s2*s3*s4*s5 = 1 with consecutive distinct
        # For length 6: s1*s2*s3*s4*s5*s6 = 1
        # etc.
        
        # This is expensive. Let's use a BFS approach on the Cayley graph
        # but only up to depth 8, checking for returns to start.
        
        # Actually, let's just build the adjacency and do a limited BFS
        # from identity, checking if we can return to identity in 5-8 steps
        # without backtracking.
        
        # State: (current_element, prev_generator_index)
        # We track visited states to avoid re-exploration
        
        # But this could be expensive. Let's use a different approach:
        # Enumerate all non-backtracking words of length up to 8 and check if product is identity.
        
        # Number of non-backtracking words of length l in 5 generators: 5 * 4^(l-1)
        # Length 5: 5*4^4 = 1280
        # Length 6: 5*4^5 = 5120
        # Length 7: 5*4^6 = 20480
        # Length 8: 5*4^7 = 81920
        # Total: ~108800 words to check. Each involves ~8 matrix multiplications.
        # That's ~870k matrix multiplications. Should be fast enough.
        
        for length in range(5, 9):
            # Generate all non-backtracking words of this length
            # Start with each of the 5 generators
            stack = [(i, involutions[i]) for i in range(5)]
            for pos in range(1, length):
                new_stack = []
                for prev_idx, current in stack:
                    for next_idx in range(5):
                        if next_idx != prev_idx:  # non-backtracking
                            new_prod = _mat_mul(current, involutions[next_idx])
                            new_stack.append((next_idx, new_prod))
                stack = new_stack
            
            # Check if any word equals identity
            for _, prod in stack:
                if prod == ident:
                    return True
        
        return False
    
    # Backtracking search for 5 involutions satisfying conditions 1 and 2
    # Then verify condition 3
    
    best_result = None
    
    def backtrack(selected, start_idx):
        nonlocal best_result
        if time.time() > deadline - 5:
            return
        
        if len(selected) == 5:
            S_list = [involutions[i] for i in selected]
            if check_short_word_5_8(S_list):
                return
            # Build graph and verify with BFS
            adj = [[] for _ in range(n)]
            for u in range(n):
                for s in S_list:
                    v = _mat_mul(elements[u], s)
                    adj[u].append(idx[v])
            
            if _girth_bfs(n, adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                best_result = (n, edges)
                return
        
        for i in range(start_idx, m):
            # Check condition 1: for all j in selected, inv_prod_is_inv[j][i] must be False
            valid = True
            for j in selected:
                if inv_prod_is_inv[j][i]:
                    valid = False
                    break
                if inv_prod_is_inv[i][j]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check condition 2: for all j in selected, inv_prod_eq[j][i] must not be in selected
            # and inv_prod_eq[i][j] must not be in selected
            for j in selected:
                if inv_prod_eq[j][i] != -1 and inv_prod_eq[j][i] in selected:
                    valid = False
                    break
                if inv_prod_eq[i][j] != -1 and inv_prod_eq[i][j] in selected:
                    valid = False
                    break
            if not valid:
                continue
            
            selected.append(i)
            backtrack(selected, i + 1)
            selected.pop()
    
    # Try to find a valid set
    backtrack([], 0)
    
    if best_result:
        return best_result
    
    # If backtracking didn't find one, try random sampling as fallback
    rng = random.Random(42)
    for _ in range(500):
        if time.time() > deadline - 5:
            break
        S = set(rng.sample(involutions, 5))
        S_list = list(S)
        
        # Quick check conditions 1 and 2
        valid = True
        for i in range(5):
            for j in range(i + 1, 5):
                if involutions[S_list.index(S_list[i])] if False else True:
                    pass
                # This is getting complicated. Let's just do the full check.
        
        if check_short_word_5_8(S_list):
            continue
        
        adj = [[] for _ in range(n)]
        for u in range(n):
            for s in S_list:
                v = _mat_mul(elements[u], s)
                adj[u].append(idx[v])
        
        if _girth_bfs(n, adj, g):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
    
    return None


def _girth_bfs(n, adj, limit):
    """Exact girth via non-backtracking BFS from every vertex. Returns True if
    girth >= limit."""
    for s in range(n):
        dist = {s: 0}
        prev = {s: -1}
        dq = deque([s])
        while dq:
            u = dq.popleft()
            du = dist[u]
            if du + 1 >= limit:
                continue
            for v in adj[u]:
                if v == prev.get(u, -1):
                    continue
                if v not in dist:
                    dist[v] = du + 1
                    prev[v] = u
                    dq.append(v)
                else:
                    if du + 1 + dist[v] < limit:
                        return False
    return True


def _bipartite_luw(q, d):
    """Incidence graph of Lazebnik-Ustimenko-Woldar CD(d,q)."""
    q = int(q)
    d = int(d)
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
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def construct(k, g):
    t0 = time.time()
    deadline = t0 + _DEADLINE

    # Baseline: LUW construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    n_luw, edges_luw = _bipartite_luw(q, d)
    
    best_n, best_edges = n_luw, edges_luw
    
    # Try SL(2,11) Cayley graph - this should give 1320 vertices
    if k == 5 and g == 9:
        result_sl2 = _construct_sl2_cayley(k, g, deadline)
        if result_sl2:
            n_sl2, edges_sl2 = result_sl2
            if n_sl2 < best_n:
                best_n, best_edges = n_sl2, edges_sl2

    return best_n, best_edges
