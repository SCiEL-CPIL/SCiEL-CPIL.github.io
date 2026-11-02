import time
import sys
from collections import deque

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

def _next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not _is_prime(n):
        n += 2
    return n

def _girth_ok_abelian(S, q, g):
    """
    Check if Cayley graph on Z_q with generating set S has girth >= g.
    S is symmetric (S == -S mod q), 0 not in S, |S| = k.
    Girth >= g iff no non-backtracking closed walk of length < g.
    In abelian group, non-backtracking closed walk of length t corresponds to
    sequence s_1..s_t in S with sum = 0 mod q, s_i != -s_{i+1} for all i (cyclic).
    We check lengths 2..g-1 via BFS on (last_gen_idx, current_sum).
    """
    S_list = list(S)
    nS = len(S_list)
    S_set = set(S_list)
    neg_map = {x: (-x) % q for x in S_set}
    
    target_len = g - 1  # we check lengths 2..g-1
    
    for s1_idx in range(nS):
        s1 = S_list[s1_idx]
        frontier = {(s1_idx, s1)}
        
        for t in range(2, target_len + 1):
            new_frontier = set()
            for last_idx, cur_sum in frontier:
                last_gen = S_list[last_idx]
                neg_last = neg_map[last_gen]
                for j in range(nS):
                    sj = S_list[j]
                    if sj == neg_last:
                        continue
                    new_sum = (cur_sum + sj) % q
                    new_frontier.add((j, new_sum))
            
            for last_idx, cur_sum in new_frontier:
                if cur_sum == 0:
                    if S_list[last_idx] != neg_map[s1]:
                        return False
            
            if not new_frontier:
                break
            frontier = new_frontier
    
    return True

def _try_cayley_graph(k, g, deadline, max_q=1200):
    """
    Search for a Cayley graph on Z_q with generating set of size k,
    symmetric, girth >= g. Try q values starting from Moore bound.
    Returns (n, edges) or None.
    """
    if g % 2 == 1:
        r = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(r))
    else:
        r = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(r))
    
    q_min = max(moore, k + 1)
    q = _next_prime(q_min)
    
    while q <= max_q and time.time() < deadline:
        if not _is_prime(q):
            q += 2
            continue
        
        candidates = []
        
        bases_to_try = []
        for base in range(1, 25):
            bases_to_try.append(base)
        for denom in range(2, 12):
            val = q // denom
            if 0 < val < q:
                bases_to_try.append(val)
            val2 = (q * denom) // (denom * denom)
            if 0 < val2 < q:
                bases_to_try.append(val2)
        
        seen = set()
        bases_unique = []
        for b in bases_to_try:
            b_mod = b % q
            if 0 < b_mod < q and b_mod not in seen:
                seen.add(b_mod)
                bases_unique.append(b_mod)
        
        found = False
        for a in bases_unique[:20]:
            for b in bases_unique[:20]:
                if b == a or b == (-a) % q:
                    continue
                c_options = []
                for c in bases_unique[:15]:
                    if c not in {a, b, (-a) % q, (-b) % q, 0}:
                        c_options.append(c)
                
                for c in c_options:
                    S = [c, a, b, (-a) % q, (-b) % q]
                    S_set = set(S)
                    if len(S_set) != k:
                        continue
                    if 0 in S_set:
                        continue
                    if S_set != {(-x) % q for x in S_set}:
                        continue
                    if _girth_ok_abelian(S, q, g):
                        n = q
                        edges = []
                        S_list = list(S_set)
                        for i in range(n):
                            for s in S_list:
                                j = (i + s) % q
                                if j > i:
                                    edges.append((i, j))
                        return n, edges
        
        if time.time() > deadline - 30:
            break
        
        q += 2
    
    return None

def _luw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar bipartite graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
            yield 1 - side, tuple(y)
    
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

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Tries Cayley graph on Z_q first (smaller), falls back to LUW construction.
    """
    t0 = time.time()
    deadline = t0 + 110.0
    
    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []
    
    result = _try_cayley_graph(k, g, deadline, max_q=1200)
    if result is not None:
        return result
    
    return _luw_construct(k, g)
