import time
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

def _girth_check(S, q, g):
    """Check if Cayley graph on Z_q with generating set S has girth >= g.
    S is symmetric, 0 not in S, |S| = k.
    Uses non-backtracking walk check from each starting generator."""
    S_list = list(S)
    nS = len(S_list)
    S_set = set(S_list)
    neg_map = {x: (-x) % q for x in S_set}
    
    target_len = g - 1
    
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

def _cayley_edges(S, q):
    """Generate edges for Cayley graph on Z_q with generating set S."""
    edges = []
    S_list = list(S)
    for i in range(q):
        for s in S_list:
            j = (i + s) % q
            if j > i:
                edges.append((i, j))
    return edges

def _try_cayley(k, g, deadline, max_q=2000):
    """Search for Cayley graph on Z_q with girth >= g."""
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
        
        # Build candidate bases
        bases = list(range(1, 30))
        for denom in range(2, 15):
            val = q // denom
            if 0 < val < q:
                bases.append(val)
        
        seen = set()
        bases_unique = []
        for b in bases:
            b_mod = b % q
            if 0 < b_mod < q and b_mod not in seen:
                seen.add(b_mod)
                bases_unique.append(b_mod)
        
        # Try generating sets of form {a, b, c, -a, -b, -c} for k=6
        # or {a, b, c, -a, -b} for k=5 (c must be self-inverse, i.e., c = -c mod q, so 2c ≡ 0 mod q)
        # For odd prime q, self-inverse elements: only 0, which is excluded.
        # So for k=5 with odd prime, we need {a, b, -a, -b, c} where c ≠ ±a, ±b, 0
        # But this isn't symmetric unless c = -c, which requires 2c ≡ 0 mod q.
        # For odd prime, only c=0 works, which is excluded.
        # So for k=5, we need a different approach.
        
        # Actually for k=5, a symmetric generating set of size 5 on Z_q (odd prime)
        # must have one element that is its own inverse, but that's only 0 for odd primes.
        # So we can't have a symmetric generating set of odd size > 1 on Z_q for odd prime q.
        # Wait, that's wrong. A symmetric set S means if x in S then -x in S.
        # For |S| = 5 (odd), we need at least one element x with x = -x, i.e., 2x ≡ 0 mod q.
        # For odd prime q, x = 0. But 0 can't be in S.
        # So for k=5 and odd prime q, no symmetric generating set of size 5 exists!
        
        # For k=5, we need even prime (q=2, too small) or composite q.
        # Let's try composite q values too.
        
        # For now, try k=5 with composite q
        if k == 5:
            # Try q = p * something, or just try various q values
            # Actually, let's try q values that are not prime
            pass
        
        # For general k, try all triples
        if k == 5:
            # For k=5, symmetric set needs one self-inverse element
            # Only possible if q is even (then q/2 is self-inverse) or q has factor 2
            # For odd q, impossible. So try even q or q with 2 as factor.
            # Let's try q = 2*p for prime p, or q = p*q2
            # Actually, let's just try q values from q_min to max_q
            pass
        
        # Try all combinations of 3 bases for k=6, or handle k=5 separately
        if k == 6:
            for i in range(min(20, len(bases_unique))):
                a = bases_unique[i]
                for j in range(i + 1, min(20, len(bases_unique))):
                    b = bases_unique[j]
                    if b == (-a) % q:
                        continue
                    for l in range(j + 1, min(20, len(bases_unique))):
                        c = bases_unique[l]
                        if c in {a, b, (-a) % q, (-b) % q, 0}:
                            continue
                        S = {a, b, c, (-a) % q, (-b) % q, (-c) % q}
                        if len(S) != 6 or 0 in S:
                            continue
                        if _girth_check(S, q, g):
                            return q, _cayley_edges(S, q)
        
        elif k == 5:
            # For k=5, need q even or q with specific properties
            # Try q = 2 * p for prime p
            # Or just skip and use LUW
            pass
        
        q += 1 if k == 5 else 2
    
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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

def _try_cayley_composite(k, g, deadline, max_q=1500):
    """Try Cayley graphs on composite groups Z_q for k=5."""
    # For k=5, we need a symmetric generating set of size 5 on Z_q
    # This requires q to be even (so q/2 is self-inverse)
    # S = {a, b, q/2, -a, -b} where a, b not 0, not q/2, not ±each other
    
    if k != 5:
        return None
    
    # Moore bound
    r = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(r))
    q_min = max(moore, 10)
    
    q = q_min
    while q <= max_q and time.time() < deadline:
        if q % 2 == 0:
            half = q // 2
            # Try various a, b
            for a in range(1, min(30, half)):
                for b in range(1, min(30, half)):
                    if b in {a, (-a) % q, 0, half}:
                        continue
                    S = {a, b, half, (-a) % q, (-b) % q}
                    if len(S) != 5 or 0 in S:
                        continue
                    if _girth_check(S, q, g):
                        return q, _cayley_edges(S, q)
        
        q += 2
    
    return None

def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    t0 = time.time()
    deadline = t0 + 110.0
    
    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []
    
    # Try composite Cayley graph for k=5
    result = _try_cayley_composite(k, g, deadline, max_q=1500)
    if result is not None:
        return result
    
    # Try prime Cayley graph
    result = _try_cayley(k, g, deadline, max_q=2000)
    if result is not None:
        return result
    
    # Fallback to LUW
    return _luw_construct(k, g)
