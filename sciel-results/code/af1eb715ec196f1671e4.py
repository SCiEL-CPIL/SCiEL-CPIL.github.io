"""Cayley-graph construction for 4-regular girth-10 graph."""
import math
from collections import deque

def construct(k, g):
    """Construct a k-regular graph of girth >= g with as few vertices as possible.
    
    Uses Cayley graphs of finite abelian groups with symmetric generating sets.
    """
    if k != 4:
        return _construct_fallback(k, g)
    
    best_n = None
    best_edges = None
    
    # Moore bound for (4, 10) is 243
    moore_bound = 243
    
    # Strategy 1: Try cyclic groups Z_m
    for m in range(moore_bound, moore_bound + 200):
        result = _try_cyclic_group(m, g)
        if result is not None:
            n, edges = result
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
            break  # Found smallest m, done with cyclic
    
    # Strategy 2: Try Z_a x Z_b groups
    if best_n is None or best_n > 300:
        for a in range(3, 40):
            for b in range(a, 40):
                n = a * b
                if n < moore_bound:
                    continue
                if best_n is not None and n >= best_n:
                    break
                result = _try_product_group(a, b, g)
                if result is not None:
                    n2, edges = result
                    if best_n is None or n2 < best_n:
                        best_n = n2
                        best_edges = edges
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback
    return _construct_fallback(4, g)


def _try_cyclic_group(m, g):
    """Try to find a 4-regular Cayley graph on Z_m with girth >= g."""
    # Generating set S = {a, -a, b, -b} with 1 <= a < b < m/2
    # Need: no non-backtracking closed walk of length < g
    
    # Prune: a and b must be coprime to m for connectedness (or at least gcd(a,b,m)=1)
    for a in range(1, m // 2 + 1):
        for b in range(a + 1, m // 2 + 1):
            # Skip if a == m-a or b == m-b (would create 2-cycles)
            if 2 * a % m == 0 or 2 * b % m == 0:
                continue
            # Skip if a == b, a == -b, b == -a (mod m)
            if a == b or a == m - b or b == m - a:
                continue
            # Check connectedness
            if math.gcd(math.gcd(a, b), m) != 1:
                continue
            
            # Verify girth
            if _check_girth_cyclic(m, a, b, g):
                # Build edges
                edge_set = set()
                for i in range(m):
                    for s in [a, b]:
                        j = (i + s) % m
                        if i < j:
                            edge_set.add((i, j))
                        else:
                            edge_set.add((j, i))
                edges = sorted(edge_set)
                return m, edges
    
    return None


def _check_girth_cyclic(m, a, b, g):
    """Check girth >= g for Cayley graph of Z_m with S={a,-a,b,-b} using BFS on non-backtracking states."""
    # State: (position, last_step) where last_step is in {0,1,2,3} for {a,-a,b,-b}
    # We want to find if there's a non-backtracking closed walk of length < g
    
    S = [a, -a, b, -b]
    neg_idx = {0: 1, 1: 0, 2: 3, 3: 2}  # index of -S[i]
    
    # BFS: state = (pos, last_step_idx, length)
    # We track visited states to avoid re-exploring
    # A closed walk means we return to pos=0 with some last_step
    
    # For each starting step, do BFS up to depth g-1
    # If we reach (0, any_last_step) at depth < g, we have a short cycle
    
    for start_idx in range(4):
        start_pos = S[start_idx] % m
        # BFS from (start_pos, start_idx)
        # visited: set of (pos, last_idx) at each depth
        visited = set()
        visited.add((start_pos, start_idx))
        queue = deque([(start_pos, start_idx, 1)])
        
        while queue:
            pos, last_idx, depth = queue.popleft()
            
            if depth >= g - 1:
                continue
            
            # Check if we can close the walk
            # A closed walk of length L means after L steps we're back at 0
            # But we need the walk to be non-backtracking throughout
            # If pos == 0 and depth >= 3, we have a closed walk of length depth
            # (since we started with one step, total length = depth)
            if pos == 0 and depth >= 3:
                return False
            
            # Extend
            for next_idx in range(4):
                if next_idx == neg_idx[last_idx]:
                    continue  # backtracking
                new_pos = (pos + S[next_idx]) % m
                new_state = (new_pos, next_idx)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_pos, next_idx, depth + 1))
    
    return True


def _try_product_group(a, b, g):
    """Try to find a 4-regular Cayley graph on Z_a x Z_b with girth >= g."""
    # Generating set S = {(s1a, s1b), (-s1a, -s1b), (s2a, s2b), (-s2a, -s2b)}
    # Need to choose s1, s2 in Z_a x Z_b such that the Cayley graph has girth >= g
    
    n = a * b
    # Enumerate possible generators
    # s1 = (x1, y1), s2 = (x2, y2)
    # For efficiency, fix s1 = (1, 0) by symmetry (if a > 1)
    # Actually, let's try a few specific patterns
    
    # Pattern 1: s1 = (1, 0), s2 = (0, 1) - grid-like
    # Pattern 2: s1 = (1, 0), s2 = (1, 1)
    # Pattern 3: s1 = (1, 0), s2 = (0, 2)
    # etc.
    
    candidates = []
    # Try s1 = (1, 0)
    candidates.append(((1, 0), (0, 1)))
    candidates.append(((1, 0), (0, 2)))
    candidates.append(((1, 0), (1, 1)))
    candidates.append(((1, 0), (1, 2)))
    candidates.append(((1, 0), (2, 1)))
    candidates.append(((1, 0), (2, 2)))
    candidates.append(((1, 0), (0, 3)))
    candidates.append(((1, 0), (1, 3)))
    candidates.append(((1, 1), (0, 1)))
    candidates.append(((1, 1), (1, 0)))
    
    for s1, s2 in candidates:
        # Validate: s1, s2, -s1, -s2 must be distinct
        if s1 == s2 or s1 == (-s1[0] % a, -s1[1] % b) or s2 == (-s2[0] % a, -s2[1] % b):
            continue
        if s1 == (-s2[0] % a, -s2[1] % b) or s2 == (-s1[0] % a, -s1[1] % b):
            continue
        
        S = [s1, (-s1[0] % a, -s1[1] % b), s2, (-s2[0] % a, -s2[1] % b)]
        
        # Check connectedness
        if not _is_connected_product(a, b, S):
            continue
        
        # Check girth
        if _check_girth_product(a, b, S, g):
            # Build edges
            edge_set = set()
            for i in range(a):
                for j in range(b):
                    v = i * b + j
                    for s in S:
                        ni = (i + s[0]) % a
                        nj = (j + s[1]) % b
                        u = ni * b + nj
                        if v < u:
                            edge_set.add((v, u))
                        else:
                            edge_set.add((u, v))
            edges = sorted(edge_set)
            return n, edges
    
    return None


def _is_connected_product(a, b, S):
    """Check if the Cayley graph of Z_a x Z_b with generating set S is connected."""
    # BFS from (0,0)
    visited = set()
    visited.add((0, 0))
    queue = deque([(0, 0)])
    
    while queue:
        i, j = queue.popleft()
        for s in S:
            ni = (i + s[0]) % a
            nj = (j + s[1]) % b
            if (ni, nj) not in visited:
                visited.add((ni, nj))
                queue.append((ni, nj))
    
    return len(visited) == a * b


def _check_girth_product(a, b, S, g):
    """Check girth >= g for Cayley graph of Z_a x Z_b with generating set S."""
    # S has 4 elements: s0, s1, s2, s3 where s1 = -s0, s3 = -s2
    neg_idx = {0: 1, 1: 0, 2: 3, 3: 2}
    
    n = a * b
    
    for start_idx in range(4):
        si, sj = S[start_idx]
        start_pos = (si % a, sj % b)
        
        visited = set()
        visited.add((start_pos[0], start_pos[1], start_idx))
        queue = deque([(start_pos[0], start_pos[1], start_idx, 1)])
        
        while queue:
            i, j, last_idx, depth = queue.popleft()
            
            if depth >= g - 1:
                continue
            
            # Check if we're back at origin
            if i == 0 and j == 0 and depth >= 3:
                return False
            
            for next_idx in range(4):
                if next_idx == neg_idx[last_idx]:
                    continue
                ni = (i + S[next_idx][0]) % a
                nj = (j + S[next_idx][1]) % b
                new_state = (ni, nj, next_idx)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((ni, nj, next_idx, depth + 1))
    
    return True


def _construct_fallback(k, g):
    """Fallback: use a known construction or return a valid graph."""
    # Use the Lazebnik-Ustimenko-Woldar style construction
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
