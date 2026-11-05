import time
import numpy as np
from collections import deque
import math

def construct(k, g):
    """Construct a k-regular graph with girth at least g."""
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    if k != 4:
        return _construct_general(k, g, start_time, deadline)
    
    best_result = None
    
    # Strategy 1: Search cyclic groups Z_m with S = {a, b, -a, -b}
    # Moore bound for (4,10) is 243, so m >= 243
    # Fast DP-based girth check
    
    for m in range(243, 500):
        if time.time() > deadline - 20:
            break
        
        found = _find_cyclic_params(m, g, deadline)
        if found is not None:
            a, b = found
            edges = _build_cyclic_edges(m, a, b)
            if best_result is None or m < best_result[0]:
                best_result = (m, edges)
            break  # Found smallest m, we're done with cyclic
    
    if best_result is not None:
        return best_result
    
    # Strategy 2: Search Z_a × Z_b with S = {(1,0),(-1,0),(a,1),(-a,1)}
    # This gives 2*n vertices where n = a*b... actually let's use Z_n × Z_2
    for n in range(122, 300):
        if time.time() > deadline - 15:
            break
        
        for a in range(1, n):
            if time.time() > deadline - 12:
                break
            
            if (2 * a) % n == 0:
                continue
            
            if _check_girth_ab2_fast(n, a, g, deadline):
                edges = _build_edges_ab2(n, a)
                total_vertices = 2 * n
                if best_result is None or total_vertices < best_result[0]:
                    best_result = (total_vertices, edges)
                break
    
    if best_result is not None:
        return best_result
    
    # Strategy 3: LUW construction fallback
    result = _construct_luw(4, g, start_time, deadline)
    if result is not None:
        return result
    
    # Last resort
    return _construct_general(k, g, start_time, deadline)


def _find_cyclic_params(m, g, deadline):
    """Find (a, b) such that Cayley graph on Z_m with S={a,b,-a,-b} has girth >= g."""
    # For each pair (a, b), check girth using DP
    # To be efficient, iterate b from a+1 upward and check
    
    for a in range(1, m // 2 + 1):
        if time.time() > deadline:
            return None
        for b in range(a + 1, m // 2 + 1):
            # Quick checks
            if a == m - a or b == m - b:
                continue
            if a == b or a == m - b or b == m - a:
                continue
            if math.gcd(math.gcd(a, b), m) != 1:
                continue
            
            if _check_girth_cyclic_fast(m, a, b, g, deadline):
                return (a, b)
    
    return None


def _check_girth_cyclic_fast(m, a, b, g, deadline):
    """Check girth >= g for Cayley graph on Z_m with S={a,b,-a,-b} using DP."""
    # Non-backtracking closed walks: sequence s1,...,sL with si in S, s_{i+1} != -s_i, sum = 0 mod m
    # DP state: (pos, last_gen_index) where last_gen_index in {0,1,2,3} for {a,b,-a,-b}
    # S = [a, b, -a, -b]
    S = [a, b, -a, -b]
    # inv: -a is index 2, -b is index 3, a is index 0, b is index 1
    inv_idx = [2, 3, 0, 1]
    
    # For each L from 3 to g-1, count non-backtracking closed walks of length L
    # DP: dp[step][pos][last_idx] = number of non-backtracking walks of length 'step' ending at pos with last step being S[last_idx]
    # We only need to check if count > 0 for any L < g
    
    for L in range(3, g):
        if time.time() > deadline:
            return False  # Conservative
        
        # dp[pos][last_idx] = count of NB walks of current length ending at pos with last gen = last_idx
        dp = [[0] * 4 for _ in range(m)]
        
        # Initialize: walks of length 1
        for i, s in enumerate(S):
            pos = s % m
            dp[pos][i] += 1
        
        # Extend to length L
        for step in range(2, L + 1):
            new_dp = [[0] * 4 for _ in range(m)]
            for pos in range(m):
                for last_i in range(4):
                    count = dp[pos][last_i]
                    if count == 0:
                        continue
                    # Can't backtrack: next gen != -last_gen
                    forbidden = inv_idx[last_i]
                    for next_i in range(4):
                        if next_i == forbidden:
                            continue
                        new_pos = (pos + S[next_i]) % m
                        new_dp[new_pos][next_i] += count
            
            dp = new_dp
        
        # Check if any closed walk: dp[0][i] > 0 for any i
        closed_count = sum(dp[0])
        if closed_count > 0:
            return False
    
    return True


def _build_cyclic_edges(m, a, b):
    """Build edge list for Cayley graph on Z_m with S={a,b,-a,-b}."""
    edge_set = set()
    for i in range(m):
        for s in [a, b]:
            j = (i + s) % m
            if i < j:
                edge_set.add((i, j))
            else:
                edge_set.add((j, i))
    return sorted(edge_set)


def _check_girth_ab2_fast(n, a, g, deadline):
    """Check girth for Cayley graph on Z_n × Z_2 with S={(1,0),(-1,0),(a,1),(-a,1)} using DP."""
    # S = [(1,0), (-1,0), (a,1), (-a,1)]
    # inv: (1,0)<->(-1,0), (a,1)<->(-a,1)
    # State: (pos_mod_n, bit, last_idx) where last_idx in {0,1,2,3}
    # pos is in Z_n × Z_2, represented as (i, z) -> i*2+z for indexing
    
    S = [(1, 0), (-1, 0), (a, 1), (-a, 1)]
    inv_idx = [1, 0, 3, 2]
    
    total_states = n * 2  # n positions in Z_n, 2 bits
    
    for L in range(3, g):
        if time.time() > deadline:
            return False
        
        # dp[state][last_idx] where state = i*2 + z
        dp = [[0] * 4 for _ in range(total_states)]
        
        # Initialize: walks of length 1
        for i, (dx, dz) in enumerate(S):
            pos = (dx % n) * 2 + dz
            dp[pos][i] += 1
        
        # Extend to length L
        for step in range(2, L + 1):
            new_dp = [[0] * 4 for _ in range(total_states)]
            for state in range(total_states):
                for last_i in range(4):
                    count = dp[state][last_i]
                    if count == 0:
                        continue
                    forbidden = inv_idx[last_i]
                    for next_i in range(4):
                        if next_i == forbidden:
                            continue
                        dx, dz = S[next_i]
                        new_i = ((state // 2) + dx) % n
                        new_z = (state % 2) + dz
                        new_state = new_i * 2 + (new_z % 2)
                        new_dp[new_state][next_i] += count
            
            dp = new_dp
        
        # Check closed: state 0 (position (0,0))
        closed_count = sum(dp[0])
        if closed_count > 0:
            return False
    
    return True


def _build_edges_ab2(n, a):
    """Build edge list for Cayley graph on Z_n x Z_2."""
    edges = set()
    for i in range(n):
        for z in [0, 1]:
            v = i * 2 + z
            neighbors = [
                ((i + 1) % n) * 2 + z,
                ((i - 1) % n) * 2 + z,
                ((i + a) % n) * 2 + (1 - z),
                ((i - a) % n) * 2 + (1 - z)
            ]
            for u in neighbors:
                if v < u:
                    edges.add((v, u))
                else:
                    edges.add((u, v))
    
    return sorted(edges)


def _construct_luw(k, g, start_time, deadline):
    """LUW construction fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    while time.time() < deadline:
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
        
        n = len(order)
        
        # Verify girth
        adj = {i: [] for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if _verify_girth(adj, n, g):
            return n, edges
        else:
            d += 1
    
    return None


def _verify_girth(adj, n, g):
    """Verify girth >= g using BFS from each vertex with parent tracking."""
    for start in range(n):
        visited = {start: (-1, 0)}
        queue = deque([(start, -1, 0)])
        while queue:
            node, parent, dist = queue.popleft()
            if dist >= g:
                continue
            for nei in adj[node]:
                if nei == parent:
                    continue
                if nei in visited:
                    cycle_len = dist + visited[nei][1] + 1
                    if cycle_len < g:
                        return False
                else:
                    visited[nei] = (node, dist + 1)
                    queue.append((nei, node, dist + 1))
    return True


def _construct_general(k, g, start_time, deadline):
    """General fallback for other (k, g) values."""
    if k == 4:
        return _construct_luw(4, g, start_time, deadline)
    
    n = 2 * k * (g // 2)
    edges = []
    for i in range(n):
        edges.append((i, (i + 1) % n))
        for j in range(2, k + 1):
            if j == k // 2 and k % 2 == 0:
                break
            j2 = n - j if j > 1 else 1
            if j2 != 1:
                edges.append((i, (i + j2) % n))
    
    unique_edges = set()
    for u, v in edges:
        if u != v:
            unique_edges.add((min(u, v), max(u, v)))
    
    return n, sorted(unique_edges)
