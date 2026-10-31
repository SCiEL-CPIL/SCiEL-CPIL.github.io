import time
import random
import numpy as np
from collections import deque

# Keep the original LUW code as a fallback
def _luw_construct(k, g):
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


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True


def _heisenberg_cayley(k, g, deadline):
    """
    Construct a Cayley graph of a finite Heisenberg group over a field.
    For k=3, use F_p with 3 generators (and their inverses handled by symmetry).
    Actually, for 3-regular, we need 3 generators where each generator is its own inverse
    or we use a set of generators {a, b, c} where a^-1, b^-1, c^-1 are also in the set.
    Simplest: use an abelian group (Z_m)^d with generators that are involutions? No, Z_m doesn't have involutions for odd m.
    
    Let's use the Heisenberg group H_q over F_q for odd prime q.
    Elements: (x, y, z) with x, y, z in F_q.
    Multiplication: (x1, y1, z1) * (x2, y2, z2) = (x1+x2, y1+y2, z1+z2 + x1*y2)
    Inverse: (x, y, z)^-1 = (-x, -y, -z - x*y)
    
    We need 3 generators S = {g1, g2, g3} such that the Cayley graph is 3-regular.
    This requires that the generators are distinct from identity and we don't include inverses separately?
    No, a Cayley graph on a set S is k-regular if |S| = k.
    If the group is not abelian, the graph is undirected iff S = S^-1.
    So we need a symmetric generating set of size 3.
    This is impossible for a non-abelian group if the group has odd order?
    In H_q, the center has order q. If q is odd, the center elements have odd order, so no involutions in the center.
    However, we can pick generators g1, g2, g3 such that {g1, g2, g3} = {g1^-1, g2^-1, g3^-1}.
    This implies each generator is an involution.
    Does H_q have involutions?
    (x, y, z)^2 = (2x, 2y, 2z + xy).
    For this to be identity (0,0,0), we need 2x=0 => x=0 (since q odd), 2y=0 => y=0, 2z=0 => z=0.
    So H_q has no involutions for odd q.
    
    Therefore, we cannot form a 3-regular undirected Cayley graph on H_q with symmetric generators of size 3.
    
    Alternative: Use a bipartite double cover? Or use a different group?
    Or use a voltage graph approach which is more flexible.
    
    Let's stick to the voltage graph approach but optimize it.
    The Petersen lift was a good direction.
    Let's try lifting other base graphs.
    The dodecahedron graph is 3-regular with girth 5.
    Lifting the dodecahedron might give better girth/size ratio?
    
    Actually, the Ramanujan-Margulis construction via Cayley graphs of SL(2, F_q) is the gold standard.
    But implementing that is complex.
    
    Let's refine the Petersen lift.
    The issue with the previous attempt was likely finding voltages.
    Let's use a more systematic search for voltages or a different base graph.
    
    How about lifting the 3-prism (two triangles connected)? No, girth 3.
    How about lifting the cube (Q3)? Girth 4.
    
    Let's try to implement a simple random search for a 3-regular graph with high girth,
    but that is very hard to get girth 16.
    
    Let's go back to the voltage graph idea.
    Base graph: Petersen (girth 5).
    We need girth >= 16.
    This means we need to avoid cycles of length 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 in the lift.
    A cycle in the lift corresponds to a closed walk in the base graph.
    If the base walk has length L and voltage sum V, the lift has a cycle of length L if V=0 mod m.
    If V != 0, the lift has no cycle of length L, but might have a cycle of length k*L?
    Actually, the lift is a covering. A closed walk in the base of length L lifts to a cycle of length L in the fiber if and only if the voltage sum is 0 mod m.
    If the voltage sum is V != 0, the lift of that walk is a path from fiber 0 to fiber V. It is not a cycle.
    However, if we take the walk L/m times (if m divides L?), no.
    The cycle length in the lift is the smallest k*L such that k*V = 0 mod m.
    So if V != 0, the cycle length is m/gcd(V, m) * L? No.
    The lift of a base cycle of length L with voltage sum V is a cycle in the lift iff V = 0 mod m.
    If V != 0, the lift is not a cycle. It is a path.
    To get a cycle, we need to traverse the base cycle multiple times?
    If we traverse the base cycle k times, the total length is k*L and the total voltage is k*V.
    This is a cycle in the lift if k*V = 0 mod m.
    The smallest such k is m/gcd(V, m).
    So the cycle length in the lift is (m/gcd(V, m)) * L.
    
    So, for a base cycle of length L, the lift has a cycle of length (m/gcd(V, m)) * L.
    We need this to be >= g for all base cycles.
    If L >= g, we are fine (since m/gcd >= 1).
    If L < g, we need (m/gcd(V, m)) * L >= g.
    Since m/gcd(V, m) >= 1, if L >= g, ok.
    If L < g, we need m/gcd(V, m) >= ceil(g/L).
    This is a weaker condition than V != 0.
    V != 0 implies m/gcd(V, m) >= 2 (if m is prime, then m).
    If m is prime, V != 0 implies m/gcd = m.
    So cycle length is m*L.
    We need m*L >= g.
    Since L >= 5 (girth of Petersen), we need 5m >= g.
    For g=16, m >= 16/5 = 3.2 => m >= 4.
    So any prime m >= 5 should work if we ensure V != 0 for all base cycles of length < g.
    
    Wait, are there base cycles of length < 5? No, girth is 5.
    So we only need to check cycles of length 5, 6, ..., g-1.
    For the Petersen graph, the cycles are of length 5, 6, 8, 9, 10.
    We need to assign voltages to the 15 edges such that for every cycle C of length L < g, the sum of voltages on C is not 0 mod m.
    
    This is a constraint satisfaction problem.
    Let's try m = 5, 7, 11, 13, 17...
    For m=5, we need to avoid sum=0 mod 5 for cycles of length 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15.
    But Petersen has no cycles of length 7 or 11, 13, 15.
    It has cycles of length 5, 6, 8, 9, 10.
    So we need to avoid sum=0 mod 5 for cycles of length 5, 6, 8, 9, 10.
    
    Let's write a function to find such voltages using backtracking or random search.
    """
    if k != 3:
        return None

    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))

    # Find all cycles in Petersen up to length g-1
    cycles = []
    seen = set()
    
    def dfs(start, curr, visited, edge_path, length):
        if length >= g:
            return
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
                
    for start in range(n_base):
        visited = {start}
        dfs(start, start, visited, [], 0)
        
    # We need to find voltages for m such that sum(v[C]) != 0 mod m for all C in cycles.
    # Try small primes m.
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for m in primes:
        if time.time() > deadline:
            break
        
        # Random search for voltages
        found = False
        for attempt in range(1000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                # Construct lift
                n = 10 * m
                edges = []
                for idx, (u, v) in enumerate(petersen_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        edges.append((a, b))
                return n, edges
    return None


def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin

    if k != 3:
        return _luw_construct(k, g)

    # Try Heisenberg/Cayley approach (actually voltage lift of Petersen)
    result = _heisenberg_cayley(k, g, deadline)
    if result is not None:
        n, edges = result
        # Verify
        if _check_girth_fast(n, edges, g):
            return n, edges
        else:
            # Should not happen
            pass

    # Fallback to LUW
    return _luw_construct(k, g)
