"""
Construct a k-regular graph with girth at least g.
Primary: voltage graph lift of Petersen graph over cyclic group.
Fallback: algebraic circulant search + LUW.
"""
import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin out of 150s limit

    def check_deadline():
        return time.time() > deadline

    if k == 3 and g == 16:
        # Strategy 1: Voltage graph lift of Petersen graph
        result = _petersen_voltage_lift(3, 16, deadline)
        if result is not None:
            return result

        # Strategy 2: Fast algebraic circulant search
        if not check_deadline():
            result2 = _find_circulant_v2(3, 16, deadline)
            if result2 is not None:
                return result2

        # Strategy 3: LUW fallback with verification
        for d in [9, 11, 13]:
            if check_deadline():
                break
            try:
                n_luw, edges_luw = _luw_construct_with_d(3, 16, d)
                if n_luw is not None and _verify_girth_fast(n_luw, edges_luw, 16):
                    return n_luw, edges_luw
            except Exception:
                pass

    return _luw_construct(k, g)


def _petersen_voltage_lift(k, g, deadline):
    """
    Construct a k-regular graph by lifting the Petersen graph using
    voltage assignments over Z_m. The Petersen graph has 10 vertices and
    15 edges. A voltage lift over Z_m gives a 3-regular graph with 10*m
    vertices.
    
    For girth >= 16, we need m >= 16 (since the Petersen graph has girth 5,
    a lift of length 5 in the base becomes a cycle of length 5*m in the cover,
    but we need to check all cycle lengths).
    
    Actually, the girth of the lift depends on the voltage assignments.
    We search for voltage assignments that maximize the girth.
    
    Key insight: For a 3-regular graph with girth >= 16, the Moore bound
    gives n >= 2*(1+2+4+8+16+32+64+128) - 1 = 2*255 - 1 = 509 (non-bipartite)
    or n >= 2*(1+2+4+8+16+32+64+128) = 510 (bipartite).
    
    With Petersen lift: n = 10*m. So m >= 51 gives n >= 510.
    We search m from 51 upward, trying different voltage assignments.
    """
    # Petersen graph: 10 vertices, 15 edges
    # Standard Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer pentagon
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    # Petersen adjacency
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    
    # For each m, try different voltage assignments
    # Voltage assignment: for each edge (u,v), assign a value in Z_m
    # The lift has vertices (v, i) for v in 0..9, i in 0..m-1
    # Edge from (u, i) to (v, i + voltage(u,v) mod m)
    
    best_result = None
    best_n = float('inf')
    
    # Search m values. n = 10*m, need n >= 510 (Moore bound for girth 16)
    # Try m from 51 upward
    m_min = 51
    m_max = 200  # upper limit to keep runtime manageable
    
    for m in range(m_min, m_max + 1):
        if time.time() > deadline - 10:
            break
        
        # Try several voltage assignment patterns
        # Pattern type 1: Linear - voltage(e) = c * edge_index mod m
        # Pattern type 2: Based on vertex labels
        # Pattern type 3: Pseudo-random from a seed
        
        for pattern in range(12):
            if time.time() > deadline - 5:
                break
            
            voltages = _generate_voltage_assignment(petersen_edges, m, pattern)
            
            # Quick girth check on the lift
            if _lift_girth_check(petersen_adj, voltages, m, g):
                # Build the full graph
                n = 10 * m
                edges = []
                for u, v in petersen_edges:
                    vol = voltages[(u, v)]
                    for i in range(m):
                        a = u * m + i
                        b = v * m + ((i + vol) % m)
                        if a < b:
                            edges.append((a, b))
                        else:
                            edges.append((b, a))
                
                if len(edges) == 15 * m:  # 3-regular: 10*m*3/2 = 15*m edges
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break  # Found valid for this m, move to next m
        
        if best_result is not None:
            # We found a valid graph. Check if we can do better with smaller m
            # but we're going in increasing m order, so this is optimal for this pattern set
            # However, different patterns might work for smaller m
            # Continue searching a bit more
            pass
    
    return best_result


def _generate_voltage_assignment(petersen_edges, m, pattern):
    """Generate a voltage assignment for the Petersen graph edges."""
    voltages = {}
    
    for idx, (u, v) in enumerate(petersen_edges):
        if pattern == 0:
            # Linear: voltage = (idx * 7) % m
            vol = (idx * 7) % m
        elif pattern == 1:
            # Vertex-based: voltage = (u * 3 + v * 5) % m
            vol = (u * 3 + v * 5) % m
        elif pattern == 2:
            # Quadratic: voltage = (idx * idx + idx + 1) % m
            vol = (idx * idx + idx + 1) % m
        elif pattern == 3:
            # Mixed: voltage = (u * v + idx) % m
            vol = (u * v + idx) % m
        elif pattern == 4:
            # Fibonacci-like: voltage = (idx * 11 + 3) % m
            vol = (idx * 11 + 3) % m
        elif pattern == 5:
            # Prime-based: voltage = (idx * 13 + 7) % m
            vol = (idx * 13 + 7) % m
        elif pattern == 6:
            # Sum-based: voltage = (u + v + idx * 2) % m
            vol = (u + v + idx * 2) % m
        elif pattern == 7:
            # Product-based: voltage = (u * v + 1) % m
            vol = (u * v + 1) % m
        elif pattern == 8:
            # Cubic: voltage = (idx * idx * idx + idx) % m
            vol = (idx * idx * idx + idx) % m
        elif pattern == 9:
            # Alternate linear: voltage = (idx * 5 + u) % m
            vol = (idx * 5 + u) % m
        elif pattern == 10:
            # Alternate: voltage = (idx * 17 + v) % m
            vol = (idx * 17 + v) % m
        else:
            # Pseudo-random via LCG
            seed = idx * 2654435761 + 12345
            seed = (seed * 1103515245 + 12345) % (2**31)
            vol = seed % m
        
        voltages[(u, v)] = vol
        # For the reverse direction, voltage should be -vol mod m
        voltages[(v, u)] = (-vol) % m
    
    return voltages


def _lift_girth_check(petersen_adj, voltages, m, g):
    """
    Check if the voltage lift of Petersen graph has girth >= g.
    Uses non-backtracking BFS from each vertex in the base graph,
    tracking the accumulated voltage.
    
    A cycle of length L in the lift corresponds to a closed walk of length L
    in the base graph where the sum of voltages is 0 mod m.
    """
    # For the lift, a cycle exists iff there's a closed walk in the base
    # with total voltage = 0 mod m.
    # We need to check all closed walks of length < g.
    # 
    # State: (vertex, prev_vertex, accumulated_voltage)
    # We do non-backtracking BFS from each starting vertex.
    
    for start in range(10):
        # State: (current_vertex, prev_vertex, voltage_sum)
        # Start: (start, -1, 0)
        visited = set()
        queue = deque()
        queue.append((start, -1, 0))
        visited.add((start, -1, 0))
        
        max_len = g - 1
        
        while queue:
            u, prev, vol = queue.popleft()
            d = len(visited)  # not quite right, need depth
            # Actually let's track depth properly
            pass
        
        # Redo with proper depth tracking
        if _lift_bfs_from(start, petersen_adj, voltages, m, g):
            return False
    
    return True


def _lift_bfs_from(start, petersen_adj, voltages, m, g):
    """
    Non-backtracking BFS from 'start' in the Petersen graph,
    tracking accumulated voltage. Returns True if a cycle of length < g
    is found (i.e., a closed walk with voltage 0 mod m and length < g).
    """
    # State: (current_vertex, prev_vertex, accumulated_voltage)
    # Depth is tracked by BFS level
    
    max_len = g - 1
    visited = set()
    queue = deque()
    
    start_state = (start, -1, 0)
    visited.add(start_state)
    queue.append((start, -1, 0, 0))  # (u, prev, vol, depth)
    
    while queue:
        u, prev, vol, depth = queue.popleft()
        
        if depth >= max_len:
            continue
        
        for v in petersen_adj[u]:
            if v == prev:
                continue  # no backtracking
            
            new_vol = (vol + voltages.get((u, v), 0)) % m
            new_state = (v, u, new_vol)
            
            if new_state in visited:
                continue
            
            visited.add(new_state)
            new_depth = depth + 1
            
            if v == start and new_vol == 0 and new_depth >= 2:
                # Found a cycle! Check its length
                # The cycle length is new_depth
                if new_depth < g:
                    return True  # Found short cycle
                else:
                    continue  # Cycle is long enough, don't expand further from this
            
            if new_depth < max_len:
                queue.append((v, u, new_vol, new_depth))
    
    return False


def _find_circulant_v2(k, g, deadline):
    """
    Fast circulant search using algebraic 4-cycle and 6-cycle filters.
    For C_n(1,a,b): 4-cycle iff exists signs s1,s2,s3,s4 in {+1,-1} with
    s1*1 + s2*a + s3*b + s4*c = 0 mod n where {s1*1, s2*a, s3*b, s4*c} are
    not immediate backtracks. Simplified: check all 4-step sums.
    """
    min_n = 512
    max_n = 940  # best known upper bound is 936

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            return None

        # Precompute all 4-step and 6-step sums to filter candidates
        # For 4-cycles: we need ±1 ±a ±b ±c = 0 where c is another generator
        # Actually for C_n(1,a,b) the generators are {1,a,b}, so a 4-cycle
        # is a sequence of 4 steps from {±1,±a,±b} summing to 0 mod n,
        # with no immediate backtracking.
        # 
        # Key algebraic fact: C_n(1,a,b) has a 4-cycle iff
        # any of the following holds mod n:
        #   2*1 = 0 (impossible for n>2)
        #   2*a = 0, 2*b = 0 (i.e., a or b = n/2)
        #   1+a = b, 1+b = a, a+b = 1 (and permutations with signs)
        #   1-a = b, 1-b = a, a-b = 1, etc.
        # More systematically: a 4-cycle exists iff there exist
        # distinct generators g1,g2,g3,g4 (with signs) such that
        # g1+g2+g3+g4 = 0 and the path doesn't backtrack.
        #
        # The cleanest necessary AND sufficient condition:
        # No 4-cycle iff for all distinct (up to sign) pairs from {1,a,b},
        # their sum and difference are not in {±1,±a,±b}.

        half = n // 2
        found = False

        # Try a in a heuristic order: values that are "far" from 1 and n/2
        # a should be roughly n/3 to n/2
        a_start = max(3, n // 4)
        a_end = half - 1

        for a in range(a_start, a_end + 1):
            if time.time() > deadline - 5:
                return None

            # Quick 4-cycle filter for fixed a:
            # Check if 2a ≡ ±1, ±a, ±b for any b -> reject a if 2a ≡ ±1 or 2a ≡ ±a
            two_a = 2 * a % n
            if two_a == 1 or two_a == n - 1 or two_a == a or two_a == n - a:
                continue
            # Also 1+a and 1-a should not be 0 (always true) 
            # and a+1, a-1 should not create issues with b later

            for b in range(a + 1, half):
                if time.time() > deadline - 5:
                    return None

                # Algebraic 4-cycle check: 
                # A 4-cycle exists iff ∃ s1,s2,s3,s4 ∈ {±1} such that
                # s1*1 + s2*a + s3*b + s4*g = 0 mod n where g ∈ {1,a,b}
                # and the walk doesn't immediately backtrack.
                # 
                # Equivalent: check if any of these is 0 mod n:
                # (±1 ±a ±b ±1), (±1 ±a ±b ±a), (±1 ±a ±b ±b)
                # excluding backtracking cases.
                #
                # Simplified: check all 4-element multisets from {±1,±a,±b}
                # that sum to 0 mod n, excluding those with immediate backtracks.
                
                if _has_4cycle(n, a, b):
                    continue
                if _has_6cycle(n, a, b):
                    continue

                # Passed 4 and 6 cycle filters. Do a full girth check.
                if _full_girth_check_circ(n, a, b, g):
                    # Build edges
                    edges = []
                    for i in range(n):
                        for shift in (1, a, b):
                            j = (i + shift) % n
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return None


def _has_4cycle(n, a, b):
    """Check if C_n(1,a,b) has a 4-cycle using algebraic criterion."""
    # All possible 4-step sums from {±1, ±a, ±b}
    # We need s1 + s2 + s3 + s4 = 0 mod n where each si ∈ {±1, ±a, ±b}
    # and the path v0→v1→v2→v3→v0 doesn't have v_{i+1} = v_{i-1}
    # 
    # For a circulant, this reduces to checking if 0 can be written as a sum
    # of 4 elements from G = {1, -1, a, -a, b, -b} with no two consecutive
    # being negatives of each other.
    #
    # Since the graph is vertex-transitive, we can WLOG start at 0.
    # A 4-cycle through 0: 0 → x → y → z → 0 where x,y,z are distinct
    # and x ∈ {±1,±a,±b}, y-x ∈ {±1,±a,±b}, z-y ∈ {±1,±a,±b}, -z ∈ {±1,±a,±b}
    # and x ≠ 0, y ≠ 0, z ≠ 0, y ≠ x (no backtrack), z ≠ y (no backtrack), 0 ≠ z (no backtrack)
    
    G = [1, -1, a, -a, b, -b]
    G_mod = [x % n for x in G]
    
    # Check all combinations of 4 steps
    for i in range(6):
        for j in range(6):
            if G_mod[i] == (-G_mod[j]) % n:
                continue  # immediate backtrack
            for k in range(6):
                if G_mod[j] == (-G_mod[k]) % n:
                    continue
                for l in range(6):
                    if G_mod[k] == (-G_mod[l]) % n:
                        continue
                    # Check if sum is 0
                    s = (G_mod[i] + G_mod[j] + G_mod[k] + G_mod[l]) % n
                    if s == 0:
                        # Also need to verify it's a valid 4-cycle (4 distinct vertices)
                        # v0=0, v1=G[i], v2=G[i]+G[j], v3=G[i]+G[j]+G[k]
                        # Need v0,v1,v2,v3 all distinct
                        v1 = G_mod[i]
                        v2 = (G_mod[i] + G_mod[j]) % n
                        v3 = (G_mod[i] + G_mod[j] + G_mod[k]) % n
                        if v1 != 0 and v2 != 0 and v3 != 0 and v1 != v2 and v2 != v3 and v1 != v3:
                            return True
    return False


def _has_6cycle(n, a, b):
    """Check if C_n(1,a,b) has a 6-cycle."""
    # A 6-cycle: 6 steps from {±1,±a,±b} summing to 0 mod n, no backtracking
    # This is more expensive. Use a necessary condition:
    # If there exist 3 steps s1,s2,s3 from G such that s1+s2+s3 is in G,
    # and the path is valid, then we might have a 6-cycle.
    # 
    # Actually, a 6-cycle through 0: 0→v1→v2→v3→v4→v5→0
    # v1 = g1, v2 = g1+g2, v3 = g1+g2+g3, v4 = v3+g4, v5 = v4+g5, 0 = v5+g6
    # where each gi ∈ G, no consecutive backtracks, all vi distinct and nonzero.
    #
    # For efficiency, use a BFS-based check limited to depth 3 from 0.
    # If any vertex at distance 3 from 0 has a neighbor at distance 3
    # (other than the parent), we have a 6-cycle candidate.
    
    G_mod = [1, n-1, a % n, (n-a) % n, b % n, (n-b) % n]
    
    # BFS to depth 3
    dist = {0: 0}
    parent = {0: None}
    level = [0]
    
    for depth in range(3):
        next_level = []
        for u in level:
            for g in G_mod:
                v = (u + g) % n
                if v == 0 and depth < 2:
                    continue  # would close a cycle too early
                if v not in dist:
                    dist[v] = depth + 1
                    parent[v] = u
                    next_level.append(v)
        level = next_level
    
    # Check for 6-cycles: for each vertex at dist 3, check if it has a neighbor
    # also at dist 3 (not its parent)
    for v in level:  # vertices at distance 3
        for g in G_mod:
            w = (v + g) % n
            if w in dist and dist[w] == 3 and parent[w] != v and parent[v] != w:
                # Potential 6-cycle: need to verify all 6 vertices distinct
                # Path: 0 → p0 → p1 → p2 → v → w → ?
                # Actually we need to trace back
                # For now, return True as a conservative check
                # (false positives are OK, false negatives are not)
                return True
    return False


def _full_girth_check_circ(n, a, b, g):
    """Full non-backtracking girth check for C_n(1,a,b) from vertex 0."""
    # Since the graph is vertex-transitive, checking from 0 suffices.
    # Non-backtracking BFS: state (u, prev)
    G_mod = [1, n-1, a % n, (n-a) % n, b % n, (n-b) % n]
    
    dist = {}
    queue = deque()
    start = (0, -1)
    dist[start] = 0
    queue.append(start)
    
    max_len = g - 1
    
    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_len:
            continue
        for g_step in G_mod:
            v = (u + g_step) % n
            if v == prev:
                continue
            ns = (v, u)
            nd = d + 1
            if nd > max_len:
                continue
            if ns not in dist or nd < dist[ns]:
                dist[ns] = nd
                if v == 0 and u != -1:
                    return False  # found cycle shorter than g
                queue.append(ns)
    return True


def _bipartite_cage(k, g, deadline):
    """
    Construct a bipartite k-regular graph with girth >= g.
    For k=3, g=16, bipartite requires n >= 1022, so this is not competitive
    with the non-bipartite Moore bound of 510. Return None.
    """
    return None


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1

    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


def _verify_girth_fast(n, edges, g):
    """
    Fast girth verification using BFS from each vertex.
    Returns True if girth >= g, False otherwise.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _bfs_short_cycle(src, adj, n, g):
            return False

    return True


def _bfs_short_cycle(src, adj, n, g):
    """
    BFS from src to find if there's a cycle shorter than g.
    Returns True if a short cycle is found.
    """
    dist = [-1] * n
    parent = [-1] * n
    dist[src] = 0
    queue = deque([src])

    max_depth = g // 2

    while queue:
        u = queue.popleft()
        depth = dist[u]

        if depth >= max_depth:
            for v in adj[u]:
                if parent[v] == u or parent[u] == v:
                    continue
                if dist[v] != -1:
                    cycle_len = depth + 1 + dist[v]
                    if cycle_len < g:
                        return True
            continue

        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = depth + 1
                parent[v] = u
                queue.append(v)
            else:
                if parent[v] == u or parent[u] == v:
                    continue
                cycle_len = depth + 1 + dist[v]
                if cycle_len < g:
                    return True

    return False

# EVOLVE-BLOCK-END
