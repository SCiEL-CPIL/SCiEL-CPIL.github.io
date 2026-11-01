"""Voltage-graph lift construction for small 5-regular graph of girth 9."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    # We focus on the task's specific parameters: k=5, g=9.
    deadline = time.time() + 115  # safety margin before 120s kill
    
    if k != 5 or g != 9:
        return _luw_construct(k, g)
    
    # Try Cayley graph of C_2 x C_558 (1116 vertices) with 5 generators
    # Element: (a, b) where a in {0,1}, b in Z_558
    # Represent as integer: a * 558 + b
    N = 1116
    M = 558
    
    # A 5-regular Cayley graph needs a symmetric generating set S of size 5
    # S = S^{-1}, so for each s in S, -s is also in S
    # Since |S| = 5 (odd), at least one element must be self-inverse
    # Self-inverse in C_2 x C_558: (a,b) with 2a=0 mod 2 and 2b=0 mod 558
    # So a can be 0 or 1, b must be 0 or 279
    # Self-inverse elements: (0,0)=identity(skip), (0,279), (1,0), (1,279)
    
    # Strategy: use (1, 0) as one generator (self-inverse, order 2)
    # Then we need 4 more generators that form 2 inverse pairs
    # i.e., s1, -s1, s2, -s2 where s1, s2 are not self-inverse
    
    # For the Cayley graph to be connected, S must generate C_2 x C_558
    # (1,0) gives the C_2 part, so we need s1, s2 to generate C_558 in the second coordinate
    
    # Let's search over choices of s1 = (0, a), s2 = (0, b) 
    # where a, b generate Z_558 (i.e., gcd(a,b,558) = 1)
    # and check girth >= 9
    
    # To verify girth: BFS with non-backtracking constraint from each vertex
    # But 1116 vertices is manageable
    
    # First, let's try a deterministic set of generators
    # We need gcd(a, b, 558) = 1 for connectivity
    
    # 558 = 2 * 3 * 3 * 31
    # Try various (a, b) pairs
    
    best_result = None
    
    # Precompute: for a given generating set S = {(1,0), (0,a), (0,-a), (0,b), (0,-b)}
    # The neighbors of vertex (x, y) are:
    # (x+1, y), (x, y+a), (x, y-a), (x, y+b), (x, y-b)  (mod appropriate)
    
    # For girth check, we do non-backtracking BFS from each vertex
    # State: (vertex, previous_vertex) -> distance
    # Start: for each neighbor v of start, state (v, start) at distance 1
    # A cycle of length L is found when we reach (start, u) where u is a neighbor of start, 
    # and the path length is L-1... actually let me think more carefully.
    
    # Non-backtracking walk: v0, v1, ..., vL where vL = v0, vi != v_{i-2}
    # BFS state: (current, prev) meaning we're at 'current' having come from 'prev'
    # Initialize: for each neighbor n of start: state (n, start) at depth 1
    # Transition: from (cur, prev), go to each neighbor nxt of cur where nxt != prev
    #   New state: (nxt, cur) at depth+1
    # Cycle found: if nxt == start and depth+1 >= 3 (and the walk is non-backtracking)
    #   Actually, we need depth+1 >= g to not be a short cycle
    #   We want to verify NO cycle of length < g exists
    
    def check_girth(a, b, girth_needed):
        """Check if Cayley graph with generators (1,0),(0,a),(0,-a),(0,b),(0,-b) has girth >= girth_needed."""
        # Vertices: 0..1115, vertex = x*558 + y where x in {0,1}, y in 0..557
        # Neighbors of (x,y): (x^1, y), (x, (y+a)%558), (x, (y-a)%558), (x, (y+b)%558), (x, (y-b)%558)
        
        # Non-backtracking BFS from vertex 0
        # State: (cur, prev) -> min depth
        # Use a dict or array: for each (cur, prev) pair, store the depth
        # cur: 0..1115, prev: 0..1115 or -1 for start
        # That's 1116*1116 = ~1.24M states, manageable
        
        # Actually, we can use a 2D array: dist[cur][prev] but that's 1116*1116
        # Let's use a dict for efficiency
        
        def get_neighbors(v):
            x = v // 558
            y = v % 558
            return [
                (x ^ 1) * 558 + y,
                x * 558 + (y + a) % 558,
                x * 558 + (y - a) % 558,
                x * 558 + (y + b) % 558,
                x * 558 + (y - b) % 558
            ]
        
        # Check from vertex 0
        start = 0
        # BFS: state (cur, prev), depth
        # Use a set of visited states to avoid reprocessing
        # But we want the SHORTEST non-backtracking cycle, so BFS gives us that
        
        from collections import deque
        
        # dist[state] where state = (cur, prev)
        # We'll use a dict
        visited = set()
        queue = deque()
        
        # Initialize: neighbors of start
        for n in get_neighbors(start):
            state = (n, start)
            if state not in visited:
                visited.add(state)
                queue.append((state, 1))
        
        while queue:
            (cur, prev), depth = queue.popleft()
            
            if depth >= girth_needed:
                # All cycles found so far have length >= depth >= girth_needed
                # But we need to check if there's a cycle of length < girth_needed
                # Since BFS processes in order of depth, if we've reached depth >= girth_needed
                # without finding a cycle, all shorter cycles are absent
                # Actually no: a cycle of length L means we return to start at depth L
                # So we need to check if we ever reach a state where cur == start
                # with depth < girth_needed
                pass
            
            # Check if cur == start (cycle found)
            if cur == start:
                # Cycle of length depth (the walk has depth edges from start back to start)
                # Wait: we started at start, went to n (depth 1), now we're back at start (depth+1?)
                # Let me re-examine: state (n, start) at depth 1 means we're at n, came from start, path length 1
                # If from (n, start) we go to start: state (start, n) at depth 2
                # That's a cycle of length 2, but that's backtracking! n->start->n would be backtrack
                # Actually (start, n) means we're at start, came from n. The walk is start->n->start, length 2
                # But this IS backtracking (we went start->n->start, reversing the edge)
                # In non-backtracking, we require v_{i+1} != v_{i-1}
                # For the walk start->n->start: v0=start, v1=n, v2=start
                # v2 should != v0? No, v_{i+1} != v_{i-1} means for i=1: v2 != v0
                # But v2 = start = v0, so this IS backtracking. So we should NOT count this.
                # 
                # Wait, I think the non-backtracking condition is: in the walk v0,v1,...,vL,
                # we require v_{i+1} != v_{i-1} for i=1,...,L-1
                # For L=2: v2 != v0. If v2=v0 (cycle), then it's backtracking.
                # So cycles of length 2 are always backtracking and not counted.
                # 
                # In our BFS: state (cur, prev) at depth d means we have a walk of length d from start to cur
                # where the last step was from prev to cur, and the walk is non-backtracking.
                # If cur == start, we have a closed non-backtracking walk of length d.
                # But we need d >= 3 for it to be non-backtracking (d=2 would be backtracking).
                # Actually, can d=2 happen? state (start, n) at depth 2: walk is start->n->start
                # This requires that from state (n, start) at depth 1, we can go to start.
                # The transition from (cur=n, prev=start): go to neighbors of n except start.
                # So we CANNOT go to start from (n, start) because start is the prev!
                # So state (start, n) at depth 2 is never reached. Good.
                #
                # So the first time we reach cur==start, it's at depth >= 3, and it's a valid non-backtracking cycle.
                
                return depth  # shortest cycle length from start
        
            # Expand
            for nxt in get_neighbors(cur):
                if nxt == prev:
                    continue  # non-backtracking
                state = (nxt, cur)
                if state not in visited:
                    visited.add(state)
                    queue.append((state, depth + 1))
        
        # If we exhausted BFS without reaching start, no cycle exists (disconnected?)
        # But the graph should be connected if generators generate the group
        return float('inf')
    
    # Search over (a, b) pairs
    # We need gcd(a, b, 558) = 1 for connectivity
    # Let's try systematic search
    
    # First, try some structured choices
    # 558 = 2 * 3^2 * 31
    
    # Try (a, b) = (1, 2), (1, 3), (1, 31), (2, 3), etc.
    # But we also need to avoid short cycles
    
    # For efficiency, let's first check connectivity (gcd condition) then girth
    
    candidates = []
    for a in range(1, 558):
        if time.time() > deadline - 20:
            break
        for b in range(a + 1, 558):
            # Check gcd(a, b, 558) == 1
            import math
            if math.gcd(math.gcd(a, b), 558) != 1:
                continue
            # Also need a != b, a != -b mod 558, etc. for distinct generators
            # (0,a), (0,-a), (0,b), (0,-b) should be 4 distinct non-identity elements
            # -a mod 558 = 558-a (if a != 0)
            if a == b or a == 558 - b or a == 558 - a:  # a == 279
                continue
            # Also (1,0) should not equal any of the (0,*) generators - they're in different cosets, so OK
            candidates.append((a, b))
    
    # Try candidates in order of some heuristic
    # Smaller a, b might give more structure
    random.seed(42)
    random.shuffle(candidates)
    
    # Limit the number of candidates to try given time constraints
    max_candidates = 500
    
    for a, b in candidates[:max_candidates]:
        if time.time() > deadline - 15:
            break
        
        girth = check_girth(a, b, 9)
        
        if girth >= 9:
            # Build the graph
            edges = []
            for v in range(N):
                x = v // 558
                y = v % 558
                neighbors = [
                    (x ^ 1) * 558 + y,
                    x * 558 + (y + a) % 558,
                    x * 558 + (y - a) % 558,
                    x * 558 + (y + b) % 558,
                    x * 558 + (y - b) % 558
                ]
                for nb in neighbors:
                    if v < nb:
                        edges.append((v, nb))
            
            best_result = (N, edges)
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: try the voltage graph approach (original)
    result = _voltage_graph_construct(5, 9, deadline)
    if result is not None:
        return result
    
    # Final fallback: L-U-W construction
    return _luw_construct(k, g)


def _voltage_graph_construct(k, g, deadline):
    """Voltage graph over K_6 - kept from original as fallback."""
    base_n = 6
    base_edges = []
    edge_index = {}
    for u in range(base_n):
        for v in range(u + 1, base_n):
            edge_index[(u, v)] = len(base_edges)
            base_edges.append((u, v))
    n_base_edges = len(base_edges)
    
    def gen_nonbacktracking_closed_walks(length):
        def walk_gen(pos, cur, prv, path):
            if pos == length:
                if cur != 0 and prv != 0:
                    yield list(path) + [0]
                return
            if pos == length - 1:
                if cur != 0 and prv != 0:
                    yield list(path) + [0]
                return
            for nv in range(base_n):
                if nv == cur or nv == prv:
                    continue
                path.append(nv)
                yield from walk_gen(pos + 1, nv, cur, path)
                path.pop()
        for v1 in range(1, base_n):
            yield from walk_gen(1, v1, 0, [0, v1])
    
    constraints_by_length = {}
    for L in range(3, 9):
        walks = list(gen_nonbacktracking_closed_walks(L))
        constraint_set = set()
        for walk in walks:
            terms = []
            for i in range(L):
                u, v = walk[i], walk[i + 1]
                eu, ev = min(u, v), max(u, v)
                eidx = edge_index[(eu, ev)]
                sign = 1 if u < v else -1
                terms.append((eidx, sign))
            constraint_set.add(tuple(terms))
        constraints_by_length[L] = list(constraint_set)
    
    for m in range(2, 200):
        if time.time() > deadline - 5:
            break
        n_vertices = base_n * m
        if n_vertices >= 1116:
            continue
        
        for attempt in range(2000):
            if time.time() > deadline - 3:
                break
            voltages = [random.randint(0, m - 1) for _ in range(n_base_edges)]
            valid = True
            for L in range(3, 9):
                for terms in constraints_by_length[L]:
                    s = 0
                    for eidx, sign in terms:
                        s += sign * voltages[eidx]
                    if s % m == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                graph_edges = []
                for u in range(base_n):
                    for v in range(u + 1, base_n):
                        eidx = edge_index[(u, v)]
                        for i in range(m):
                            vid1 = u * m + i
                            j = (i + voltages[eidx]) % m
                            vid2 = v * m + j
                            if vid1 < vid2:
                                graph_edges.append((vid1, vid2))
                return (n_vertices, graph_edges)
    
    return None


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction from the seed program."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

# EVOLVE-BLOCK-END
