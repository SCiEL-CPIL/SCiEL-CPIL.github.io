import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    if k == 3 and g == 16:
        # Try Cayley graph constructions with 3 involutions
        # Then fall back to optimized LUW

        # Attempt 1: Cayley graph on (Z_2)^m with carefully chosen involutions
        # A Cayley graph on an abelian group with 3 involutions gives a bipartite
        # 3-regular graph. Girth can be high if generators are "independent".
        # For (Z_2)^m, the graph is a subgraph of a hypercube-like structure.
        # Actually, for abelian groups, the Cayley graph with involutions is bipartite
        # and girth is related to the minimal relation among generators.
        
        # Try different abelian groups: Z_{p1} x Z_{p2} x ... 
        # For 3 involutions in an abelian group, each generator has order 2.
        # So the group must be a 2-group or have elements of order 2.
        # Let's try (Z_2)^m for various m, but that gives at most 2^m vertices
        # and the graph is a partial cube. Girth of such graphs is limited.
        
        # Better: use non-abelian groups. Try dihedral groups D_n or other small groups.
        # Actually, for high girth, we need large groups.
        
        # Strategy: Use Z_p x Z_p (cyclic of order p^2) with 3 involutions.
        # But Z_p (p odd) has no elements of order 2. So use Z_2 x Z_p x Z_p?
        # Elements of order 2 in Z_2 x Z_p x Z_p: (1, 0, 0) only.
        # That gives only 1 involution, not 3.
        
        # Alternative: Use the group (Z_2)^3 x Z_m for some m.
        # Involutions are elements of order dividing 2, so (a, 0) where a in (Z_2)^3.
        # There are 7 non-identity involutions. Pick 3 of them.
        # The Cayley graph has 8*m vertices.
        # Girth: need no word of length < 16 in the generators equals identity.
        
        # Let's try (Z_2)^3 x Z_m with generators (e1, a), (e2, b), (e3, c)
        # where e1,e2,e3 are standard basis of (Z_2)^3 and a,b,c in Z_m.
        # A word w of length L maps to (product of e's in (Z_2)^3, sum of corresponding a/b/c in Z_m).
        # For the word to be identity, we need the (Z_2)^3 part to be 0 (even number of each e_i)
        # AND the Z_m part to be 0.
        
        # To maximize girth, choose a, b, c so that no short combination sums to 0 mod m.
        # This is like a coding theory problem.
        
        best_result = None
        best_n = float('inf')
        
        # Try m from small to large, with different (a,b,c) choices
        # For girth 16, we need m large enough.
        # The shortest non-trivial word in e1,e2,e3 with even counts is length 2 (e.g., e1*e1).
        # But that maps to (0, 2a) in Z_m. So we need 2a != 0, 2b != 0, 2c != 0 mod m.
        # Next: e1*e2*e1*e2 maps to (0, 0) in (Z_2)^3 and (a+b+a+b) = 2(a+b) in Z_m.
        # So we need 2(a+b) != 0 mod m.
        # In general, we need that for all words of length < 16 with even count of each generator,
        # the sum is not 0 mod m.
        
        # This is equivalent to: the set {sum of coefficients * values} for short words avoids 0.
        
        # Let's parameterize: generators are (e1, a), (e2, b), (e3, c) in (Z_2)^3 x Z_m.
        # A word is a sequence of {1,2,3}. The (Z_2)^3 component is the XOR of the generators used.
        # The Z_m component is the sum of a, b, or c for each use.
        
        # For the word to reduce to identity, we need each generator used an even number of times
        # (so XOR is 0), and the total sum is 0 mod m.
        
        # The shortest such word: use each generator 0 or 2+ times, total length < 16.
        # Minimum non-trivial: 2 uses of one generator: length 2, sum = 2*value.
        # Next: 2 of one, 2 of another: length 4, sum = 2a + 2b or similar.
        
        # To avoid short cycles, we need:
        # 2a, 2b, 2c, 2(a+b), 2(a+c), 2(b+c), 2(a+b+c), etc. != 0 mod m
        # More generally, for any even-length word < 16, the weighted sum != 0.
        
        # Let's try m as a prime ~ 200-500 and search for good (a,b,c).
        
        for m in [199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523]:
            if time.time() > deadline - 15:
                break
            
            n = 8 * m  # |Z_2^3| * |Z_m|
            if n >= best_n:
                continue
            
            # Try a few random (a, b, c)
            for trial in range(50):
                if time.time() > deadline - 10:
                    break
                a = random.randint(1, m - 1)
                b = random.randint(1, m - 1)
                c = random.randint(1, m - 1)
                
                # Quick check: 2a, 2b, 2c, 2(a+b), 2(a+c), 2(b+c) != 0 mod m
                # Since m is odd prime, 2x = 0 iff x = 0. So we need a,b,c,a+b,a+c,b+c != 0 mod m.
                if a == 0 or b == 0 or c == 0:
                    continue
                if (a + b) % m == 0 or (a + c) % m == 0 or (b + c) % m == 0:
                    continue
                
                # More thorough check via BFS on the Cayley graph
                # Vertices: (x1, x2, x3, t) where xi in {0,1}, t in Z_m
                # Generators: g1 = (1,0,0,a), g2 = (0,1,0,b), g3 = (0,0,1,c)
                # Since generators are involutions, g_i^{-1} = g_i.
                
                # BFS from identity (0,0,0,0) to find shortest cycle.
                # In a Cayley graph, girth = shortest non-trivial word = identity.
                # BFS: state is (x1,x2,x3,t). Start at (0,0,0,0).
                # We want the shortest word that returns to start without backtracking.
                # In a Cayley graph with involutions, non-backtracking means we don't immediately
                # apply the same generator twice consecutively (since g_i * g_i = e).
                
                # Actually, for girth computation, we can do BFS on the directed graph of
                # (vertex, previous_generator) pairs, or just BFS and track the shortest
                # cycle through the start node.
                
                # Simpler: BFS from (0,0,0,0) and find the shortest path back to (0,0,0,0)
                # that doesn't use the trivial 2-step (g_i then g_i).
                # Since all generators are involutions, g_i * g_i = e gives a 2-cycle,
                # but that's not a valid cycle in a simple graph (it's just traversing an edge back).
                # In a simple graph, the girth is the shortest cycle of length >= 3.
                # But our graph is bipartite (abelian group with involutions => bipartite),
                # so all cycles have even length. Shortest possible is 4.
                
                # BFS: dist[(x1,x2,x3,t)] = shortest distance from start.
                # If we find start again at distance d > 0, that's a cycle of length d.
                # But we need to avoid the trivial back-and-forth.
                
                # Let's do BFS properly:
                # State: (x1, x2, x3, t, prev_gen) where prev_gen in {-1, 1, 2, 3}
                # -1 means start.
                # From (x1,x2,x3,t,prev), we can go to neighbors using gen != prev (non-backtracking).
                # Wait, non-backtracking in a Cayley graph: from v, using gen i to reach v',
                # we can't immediately use gen i again to go back to v (since g_i is involution).
                # So we track the last generator used.
                
                # BFS with state (x1,x2,x3,t, last_gen):
                # Start: (0,0,0,0, -1)
                # For each state, try all 3 generators except the last one used.
                # If we reach (0,0,0,0) with last_gen != -1, we found a cycle of length = dist.
                
                # This could be expensive for large m. Let's limit the BFS depth.
                # We only need to check up to depth 15 (for girth >= 16).
                
                found_girth_16 = True
                # BFS up to depth 15
                # State: (x1, x2, x3, t, last_gen)
                # Use a dict for visited states at each depth.
                
                # For efficiency, note that x1,x2,x3 are in {0,1} and t in Z_m.
                # last_gen in {0,1,2,3} (0 = none/start).
                # Total states: 8 * m * 4 = 32m. For m=500, that's 16000 states.
                # BFS up to depth 15: branching factor 2 (can't use last gen), so ~2^15 = 32768 paths.
                # But many merge. Should be manageable.
                
                from collections import defaultdict
                
                # dist[state] = depth
                # state = (x1, x2, x3, t, last_gen)
                start_state = (0, 0, 0, 0, 0)  # last_gen=0 means none
                visited = {start_state: 0}
                queue = deque([(0, 0, 0, 0, 0, 0)])  # x1,x2,x3,t,last_gen,depth
                
                while queue:
                    x1, x2, x3, t, last_gen, depth = queue.popleft()
                    if depth >= 15:
                        continue
                    # Try each generator 1,2,3 except last_gen
                    for gen in range(1, 4):
                        if gen == last_gen:
                            continue
                        # Apply generator
                        nx1, nx2, nx3, nt = x1, x2, x3, t
                        if gen == 1:
                            nx1 = 1 - nx1
                            nt = (nt + a) % m
                        elif gen == 2:
                            nx2 = 1 - nx2
                            nt = (nt + b) % m
                        elif gen == 3:
                            nx3 = 1 - nx3
                            nt = (nt + c) % m
                        new_state = (nx1, nx2, nx3, nt, gen)
                        if new_state == start_state:
                            # Found a cycle of length depth + 1
                            if depth + 1 < 16:
                                found_girth_16 = False
                                break
                        if new_state not in visited:
                            visited[new_state] = depth + 1
                            queue.append((nx1, nx2, nx3, nt, gen, depth + 1))
                    if not found_girth_16:
                        break
                
                if found_girth_16:
                    # Verify more carefully and build the edge list
                    # Build edges: for each vertex, connect to 3 neighbors via generators
                    edges = []
                    for x1 in range(2):
                        for x2 in range(2):
                            for x3 in range(2):
                                for t in range(m):
                                    v = ((x1 * 2 + x2) * 2 + x3) * m + t
                                    # Generator 1
                                    nx1 = 1 - x1
                                    nt = (t + a) % m
                                    u = ((nx1 * 2 + x2) * 2 + x3) * m + nt
                                    if v < u:
                                        edges.append((v, u))
                                    # Generator 2
                                    nx2 = 1 - x2
                                    nt = (t + b) % m
                                    u = ((x1 * 2 + nx2) * 2 + x3) * m + nt
                                    if v < u:
                                        edges.append((v, u))
                                    # Generator 3
                                    nx3 = 1 - x3
                                    nt = (t + c) % m
                                    u = ((x1 * 2 + x2) * 2 + nx3) * m + nt
                                    if v < u:
                                        edges.append((v, u))
                    
                    # Verify girth with a proper check
                    if _verify_girth(edges, n, g):
                        if n < best_n:
                            best_n = n
                            best_result = (n, edges)
                        break  # Found one for this m, try smaller m? No, we want smallest n.
                    # If verification fails, continue searching
            # End trial loop
        # End m loop
        
        if best_result is not None:
            return best_result
    
    # Fallback: optimized LUW
    return _luw_construct(k, g)


def _verify_girth(edges, n, g):
    """Verify that the graph has girth at least g using non-backtracking BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return False
    
    # Non-backtracking BFS from each vertex
    # State: (vertex, prev_vertex)
    # From (u, prev), we can go to any neighbor v of u where v != prev.
    # Start: (i, -1) for each i.
    # If we find a state (i, prev) where prev != -1, we have a cycle of length = dist.
    
    # For efficiency, we only need to check up to depth g-1.
    # If no cycle of length < g is found from any start, girth >= g.
    
    for i in range(n):
        # BFS from (i, -1)
        dist = {}
        queue = deque()
        queue.append((i, -1, 0))
        dist[(i, -1)] = 0
        
        while queue:
            u, prev, d = queue.popleft()
            if d >= g - 1:
                continue
            for v in adj[u]:
                if v == prev:
                    continue
                new_state = (v, u)
                if new_state == (i, -1):
                    # Cycle of length d+1
                    if d + 1 < g:
                        return False
                elif new_state not in dist:
                    dist[new_state] = d + 1
                    queue.append((v, u, d + 1))
    
    return True


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
