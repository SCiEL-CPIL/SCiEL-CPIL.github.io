"""Construct k-regular graph of girth g using random Cayley graphs on non-abelian groups."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Build a Cayley graph on a non-abelian group using k involutory generators.
    Verify girth via non-backtracking BFS from identity. Try multiple random generator
    sets until one satisfies the girth constraint.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave margin for evaluation overhead
    
    # Moore bound for reference
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    # We'll search over groups of increasing size
    # For k=5, g=9, we need groups where 5 involutions can have girth >= 9
    # The group must be large enough: n >= moore_bound(k, g) = 1 + 5*(1+4+16+64+256) = 1 + 5*341 = 1706? 
    # Wait: for odd g=9, (g-3)/2 = 3, so sum i=0 to 3 of (k-1)^i = 1+4+16+64 = 85
    # n >= 1 + 5*85 = 426
    
    # Let's try groups of various sizes
    # Approach: Use the group G = Z_2^m (elementary abelian 2-group) as a starting point,
    # but that's abelian. For non-abelian, use products or other groups.
    
    # Actually, for Cayley graphs with involutory generators, the group structure matters.
    # Let's try:
    # 1. Small non-abelian groups: S_3, S_4, D_n, etc.
    # 2. Products of small groups
    # 3. For larger groups, use random permutations
    
    # Key insight: For a Cayley graph with generators S (all involutions, |S|=k),
    # a non-backtracking walk of length L from identity: s_1, s_2, ..., s_L
    # where s_i * s_{i+1} != e (i.e., s_i != s_{i+1} since involutions)
    # A cycle exists if some product equals identity.
    
    # For girth >= g, we need: no non-backtracking product of length < g equals identity.
    
    # Let's try to work with groups represented as permutation groups or via multiplication tables.
    
    # Strategy: Try groups of size n = m^2 or similar, using structured generators.
    
    # For k=5, g=9: Try groups of size around 426-1000
    # Use G = (Z_p)^2 with a twist, or use specific non-abelian groups.
    
    # Let's try a systematic approach:
    # Generate candidate groups and test random generator sets.
    
    # Approach 1: Use the group of affine transformations or similar
    
    # Let's try: G = Z_m x Z_m with a non-standard multiplication (semi-direct product)
    # Or simpler: try random groups of size n and see if random involutions work.
    
    # For efficiency, let's use a representation where we can compute products quickly.
    
    # Let's try: G = (Z/q)^2 with generators being specific involutions.
    # In an abelian group, involutions are elements of order 2.
    # In (Z/q)^2, elements of order 2 exist only if q is even.
    
    # Better approach: Use a non-abelian group like a semidirect product.
    
    # Let's try: G = Z_m ⋊ Z_2 where the action is multiplication by -1 (dihedral-like)
    # Elements: (a, 0) and (a, 1) for a in Z_m
    # Multiplication: (a, i) * (b, j) = (a + (-1)^i * b, i+j mod 2)
    # Involutions: (a, 0) is involution iff 2a = 0 mod m, i.e., a = 0 or a = m/2 (if m even)
    # (a, 1) is always an involution: (a,1)*(a,1) = (a + (-1)*a, 0) = (0, 0)
    
    # So in this group of size 2m, we have:
    # - (0,0): identity
    # - (m/2, 0): involution (if m even)
    # - All (a, 1) for a in Z_m: involutions (m of them)
    # Total involutions: 1 + m (if m even) or 1 + m (the (0,0) is identity, not a generator)
    # Non-identity involutions: (m/2, 0) if m even, plus all m elements (a,1)
    
    # For k=5, we need 5 involutory generators. Let's pick 5 from the m+1 available.
    
    # Let's try this approach with various m values.
    
    def try_semidirect(m, num_trials=200, seed=None):
        """Try to find 5 involutions in Z_m ⋊ Z_2 with girth >= g."""
        if m < 2:
            return None
        
        # Group elements: (a, i) where a in 0..m-1, i in {0,1}
        # Index: a*2 + i
        n = 2 * m
        
        # Multiplication: (a, i) * (b, j) = (a + ((-1)**i) * b mod m, (i+j) mod 2)
        def mul(idx1, idx2):
            a1, i1 = idx1 // 2, idx1 % 2
            a2, i2 = idx2 // 2, idx2 % 2
            a = (a1 + ((-1)**i1) * a2) % m
            i = (i1 + i2) % 2
            return a * 2 + i
        
        # Involutions (non-identity):
        # (a, 1) for all a: always involutions
        # (a, 0) is involution iff 2*a % m == 0 and a != 0
        involutions = []
        for a in range(m):
            involutions.append(a * 2 + 1)  # (a, 1)
        # Check (a, 0) for a > 0
        for a in range(1, m):
            if (2 * a) % m == 0:
                involutions.append(a * 2)  # (a, 0)
        
        if len(involutions) < k:
            return None
        
        # Try random selections of k involutions
        rng = random.Random(seed)
        for trial in range(num_trials):
            if time.time() - start_time > time_limit:
                return None
            
            # Pick k random involutions (avoiding identity)
            generators = rng.sample(involutions, k)
            
            # Check for short cycles via non-backtracking BFS from identity
            # A non-backtracking walk: s_1, s_2, ..., s_L where s_i != s_{i+1}
            # (since generators are involutions, s_i * s_{i+1} = e iff s_i = s_{i+1})
            # Cycle of length L exists if product s_1 * s_2 * ... * s_L = e
            # and s_1 != s_2, s_2 != s_3, ..., s_{L-1} != s_L, and s_L != s_1 (for closing)
            
            # Actually, for a cycle in the Cayley graph: v_0, v_1, ..., v_L = v_0
            # where v_{i+1} = v_i * s_{i+1}
            # Non-backtracking: s_{i+1} != s_i^{-1} = s_i (since involutions)
            # So s_i != s_{i+1} for all i (mod L)
            
            # BFS on the state space (current_group_element, last_generator_used)
            # State: (g, last_gen_index) where g is group element, last_gen is which gen was used to get here
            # Start: (e, -1) - at identity, no previous generator
            # We want to find if any state (e, last_gen) is reachable with path length < g
            # where last_gen != -1 (meaning we came back to identity via some generator)
            
            # Actually, simpler: do BFS tracking (group_element, previous_generator)
            # A cycle of length L < g exists if we can reach (e, j) at distance L
            # where j is the generator used to arrive at e.
            
            gen_set = set(generators)
            
            # BFS: state = (group_element, last_gen_idx)
            # last_gen_idx: index in generators list, or -1 for start
            # Distance = length of walk
            # We check if we can reach identity with a non-trivial path of length < g
            
            visited = {}  # (group_elem, last_gen_idx) -> distance
            queue = deque()
            queue.append((0, -1, 0))  # (group_elem, last_gen, distance)
            visited[(0, -1)] = 0
            
            found_short_cycle = False
            while queue:
                gelem, last_gen, dist = queue.popleft()
                
                if dist >= g - 1:
                    # We've explored up to depth g-1; if we haven't found a cycle, girth >= g
                    # Actually, we need to check: if we reach identity at distance L < g with last_gen != -1
                    break
                
                for gi, gen in enumerate(generators):
                    if gi == last_gen:
                        continue  # non-backtracking
                    
                    new_gelem = mul(gelem, gen)
                    new_state = (new_gelem, gi)
                    
                    if new_state in visited:
                        continue
                    
                    new_dist = dist + 1
                    visited[new_state] = new_dist
                    
                    if new_gelem == 0 and gi != -1:
                        # We've returned to identity via generator gi
                        # This forms a cycle of length new_dist
                        if new_dist < g:
                            found_short_cycle = True
                            break
                        else:
                            # Cycle of length >= g, don't expand further from here
                            # Actually, we should still track but not count as short cycle
                            pass
                    
                    if new_dist < g - 1:
                        queue.append((new_gelem, gi, new_dist))
                
                if found_short_cycle:
                    break
            
            if not found_short_cycle:
                # Verify more carefully: check all paths of length < g
                # Let's do a complete check
                # Re-do BFS more carefully
                
                # Check all non-backtracking walks of length 1 to g-1
                # A walk of length L is a sequence s_1, ..., s_L with s_i != s_{i+1}
                # It forms a cycle if product = e and s_L != s_1 (for L >= 2)
                # For L=1: s_1 = e, but generators are non-identity, so no 1-cycles
                # For L=2: s_1 * s_2 = e, s_1 != s_2. Since involutions, s_1*s_2=e means s_1=s_2, contradiction. So no 2-cycles.
                # For L=3: s_1*s_2*s_3 = e, s_1!=s_2, s_2!=s_3, s_3!=s_1
                # ...
                
                # The BFS above should catch this. Let's trust it but do a second verification.
                
                # Build the graph and verify girth properly
                # Create adjacency: vertex v connects to v*gen for each gen
                adj = [[] for _ in range(n)]
                for v in range(n):
                    for gen in generators:
                        w = mul(v, gen)
                        if w > v:  # avoid duplicates
                            adj[v].append(w)
                            adj[w].append(v)
                
                # Verify degree
                for v in range(n):
                    if len(adj[v]) != k:
                        found_short_cycle = True
                        break
                if found_short_cycle:
                    continue
                
                # Verify no short cycles using BFS from each vertex (or just identity for Cayley)
                # For Cayley graphs, girth can be checked from identity
                # Do BFS on (vertex, prev_vertex) states to detect non-backtracking cycles
                
                # Actually, for a Cayley graph, we can check girth by looking at
                # non-backtracking walks from identity.
                
                # Let's do a thorough check: BFS on (current, previous) pairs
                # State: (current_vertex, previous_vertex) where previous_vertex is the vertex we came from
                # Start states: (gen, 0) for each generator gen, distance 1
                # We want to see if we can reach (0, gen) for any gen at distance < g
                
                visited2 = set()
                queue2 = deque()
                for gen in generators:
                    queue2.append((gen, 0, 1))
                    visited2.add((gen, 0))
                
                short_cycle = False
                while queue2:
                    cur, prev, dist = queue2.popleft()
                    
                    if cur == 0 and dist >= 2:
                        # Returned to identity
                        if dist < g:
                            short_cycle = True
                            break
                        continue  # Don't expand from identity (would create backtracking)
                    
                    if dist >= g - 1:
                        continue
                    
                    for gen in generators:
                        nxt = mul(cur, gen)
                        if nxt == prev:
                            continue  # non-backtracking
                        state = (nxt, cur)
                        if state not in visited2:
                            visited2.add(state)
                            queue2.append((nxt, cur, dist + 1))
                
                if not short_cycle:
                    # Success! Build the edge list
                    edges = []
                    for v in range(n):
                        for gen in generators:
                            w = mul(v, gen)
                            if v < w:
                                edges.append((v, w))
                    return n, edges
        
        return None
    
    # Try various group sizes
    # For k=5, g=9, Moore bound is 426
    # Try m values such that 2m >= 426, so m >= 213
    
    # Let's try a range of m values with more trials for smaller m
    base_seed = 42
    found = None
    
    # Try increasing m values
    for m in range(213, 600):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Ensure m is even for extra involutions
        if m % 2 != 0:
            m += 1
        
        # More trials for smaller groups
        num_trials = max(20, 200 - (m - 213) // 2)
        result = try_semidirect(m, num_trials=num_trials, seed=base_seed + m)
        if result is not None:
            found = result
            break
    
    if found is not None:
        return found
    
    # Fallback: try a simpler approach with a known good construction
    # Use the L-U-W construction as ultimate fallback
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
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
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
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
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
