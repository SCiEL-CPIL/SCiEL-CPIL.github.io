"""Construct k-regular graph of girth g using random Cayley graphs on (Z/2Z)^d."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque
from itertools import permutations

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try random Cayley graphs on (Z/2Z)^d with k involutory generators.
    Fall back to L-U-W construction if no valid graph is found in time.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    time_budget = 135.0  # Leave 15s margin for verification within 150s limit
    
    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction as fallback
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
    
    # Verify girth using non-backtracking BFS from identity
    def verify_girth_cayley(gens, g, d):
        """Check if Cayley graph on (Z/2Z)^d with generators gens has girth >= g.
        
        A cycle of length L exists iff there's a non-backtracking closed walk of
        length L from 0 back to 0. We detect any closed non-backtracking walk
        of length < g.
        
        For even L: need a non-backtracking walk 0 -> v1 -> ... -> v_{L-1} -> 0
        where consecutive edges use different generators.
        For odd L: same condition.
        
        Key insight: a non-backtracking walk is determined by the sequence of
        generators used. A closed walk of length L means g_1 ^ g_2 ^ ... ^ g_L = 0
        with g_i != g_{i-1} for all i (and g_L != g_1 for the cycle to be non-backtracking
        at the closing point, but actually the non-backtracking condition is only on
        consecutive edges in the walk, so g_i != g_{i-1} for i=2..L, and also
        g_1 != g_L for the walk to be a proper cycle... actually no: non-backtracking
        means at each internal vertex you don't go back to where you came from.
        So for walk 0->v1->...->v_{L-1}->0, at v1 we need g2 != g1, at v2 we need
        g3 != g2, ..., at v_{L-1} we need g_L != g_{L-1}. The condition g_1 != g_L
        is NOT required for non-backtracking (it would be required for the walk to
        not be a "backtracking" at the start/end, but that's not part of the standard
        non-backtracking definition for cycles).
        
        Actually, let me reconsider. A non-backtracking walk of length L from 0 to 0
        is a sequence v_0=0, v_1, ..., v_L=0 where v_i is adjacent to v_{i-1} and
        v_{i+1} != v_{i-1} for 1 <= i <= L-1. In a Cayley graph, v_i = v_{i-1} ^ g_i
        where g_i is the generator used. The condition v_{i+1} != v_{i-1} means
        v_{i-1} ^ g_i ^ g_{i+1} != v_{i-1}, i.e., g_i ^ g_{i+1} != 0, i.e., g_i != g_{i+1}.
        So we need g_i != g_{i+1} for i = 1, ..., L-1.
        
        The walk is closed if g_1 ^ g_2 ^ ... ^ g_L = 0.
        
        So we need: find a sequence (g_1, ..., g_L) with g_i in {1,...,k}, 
        g_i != g_{i+1} for i=1..L-1, and g_1^...^g_L = 0, with L < g.
        
        For the BFS approach:
        - State: (current_vertex, last_generator_index)
        - Start states: (gens[i], i) for each i, at depth 1
        - Transition: from (v, i) at depth d, go to (v ^ gens[j], j) for j != i, at depth d+1
        - Goal: reach vertex 0 at any depth d where 2 <= d < g
        
        But we need to be careful: we start at depth 1 (one edge from 0). 
        A closed walk of length L has L edges. The BFS starts after 1 edge.
        So we're looking for depth d where d < g and d >= 2 (at least 2 edges for a cycle).
        
        Wait, actually the minimum cycle length in a simple graph is 3 (triangle).
        For a non-backtracking closed walk, the minimum length is 3 (since length 2
        would be going out and back, which is backtracking).
        
        So we check: is there a state (0, j) reached at depth d where 3 <= d < g?
        (Actually d >= 2 would suffice since d=2 means g1^g2=0 with g1!=g2, which
        implies g1=g2, contradiction. So d=2 is impossible. d=3 is the minimum.)
        
        Let me just check d < g and d >= 2 to be safe.
        """
        k = len(gens)
        if k == 0:
            return True
        if g <= 2:
            return False  # Any edge creates a 2-cycle (but graph is simple, so no 2-cycles)
            # Actually in a simple graph, minimum cycle is 3, so girth >= 3 always.
            # But the problem says "no cycle shorter than g", so if g <= 3, any simple
            # graph works. Let's handle this edge case.
        
        # For g <= 3, any simple graph has girth >= 3, so return True
        if g <= 3:
            return True
        
        min_depth = {}
        queue = deque()
        
        for i in range(k):
            v = gens[i]
            state = (v, i)
            min_depth[state] = 1
            queue.append((v, i, 1))
        
        while queue:
            v, last_gen, depth = queue.popleft()
            
            if depth >= g - 1:
                continue
            
            for gi in range(k):
                if gi == last_gen:
                    continue
                new_v = v ^ gens[gi]
                new_state = (new_v, gi)
                new_depth = depth + 1
                
                if new_state not in min_depth:
                    min_depth[new_state] = new_depth
                    if new_v == 0 and new_depth < g and new_depth >= 2:
                        return False  # Found a cycle shorter than g
                    queue.append((new_v, gi, new_depth))
        
        return True
    
    def quick_reject(gens, g):
        """Fast pre-filter: reject generator sets that obviously have short cycles.
        
        1. Check for 3-cycles: g_i ^ g_j ^ g_k = 0 with all distinct, and the
           non-backtracking condition is automatically satisfied for length 3
           (g_i != g_j, g_j != g_k, and we don't need g_k != g_i for non-backtracking
           at the closing point... wait, for a 3-cycle 0->a->b->0, the non-backtracking
           condition at a is g_2 != g_1, at b is g_3 != g_2. We don't require g_1 != g_3.
           But if g_1 = g_3, then g_1^g_2^g_1 = g_2 = 0, which is impossible since
           generators are non-zero. So for 3-cycles, all three generators must be distinct
           and their XOR is 0.
        
        2. Check for 4-cycles: g_i ^ g_j ^ g_k ^ g_l = 0 with g_i != g_j, g_j != g_k,
           g_k != g_l. The sequence can have repeats as long as consecutive are different.
           For example, g_i, g_j, g_i, g_j would give g_i^g_j^g_i^g_j = 0, but this
           corresponds to walk 0 -> g_i -> g_i^g_j -> g_j -> 0, which is a 4-cycle
           (if g_i != g_j and g_i^g_j != 0 and g_i^g_j != g_i and g_i^g_j != g_j,
           which is true if g_i != g_j and both non-zero).
           Wait, but this walk uses generators g_i, g_j, g_i, g_j. The non-backtracking
           condition: at step 2 (vertex g_i), next gen is g_j != g_i. At step 3 (vertex
           g_i^g_j), next gen is g_i != g_j. At step 4 (vertex g_j), next gen is g_j != g_i.
           So this is a valid non-backtracking 4-walk. And it's closed since the XOR is 0.
           So ANY pair of distinct generators creates a 4-cycle!
           
           Wait, that can't be right. Let me re-examine.
           Walk: 0 --g_i--> g_i --g_j--> g_i^g_j --g_i--> g_j --g_j--> 0
           Vertices: 0, g_i, g_i^g_j, g_j, 0
           For this to be a 4-cycle (not a shorter cycle), we need all intermediate
           vertices distinct from 0 and from each other:
           - g_i != 0 (true, generators non-zero)
           - g_i^g_j != 0 (true, since g_i != g_j)
           - g_i^g_j != g_i (true, since g_j != 0)
           - g_i^g_j != g_j (true, since g_i != 0)
           - g_j != 0 (true)
           - g_j != g_i (true)
           
           So yes, any two distinct generators create a 4-cycle in the Cayley graph!
           
           This means: for girth >= 5, we CANNOT have any two generators whose XOR
           creates a 4-cycle. But as shown, ANY two distinct generators create a 4-cycle.
           
           Wait, that means Cayley graphs on (Z/2Z)^d with involutory generators ALWAYS
           have girth <= 4! That's a critical insight.
           
           Let me double-check with a simple example. Take (Z/2Z)^2 with generators
           e1=01, e2=10. The graph is a 4-cycle: 00-01-11-10-00. Girth is 4. Yes.
           
           Take (Z/2Z)^3 with generators 001, 010, 100. The graph is the 3-cube.
           Girth is 4 (e.g., 000-001-011-010-000). Yes.
           
           So indeed, any Cayley graph on (Z/2Z)^d with involutory generators has
           girth at most 4. This means the random Cayley graph approach CANNOT work
           for girth >= 5!
           
           This is a fundamental flaw in the current approach. The L-U-W construction
           is the only thing that works for girth >= 5.
        """
        # Since any Cayley graph on (Z/2Z)^d with involutory generators has girth <= 4,
        # for g >= 5, we should immediately return False (reject all).
        if g >= 5:
            return False
        # For g <= 4, we need to check if there's a cycle of length < g.
        # For g = 4: reject if there's a 3-cycle.
        # For g = 3: always accept (simple graph has girth >= 3).
        if g <= 3:
            return True
        # g == 4: check for 3-cycles
        k = len(gens)
        for i in range(k):
            for j in range(i+1, k):
                for l in range(j+1, k):
                    if gens[i] ^ gens[j] ^ gens[l] == 0:
                        return False
        return True
    
    def generate_random_generators(d, k):
        """Generate k random distinct non-zero elements of (Z/2Z)^d."""
        gens = set()
        while len(gens) < k:
            x = random.randint(1, (1 << d) - 1)
            gens.add(x)
        return list(gens)
    
    def build_edges_cayley(gens, d):
        """Build edge list for Cayley graph on (Z/2Z)^d with generators gens."""
        n = 1 << d
        edges = []
        for v in range(n):
            for gen in gens:
                u = v ^ gen
                if v < u:
                    edges.append((v, u))
        return n, edges
    
    # Try random Cayley graph search
    best_result = None
    best_n = float('inf')
    
    # Try various dimensions
    # For girth g, we need the diameter of the non-backtracking graph to be at least (g-1)/2
    # The Cayley graph on (Z/2Z)^d with k generators has n = 2^d vertices
    # We want n as small as possible but >= Moore bound
    
    # Start from dimensions that give n close to Moore bound
    min_d = max(1, (mb).bit_length() - 1)  # 2^min_d >= mb roughly
    max_d = min(20, 15)  # Limit dimension for performance
    
    random.seed(42)
    
    # Quick rejection: for g >= 5, Cayley graphs on (Z/2Z)^d with involutory
    # generators ALWAYS have girth <= 4, so skip the search entirely.
    if g >= 5:
        # Skip random Cayley search, go straight to L-U-W
        luw_n, luw_edges = luw_construct(k, g)
        return luw_n, luw_edges
    
    for d in range(min_d, max_d + 1):
        if time.time() - start_time > time_budget:
            break
        
        n = 1 << d
        if n < mb:
            continue
        
        # Try multiple random generator sets
        trials = 5000 if d <= 10 else 1000
        
        for trial in range(trials):
            if time.time() - start_time > time_budget:
                break
            
            gens = generate_random_generators(d, k)
            
            # Quick pre-filter
            if not quick_reject(gens, g):
                continue
            
            # Full girth verification
            if verify_girth_cayley(gens, g, d):
                if n < best_n:
                    best_n = n
                    best_result = build_edges_cayley(gens, d)
    
    # Fall back to L-U-W if no better Cayley graph found
    luw_n, luw_edges = luw_construct(k, g)
    
    if best_result is not None and best_n <= luw_n:
        return best_result
    else:
        return luw_n, luw_edges
# EVOLVE-BLOCK-END
