import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # Fallback: L-U-W construction from the original code
    def luw_construct(k, g):
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

    # Moore bound for even girth g
    if g % 2 == 0:
        moore_bound = 1 + k * sum((k-1)**i for i in range(g//2 - 1))
    else:
        moore_bound = 2 * sum((k-1)**i for i in range((g-1)//2))
        
    # The prompt says best lower bound is 243 for k=4, g=10.
    # We start searching from max(moore_bound, 243) if k==4 and g==10, else moore_bound.
    search_start = max(moore_bound, 243) if (k == 4 and g == 10) else moore_bound
    
    def check_girth(n, generators):
        # Check if the Cayley graph on Z_n with generators has girth >= g
        # g is even (10).
        # We need to ensure no cycles of length 3, 4, ..., g-1.
        # Since it's a Cayley graph, we can check from vertex 0.
        # But girth is global. However, for vertex-transitive graphs, girth is determined by shortest closed walk from identity.
        # Wait, girth is min length of any cycle. In a vertex transitive graph, every vertex looks the same.
        # So we just need to find the shortest non-backtracking closed walk starting at 0.
        
        # State: (current_pos, last_gen_index)
        # last_gen_index is the index of the generator used to arrive here, or -1 if start.
        # To avoid backtracking, we cannot use -generator[last_gen_index].
        
        # We do BFS on the state space (pos, last_idx).
        # Depth d.
        # If we reach (0, idx) with depth d >= 1, we have a closed walk.
        # But we must ensure it's a simple cycle?
        # A closed walk of length L in a simple graph contains a cycle of length <= L.
        # If we find a closed walk of length L < g, does it imply a cycle < g?
        # Yes, any closed walk contains a simple cycle.
        # So if the shortest closed non-backtracking walk has length L, the girth is L.
        # (Assuming no shorter cycles exist that are "hidden" by backtracking? No, non-backtracking walk covers all simple cycles).
        
        # BFS
        # visited: set of (pos, last_idx)
        # queue: list of (pos, last_idx, depth)
        
        visited = set()
        queue = [(0, -1, 0)]
        visited.add((0, -1))
        
        while queue:
            pos, last_idx, depth = queue.pop(0)
            
            if depth >= g:
                return True # Girth >= g
            
            # If we return to 0 with depth > 0, we have a cycle of length depth.
            # But we must ensure it's not a trivial back-and-forth?
            # The non-backtracking constraint prevents immediate reversal.
            # So any return to 0 with depth > 0 is a valid cycle candidate.
            if pos == 0 and depth > 0:
                return depth >= g
            
            for i, gen in enumerate(generators):
                next_pos = (pos + gen) % n
                # Check backtracking
                if last_idx != -1:
                    # The generator used to arrive was generators[last_idx].
                    # We cannot use -generators[last_idx].
                    if gen == (-generators[last_idx]) % n:
                        continue
                
                state = (next_pos, i)
                if state not in visited:
                    visited.add(state)
                    queue.append((next_pos, i, depth + 1))
        
        return True # If queue empties, no cycle < g found (graph might be disconnected? No, we started at 0. If connected, all reachable. If not connected, girth is inf for other components? No, girth is min over all cycles. If component of 0 has girth >= g, but another has < g, we fail. But Cayley graphs are vertex transitive, so all components are isomorphic if connected? No, if not connected, they are cosets. All cosets have same girth. So checking 0 is sufficient.)

    def is_valid_generators(n, generators):
        # Check simple graph: no loops, no multi-edges.
        # No loops: no gen is 0.
        if any(gen == 0 for gen in generators):
            return False
        # No multi-edges: all generators distinct, and no gen is -other.
        gen_set = set(generators)
        if len(gen_set) != k:
            return False
        for gen in generators:
            if (-gen) % n in gen_set:
                return False
        
        # Check connectivity?
        # For Cayley graph, connected iff generators generate Z_n.
        # gcd of generators and n must be 1.
        import math
        g_val = n
        for gen in generators:
            g_val = math.gcd(g_val, gen)
        if g_val != 1:
            return False
            
        # Check girth
        return check_girth(n, generators)

    # Search loop
    current_n = search_start
    max_n = 1000 # Upper bound for search
    
    while current_n <= max_n and time.time() < deadline:
        # Try random generators
        attempts = 0
        max_attempts = 5000 if current_n < 300 else 10000
        found = False
        
        while attempts < max_attempts and time.time() < deadline:
            # Generate k distinct non-zero, non-inverse pairs
            # Available elements: 1..n-1
            # We need k elements such that no two are inverses.
            # Pick k elements randomly from 1..n-1, ensuring distinct and no inverses.
            
            used = set()
            gens = []
            valid = True
            for _ in range(k):
                gen = random.randint(1, n - 1)
                if gen in used or (-gen) % n in used:
                    valid = False
                    break
                used.add(gen)
                gens.append(gen)
            
            if not valid:
                attempts += 1
                continue
                
            if is_valid_generators(current_n, gens):
                # Construct edges
                edges = []
                for u in range(current_n):
                    for v_gen in gens:
                        v = (u + v_gen) % current_n
                        if u < v:
                            edges.append((u, v))
                return current_n, edges
        
        current_n += 1

    # Fallback
    return luw_construct(k, g)
