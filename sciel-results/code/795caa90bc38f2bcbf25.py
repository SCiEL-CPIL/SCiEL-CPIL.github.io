"""Voltage-lift construction over Petersen graph with simulated annealing."""
import time
import random
import sys

def construct(k, g):
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Base graph: Petersen graph (10 vertices, 15 edges, girth 5, 3-regular)
    # Petersen graph edges
    pet_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),  # outer 5-cycle
        (5,6), (6,7), (7,8), (8,9), (9,5),  # inner 5-cycle
        (0,5), (1,6), (2,7), (3,8), (4,9)   # spokes
    ]
    n_base = 10
    m_edges = len(pet_edges)  # 15
    
    # Build adjacency for Petersen
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(pet_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    # Enumerate all non-backtracking closed walks up to length g-1 in Petersen
    # A non-backtracking walk: v0, v1, ..., vl where v0=vl, and no immediate backtracking
    # We track (current_vertex, previous_vertex, length)
    # For girth g, we need walks of length 5 to g-1 (since girth 5 means 5-cycles exist)
    
    # Actually for girth >= g, we need no cycles of length < g.
    # In the lift, a non-backtracking closed walk of length L in the base lifts to a cycle
    # of length L iff the voltage product is trivial (sum = 0 mod m).
    # So we need: for every non-backtracking closed walk of length L in [5, g-1], voltage sum != 0 mod m.
    
    # Enumerate non-backtracking closed walks
    # State: (vertex, prev_vertex) - walks start and end at same vertex
    # We'll enumerate walks of each length by BFS on directed edges
    
    # Directed edges: (u, v) for each undirected edge
    dir_edges = []
    dir_edge_idx = {}
    for i, (u, v) in enumerate(pet_edges):
        dir_edges.append((u, v))
        dir_edges.append((v, u))
        dir_edge_idx[(u, v)] = i  # map to original edge index
        dir_edge_idx[(v, u)] = i
    
    n_dir = len(dir_edges)  # 30
    
    # For each directed edge (u,v), the next directed edge must start at v and not go back to u
    next_dir = [[] for _ in range(n_dir)]
    for i, (u, v) in enumerate(dir_edges):
        for j, (w, x) in enumerate(dir_edges):
            if w == v and x != u:
                next_dir[i].append(j)
    
    # Find all non-backtracking closed walks of length L for L in [5, g-1]
    # A closed walk of length L: sequence of directed edges e0, e1, ..., e_{L-1}
    # where e_i ends at start of e_{i+1}, and e_{L-1} ends at start of e_0
    # Walk: v0 -e0-> v1 -e1-> v2 ... v_{L-1} -e_{L-1}-> v0
    
    # For efficiency, enumerate walks up to length g-1
    max_len = g - 1  # 15 for g=16
    
    # We'll store walks as tuples of directed edge indices
    # For each length, find all closed walks
    # Use: for each starting directed edge, extend walks
    
    all_walks = []  # list of (length, [dir_edge_indices])
    
    # For each length from 5 to max_len
    # Start: for each directed edge e, a walk of length 1 is just [e]
    # Extend: a walk [e0, ..., e_{L-1}] can be extended by e_L if e_{L-1} ends at start of e_L
    # Closed: e_{L-1} ends at start of e_0
    
    # More efficient: for each length L, find all sequences e0...e_{L-1} such that
    # end(e_{L-1}) = start(e_0) and consecutive edges connect properly
    
    # Let's enumerate by starting edge and using iterative extension
    # For L=1: walk [e] is closed if end(e) = start(e) - never true for simple graph
    # For L>=2: we track partial walks
    
    # Represent a partial walk by (last_dir_edge, start_dir_edge) and the sequence
    # But sequences can be long. Let's just enumerate all closed walks directly.
    
    # For each starting directed edge s, find all walks of length L starting with s
    # that are closed (end at start of s)
    
    # Walk of length L starting with directed edge s=(u0,v0):
    # e0 = s, e1 in next_dir[s], e2 in next_dir[e1], ..., e_{L-1} in next_dir[e_{L-2}]
    # Closed: end(e_{L-1}) = u0 (start of s)
    
    # This is exponential in L, but L <= 15 and branching factor is 2 (since 3-regular, 
    # from a directed edge you can go to 2 other directed edges)
    # So at most 2^14 = 16384 walks per starting edge per length. 30 starting edges.
    # Total: 30 * sum_{L=5}^{15} 2^{L-1} ≈ 30 * 2^15 ≈ 1M. Manageable.
    
    for L in range(5, max_len + 1):
        for s in range(n_dir):
            # DFS to find all walks of length L starting with s
            # Stack: (current_dir_edge, length_so_far, path)
            stack = [(s, 1, [s])]
            while stack:
                cur, ln, path = stack.pop()
                if ln == L:
                    # Check if closed: end of cur = start of s
                    cur_end = dir_edges[cur][1]
                    s_start = dir_edges[s][0]
                    if cur_end == s_start:
                        all_walks.append((L, path))
                    continue
                # Extend
                for nxt in next_dir[cur]:
                    stack.append((nxt, ln + 1, path + [nxt]))
    
    # Convert walks to edge-index sequences (for voltage assignment)
    # Each directed edge maps to an undirected edge index
    # The voltage of a walk is the sum of voltages of its edges (with signs based on direction)
    # If edge i has voltage v_i, and the walk uses edge i in forward direction, contribution is +v_i
    # If in reverse direction, contribution is -v_i
    
    # For each walk, compute the signed edge usage
    # walk_edge_signs: list of (edge_idx, sign) where sign is +1 or -1
    
    walk_constraints = []  # list of lists of (edge_idx, sign)
    for L, path in all_walks:
        signs = []
        for de in path:
            u, v = dir_edges[de]
            ei = dir_edge_idx[(u, v)]
            # Determine direction: if (u,v) matches original edge direction, sign +1, else -1
            orig_u, orig_v = pet_edges[ei]
            if u == orig_u and v == orig_v:
                sign = 1
            else:
                sign = -1
            signs.append((ei, sign))
        walk_constraints.append(signs)
    
    n_constraints = len(walk_constraints)
    print(f"Found {n_constraints} non-backtracking closed walks of length 5-{max_len}", file=sys.stderr)
    
    # Now assign voltages to edges (15 edges) from Z_m
    # We need: for each constraint, sum(sign_i * v_{edge_i}) != 0 mod m
    # This is a system of linear inequalities mod m
    
    # Simulated annealing: start with random assignment, try to minimize number of violated constraints
    
    # Choose m: we want 10*m vertices, aim for n close to 512 but not below Moore bound
    # Moore bound for (3,16): 1 + 3*sum_{i=0}^{7} 2^i = 1 + 3*(2^8 - 1) = 1 + 3*255 = 766? 
    # Wait, Moore bound for cubic girth g: 1 + 3*sum_{i=0}^{(g-1)/2} 2^i for odd g
    # g=16 is even: 3*sum_{i=0}^{g/2 - 1} 2^i = 3*(2^8 - 1) = 3*255 = 765
    # Hmm, but the task says lower bound is 512. Let me recheck.
    # Actually for even girth g=2r: Moore bound = 3*sum_{i=0}^{r-1} 2^i = 3*(2^r - 1)
    # g=16, r=8: 3*(256-1) = 765. But task says 512. Maybe different formula.
    # Regardless, we want n = 10*m to be as small as possible while valid.
    # Try m values: 52 (n=520), 53 (n=530), etc.
    
    # For the voltage assignment to work, m should be large enough that random assignments
    # have low probability of hitting 0. With ~1M constraints and 15 variables, we need m large.
    # Probability a random assignment satisfies one constraint: (m-1)/m
    # All constraints: ((m-1)/m)^{1M} - this is tiny for any reasonable m.
    # So random won't work. We need a smarter approach.
    
    # Better: use the fact that we only need to avoid 0. Use greedy + local search.
    # Or: use a different base graph with fewer short walks.
    
    # Alternative: use the dodecahedron (20 vertices, girth 6) as base.
    # Or: just try to find ANY valid assignment with simulated annealing, accepting that
    # we might need to iterate.
    
    # Actually, let me reconsider. The number of constraints might be manageable if m is large.
    # Let's just try simulated annealing with m = 52 (giving n=520).
    
    best_result = None
    
    # Try several values of m
    for m in [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]:
        if time.time() > deadline - 30:
            break
        
        # Simulated annealing
        n_edges_base = m_edges  # 15
        # Initial random assignment
        voltages = [random.randint(0, m - 1) for _ in range(n_edges_base)]
        
        def count_violations(v):
            count = 0
            for signs in walk_constraints:
                total = 0
                for ei, sign in signs:
                    total += sign * v[ei]
                if total % m == 0:
                    count += 1
            return count
        
        best_v = voltages[:]
        best_count = count_violations(best_v)
        current_v = voltages[:]
        current_count = best_count
        
        T = 1.0
        cooling = 0.999
        steps = 200000
        step = 0
        
        while step < steps and time.time() < deadline - 30:
            step += 1
            # Pick random edge to change
            ei = random.randint(0, n_edges_base - 1)
            old_val = current_v[ei]
            new_val = random.randint(0, m - 1)
            if new_val == old_val:
                continue
            current_v[ei] = new_val
            new_count = count_violations(current_v)
            delta = new_count - current_count
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                current_count = new_count
                if new_count < best_count:
                    best_count = new_count
                    best_v = current_v[:]
            else:
                current_v[ei] = old_val
            T *= cooling
            
            if best_count == 0:
                break
        
        if best_count == 0:
            # Valid assignment found! Build the lift
            lift_voltages = best_v
            n_lift = n_base * m  # 10 * m
            
            # Build lifted graph
            # Vertex (base_vertex, group_element)
            # Edge between (u, g) and (v, g + voltage(u,v)) for each base edge (u,v)
            # Need consistent orientation: for edge i between (a,b), voltage v_i
            # From a to b: add v_i; from b to a: add -v_i (i.e., m - v_i)
            
            lift_edges = []
            for i, (u, v) in enumerate(pet_edges):
                vol = lift_voltages[i]
                for g in range(m):
                    # Edge from (u, g) to (v, (g + vol) % m)
                    vu = u * m + g
                    vv = v * m + ((g + vol) % m)
                    lift_edges.append((vu, vv))
                    # Edge from (v, g) to (u, (g - vol) % m)
                    # But this is the same undirected edge, so we only add once
                    # Actually, for each base edge, we add m edges in the lift
                    # The edge (u,g)-(v,g+vol) is the same as (v,g+vol)-(u,g)
                    # So we just add it once per (i, g)
            
            # Wait, I'm double counting. For each base edge i and each g in Z_m,
            # there's exactly one lifted edge: (u, g) -- (v, g + vol_i)
            # So total edges = 15 * m. For m=52, that's 780 edges.
            # n_lift = 10*m = 520 vertices, 780 edges. 3-regular: 520*3/2 = 780. Correct.
            
            # Verify it's valid (quick check)
            # Check regularity
            deg = [0] * n_lift
            for u, v in lift_edges:
                deg[u] += 1
                deg[v] += 1
            if all(d == k for d in deg):
                # Check no short cycles via non-backtracking (simplified: trust the construction)
                # The construction guarantees girth >= g if voltage assignment is valid
                best_result = (n_lift, lift_edges)
                break
        
        if best_result:
            break
    
    if best_result:
        return best_result
    
    # Fallback: return the original Lazebnik-Ustimenko construction
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
