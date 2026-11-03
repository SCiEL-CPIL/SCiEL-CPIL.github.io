import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    if k == 4 and g == 10:
        result = _construct_4_10_voltages(start_time, deadline)
        if result:
            return result
        # Fallback
        return _luw_construct(k, g)
    
    return _luw_construct(k, g)

def _construct_4_10_voltages(start_time, deadline):
    # Base graph: K5
    # Vertices: 0, 1, 2, 3, 4
    # Edges: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
    # Map edge to index
    edge_list = [
        (0,1), (0,2), (0,3), (0,4),
        (1,2), (1,3), (1,4),
        (2,3), (2,4),
        (3,4)
    ]
    edge_index = {e: i for i, e in enumerate(edge_list)}
    edge_map = {}
    for i, (u, v) in enumerate(edge_list):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    # Adjacency for K5
    adj = {i: [j for j in range(5) if j != i] for i in range(5)}
    
    # Enumerate all closed walks of length 3 to 9
    constraints = []
    
    def add_constraint(walk):
        # walk is a list of vertices [v0, v1, ..., vL] where v0 == vL
        coeffs = [0] * 10
        for i in range(len(walk) - 1):
            u, v = walk[i], walk[i+1]
            idx = edge_map[(u, v)]
            # Determine sign
            # We defined edge_list with u < v?
            # edge_list: (0,1), (0,2)... yes u < v.
            # If u < v, we traverse in forward direction, coeff +1.
            # If u > v, we traverse in reverse direction, coeff -1.
            if u < v:
                coeffs[idx] += 1
            else:
                coeffs[idx] -= 1
        # Check if constraint is trivial (all zeros)?
        # A closed walk in a simple graph cannot have all zero coefficients unless it's length 0.
        # But we might have backtracking like 0-1-0-1-0.
        # Coeffs: (0,1)+1, (1,0)-1, (0,1)+1, (1,0)-1 -> 0.
        # Such walks are "trivial" in terms of voltage sum (sum is always 0).
        # We MUST exclude these from constraints, because we cannot prevent sum=0 for them.
        # Wait, if a walk has coefficient vector 0, the voltage sum is always 0.
        # Does this correspond to a cycle in the lift?
        # Yes, if the walk is a closed walk in the base, and the voltage sum is 0,
        # it closes in the lift.
        # If the walk is 0-1-0-1-0, it is not a simple cycle in the base.
        # In the lift, 0-1-0-1-0 corresponds to a walk.
        # Is it a cycle? A cycle must be simple (no repeated vertices).
        # The walk 0-1-0-1-0 has repeated vertices.
        # The task requires "no cycle shorter than g".
        # A "cycle" usually implies a simple cycle.
        # The girth is the length of the shortest simple cycle.
        # So we only need to prevent **simple** cycles in the lift.
        # A simple cycle in the lift projects to a **closed walk** in the base.
        # However, if the closed walk in the base is not a simple cycle (i.e., it backtracks or repeats vertices),
        # can it form a simple cycle in the lift?
        # If the base walk is 0-1-0-1-0, the lift walk is (0,k)-(1,k+w)-(0,k)-(1,k+w)-(0,k).
        # This is not a simple cycle because (0,k) and (1,k+w) are repeated.
        # In general, a simple cycle in the lift must project to a **closed walk** in the base
        # that does not contain any "backtracking" or "repeated vertices" in a way that causes repetition in the lift.
        # Actually, it's safer to just check **all** closed walks.
        # If a non-simple closed walk in the base has voltage sum 0, it creates a closed walk in the lift.
        # Does it create a simple cycle?
        # If the base walk is 0-1-0-1-0, the lift walk is not simple.
        # So we only care about base walks that are **simple cycles** in the base?
        # No. Consider base walk 0-1-2-1-0. (Length 4).
        # Lift: (0,k)-(1,k+w1)-(2,k+w1+w2)-(1,k+w1)-(0,k).
        # Vertices: (0,k), (1,k+w1), (2,k+w1+w2), (1,k+w1), (0,k).
        # Repeated: (1,k+w1) and (0,k). Not simple.
        # It seems that if the base walk is not a simple cycle, the lift walk is not a simple cycle.
        # Proof sketch: If the base walk repeats a vertex $v$, say at steps $i$ and $j$ ($i < j$),
        # the lift walk visits $(v, s_i)$ and $(v, s_j)$.
        # If $s_i \neq s_j$, they are distinct vertices in the lift.
        # But the walk is closed, so $s_0 = s_L$.
        # If the base walk is not simple, it contains a sub-walk that is a closed walk.
        # This implies the lift walk contains a sub-walk that is a closed walk.
        # A simple cycle cannot contain a proper closed sub-walk.
        # Therefore, we only need to check **simple cycles** in the base graph.
        
        # However, $K_5$ has girth 3.
        # Simple cycles in $K_5$:
        # Length 3: Triangles.
        # Length 4: Squares.
        # Length 5: Pentagons.
        # There are no simple cycles of length > 5 in $K_5$ (since it has 5 vertices).
        # Wait, the lift has 320 vertices.
        # A simple cycle in the lift of length $L$ projects to a closed walk of length $L$ in the base.
        # If $L > 5$, the closed walk in the base MUST repeat vertices (since base has 5 vertices).
        # So the closed walk is NOT a simple cycle in the base.
        # Does this mean we don't need to check them?
        # No! The projection of a simple cycle in the lift is a closed walk in the base.
        # If the closed walk in the base repeats vertices, the lift walk is still potentially simple.
        # Example: Base walk 0-1-2-3-1-4-0 (Length 6).
        # Repeats 1.
        # Lift: (0,k)-(1,k+w01)-(2,k+w01+w12)-(3,k+w01+w12+w23)-(1,k+w01+w12+w23+w31)-(4,...)-(0,k).
        # The vertex (1, ...) appears twice in the lift walk?
        # First time: (1, k+w01).
        # Second time: (1, k+w01+w12+w23+w31).
        # If these are different, the lift walk is simple (so far).
        # So YES, we MUST check all closed walks of length < 10.
        # But we can skip walks that are "obviously" non-simple in the lift?
        # It's hard to tell.
        # So we check all closed walks of length 3..9.
        # But we MUST skip walks where the coefficient vector is all zeros.
        # Because for those, the voltage sum is ALWAYS 0.
        # And they DO correspond to closed walks in the lift.
        # BUT, do they correspond to SIMPLE cycles in the lift?
        # If the coefficient vector is zero, it means the walk is a "null" walk in the voltage sense.
        # Usually, these are walks with backtracking (e.g., 0-1-0).
        # As argued, backtracking walks in the base produce non-simple walks in the lift.
        # So we can safely ignore constraints where coeffs are all zero.
        
        if all(c == 0 for c in coeffs):
            return
        constraints.append(coeffs)
        
    # Enumerate
    for start in range(5):
        # DFS
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            L = len(path) - 1
            if 3 <= L <= 9:
                if node == start:
                    add_constraint(path)
            if L < 9:
                for neighbor in adj[node]:
                    # To avoid duplicates, we can enforce an ordering?
                    # Or just add all.
                    # For K5, number of walks is small.
                    stack.append((neighbor, path + [neighbor]))
                    
    # Deduplicate constraints?
    # Many walks will have the same coefficient vector.
    # We can use a set of tuples.
    unique_constraints = set(tuple(c) for c in constraints)
    constraints = [list(c) for c in unique_constraints]
    
    print(f"Found {len(constraints)} unique constraints", file=sys.stderr)
    
    # Local Search
    voltages = [random.randint(0, 63) for _ in range(10)]
    
    def count_bad(v):
        bad = 0
        for c in constraints:
            s = 0
            for i in range(10):
                if c[i] != 0:
                    s += c[i] * v[i]
            if s % 64 == 0:
                bad += 1
        return bad
        
    current_bad = count_bad(voltages)
    
    # Hill Climbing
    max_steps = 10000
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        if current_bad == 0:
            break
            
        # Pick random edge
        e_idx = random.randint(0, 9)
        old_v = voltages[e_idx]
        
        best_v = old_v
        best_bad = current_bad
        
        for new_v in range(64):
            if new_v == old_v:
                continue
            voltages[e_idx] = new_v
            bad = count_bad(voltages)
            if bad < best_bad:
                best_bad = bad
                best_v = new_v
        
        voltages[e_idx] = best_v
        current_bad = best_bad
        
    if current_bad == 0:
        # Construct graph
        n_lift = 320
        edges = []
        for u, v_base in edge_list:
            w = voltages[edge_map[(u, v_base)]]
            for x in range(64):
                u_id = u * 64 + x
                v_id = v_base * 64 + (x + w) % 64
                edges.append((u_id, v_id))
                
        if _check_girth_fast(n_lift, edges, 10):
            return n_lift, edges
            
    return None

def _check_girth_fast(n, edges, min_girth):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = min_girth // 2 - 1
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            d = dist[u]
            if d >= max_depth:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = d + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    cycle_len = d + 1 + dist[v]
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g):
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
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)

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
