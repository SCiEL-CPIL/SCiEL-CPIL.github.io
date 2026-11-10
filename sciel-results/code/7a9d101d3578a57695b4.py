import random
import time
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # If k=4, g=10, try to find a smaller graph via voltage lift
    if k == 4 and g == 10:
        # Try to construct a lift of K_4,4
        # Base graph: K_4,4
        # Vertices 0..3 (U), 4..7 (V)
        # Edges (u, 4+v) for u in 0..3, v in 0..3
        
        # We want a lift of size 8*m
        # Try m values
        for m in [40, 50, 60, 80, 100]:
            if time.time() > deadline - 20:
                break
            
            # Random search for voltages
            # Fix voltages on a spanning tree to 0 to reduce symmetry
            # Spanning tree of K_4,4: 
            # (0,4), (1,4), (2,4), (3,4), (0,5), (1,5), (2,5)
            # Remaining edges: 9
            # Let's just assign random voltages to all 16 edges and check
            # Or fix 7 to 0.
            
            found = False
            attempts = 0
            while time.time() < deadline - 10 and attempts < 5000:
                attempts += 1
                # Assign random voltages to non-tree edges
                # Tree edges are 0
                # Non-tree edges indices:
                # Total 16 edges.
                # Tree: 
                # 0-4, 1-4, 2-4, 3-4 (4 edges)
                # 0-5, 1-5, 2-5 (3 edges)
                # Total 7 tree edges.
                # Non-tree: 9 edges.
                
                # Let's just use a dict for voltages
                voltages = {}
                for u in range(4):
                    for v in range(4):
                        # Edge (u, 4+v)
                        if (u == 0 and v == 0) or (u == 1 and v == 0) or (u == 2 and v == 0) or (u == 3 and v == 0) or \
                           (u == 0 and v == 1) or (u == 1 and v == 1) or (u == 2 and v == 1):
                            voltages[(u, 4+v)] = 0
                        else:
                            voltages[(u, 4+v)] = random.randint(0, m-1)
                
                # Check validity
                # We need to check all non-backtracking closed walks of length < 10
                # This is equivalent to checking that the lift has girth >= 10
                # We can simulate BFS on the lift? No, lift is too big.
                # We can check the voltage sums of closed walks in the base.
                
                # Generate all non-backtracking closed walks in K_4,4 of length 4, 6, 8
                # Length 4:
                # u -> v -> u' -> v' -> u
                # u != u', v != v'
                # Sum: V(u,v) - V(u',v) + V(u',v') - V(u,v')
                # Note: direction matters.
                # Edge (u, 4+v) has voltage x. Reverse has -x.
                
                valid = True
                # Length 4
                for u in range(4):
                    for v in range(4):
                        for u2 in range(4):
                            if u2 == u: continue
                            for v2 in range(4):
                                if v2 == v: continue
                                # Walk: u -> 4+v -> u2 -> 4+v2 -> u
                                # Edges: (u, 4+v), (u2, 4+v) [rev], (u2, 4+v2), (u, 4+v2) [rev]
                                s = voltages[(u, 4+v)] - voltages[(u2, 4+v)] + voltages[(u2, 4+v2)] - voltages[(u, 4+v2)]
                                if s % m == 0:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: continue
                
                # Length 6
                # u -> v -> u2 -> v2 -> u3 -> v3 -> u
                # u, u2, u3 distinct? Not necessarily, but non-backtracking means u3 != u2.
                # Also v3 != v2, v2 != v, v != v3 (if u3=u).
                # This is getting complex. 
                # Simpler: Just build the lift graph for this m and check girth directly?
                # Lift size 8*m. For m=40, 320 vertices.
                # Checking girth 10 on 320 vertices is fast.
                
                # Let's build the lift
                n_lift = 8 * m
                # Vertices: (base_node, coset)
                # Edges: (u, c) connected to (4+v, (c + V(u,v)) % m)
                
                # Build adjacency list
                adj = [[] for _ in range(n_lift)]
                for u in range(4):
                    for v in range(4):
                        vol = voltages[(u, 4+v)]
                        for c in range(m):
                            u_idx = u * m + c
                            v_idx = (4+v) * m + (c + vol) % m
                            adj[u_idx].append(v_idx)
                            adj[v_idx].append(u_idx)
                
                # Check girth
                # BFS from each vertex to depth 5
                girth_ok = True
                for start in range(n_lift):
                    visited = [-1] * n_lift
                    visited[start] = 0
                    queue = deque([(start, 0)])
                    while queue:
                        node, dist = queue.popleft()
                        if dist >= 5:
                            continue
                        for nei in adj[node]:
                            if visited[nei] == -1:
                                visited[nei] = dist + 1
                                queue.append((nei, dist + 1))
                            elif visited[nei] != dist - 1:
                                # Found a cycle
                                # Cycle length = dist + 1 + visited[nei]
                                # If dist + 1 + visited[nei] < 10, girth is violated
                                if dist + 1 + visited[nei] < 10:
                                    girth_ok = False
                                    break
                        if not girth_ok:
                            break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    # We found a valid graph
                    edges = []
                    for u in range(n_lift):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n_lift, edges
            # End while attempts
        # End for m
        
    # Fallback to LUW
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
