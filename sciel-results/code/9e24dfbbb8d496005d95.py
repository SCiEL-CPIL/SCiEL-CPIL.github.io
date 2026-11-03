import time
import random
import sys
from collections import deque
import numpy as np

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
    edge_list = [
        (0,1), (0,2), (0,3), (0,4),
        (1,2), (1,3), (1,4),
        (2,3), (2,4),
        (3,4)
    ]
    edge_map = {}
    for i, (u, v) in enumerate(edge_list):
        edge_map[(u, v)] = i
        edge_map[(v, u)] = i
        
    adj = {i: [j for j in range(5) if j != i] for i in range(5)}
    
    # Enumerate all closed walks of length 3 to 9
    constraints = []
    
    def add_constraint(walk):
        coeffs = [0] * 10
        for i in range(len(walk) - 1):
            u, v = walk[i], walk[i+1]
            idx = edge_map[(u, v)]
            if u < v:
                coeffs[idx] += 1
            else:
                coeffs[idx] -= 1
        if all(c == 0 for c in coeffs):
            return
        constraints.append(coeffs)
        
    for start in range(5):
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            L = len(path) - 1
            if 3 <= L <= 9:
                if node == start:
                    add_constraint(path)
            if L < 9:
                for neighbor in adj[node]:
                    stack.append((neighbor, path + [neighbor]))
                    
    unique_constraints = set(tuple(c) for c in constraints)
    constraints = [list(c) for c in unique_constraints]
    
    # Pre-compute constraint matrix as numpy array for fast evaluation
    C = np.array(constraints, dtype=np.int32)  # shape: (num_constraints, 10)
    num_constraints = C.shape[0]
    
    print(f"Found {num_constraints} unique constraints", file=sys.stderr)
    
    # Simulated annealing with multiple restarts
    best_result = None
    max_restarts = 80
    restart_budget = (deadline - start_time - 10) / max_restarts if deadline > start_time + 10 else 0
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        voltages = np.array([random.randint(0, 63) for _ in range(10)], dtype=np.int32)
        
        def count_bad(v):
            sums = C @ v
            return np.sum(sums % 64 == 0)
            
        current_bad = count_bad(voltages)
        
        if current_bad == 0:
            best_result = voltages.copy()
            break
            
        # Simulated annealing
        T = 2.0
        cooling = 0.995
        max_steps = int(min(5000, max(500, restart_budget * 50)))
        
        for step in range(max_steps):
            if time.time() > deadline - 10:
                    break
            if current_bad == 0:
                best_result = voltages.copy()
                break
                
            e_idx = random.randint(0, 9)
            old_v = voltages[e_idx]
            
            # Try all 64 values for this edge
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
            
            if best_bad < current_bad:
                voltages[e_idx] = best_v
                current_bad = best_bad
            elif random.random() < np.exp(-(best_bad - current_bad) / T):
                voltages[e_idx] = best_v
                current_bad = best_bad
            else:
                voltages[e_idx] = old_v
            
            T *= cooling
        
        if current_bad == 0:
            best_result = voltages.copy()
            break
            
    if best_result is not None:
        voltages = best_result
        n_lift = 320
        edges = []
        for u, v_base in edge_list:
            w = int(voltages[edge_map[(u, v_base)]])
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
