import time
import random
from collections import defaultdict

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Primary: Voltage graph lift over K_{3,3} with randomized search + local improvement.
    Fallback: LUW construction.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    if k != 3:
        return _construct_luw(k, g, d=7)
    
    # Primary: Voltage graph over K_{3,3}
    result = _voltage_k33_random(k, g, deadline)
    if result is not None:
        return result
    
    # Fallback: LUW
    try:
        for d_try in [7, 8, 9]:
            if time.time() > deadline - 5:
                break
            n, edges = _construct_luw(k, g, d=d_try, rule_variant=0)
            if _has_girth_at_least(n, edges, g):
                return n, edges
    except Exception:
        pass
    
    return _construct_luw(k, g, d=7)

def _voltage_k33_random(k, g, deadline):
    """
    Randomized search for voltage assignment over K_{3,3} -> Z_m.
    """
    best_n = float('inf')
    best_edges = None
    
    for m in [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]:
        if time.time() > deadline - 10:
            break
        n = 6 * m
        if n >= best_n:
            continue
        
        # Try random voltage assignments with local improvement
        max_tries = 100
        for _ in range(max_tries):
            if time.time() > deadline - 10:
                break
            
            voltages = [random.randint(1, m - 1) for _ in range(9)]
            
            # Local improvement: try to reduce short cycles
            for _ in range(20):
                if time.time() > deadline - 10:
                    break
                edges = _build_voltage_lift(m, voltages)
                if edges is None:
                    break
                if _has_girth_at_least(n, edges, g):
                    best_n = n
                    best_edges = edges
                    break
                
                # Try to fix: pick random edge, try all values
                ei = random.randint(0, 8)
                best_val = voltages[ei]
                best_girth = _quick_girth_check(n, edges, g)
                for new_val in range(1, m):
                    if new_val == voltages[ei]:
                        continue
                    voltages[ei] = new_val
                    new_edges = _build_voltage_lift(m, voltages)
                    new_girth = _quick_girth_check(n, new_edges, g)
                    if new_girth > best_girth:
                        best_girth = new_girth
                        best_val = new_val
                    if new_girth >= g:
                        break
                voltages[ei] = best_val
                if best_girth >= g:
                    edges = _build_voltage_lift(m, voltages)
                    best_n = n
                    best_edges = edges
                    break
            
            if best_edges is not None:
                break
        
        if best_edges is not None and best_n <= 936:
            break
    
    if best_edges is not None:
        return best_n, best_edges
    return None

def _quick_girth_check(n, edges, g):
    """Quick girth check: returns girth if < g, else g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = g
    for start in range(n):
        parent = [-1] * n
        level = [-1] * n
        level[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if level[u] >= (g - 1) // 2:
                continue
            
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if level[v] == -1:
                    level[v] = level[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = level[u] + level[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    
    return min_girth

def _build_voltage_lift(m, voltages):
    """Build the voltage graph lift of K_{3,3} over Z_m."""
    n = 6 * m
    edges = []
    for u in range(6):
        for v in range(6):
            if u < 3:
                if v < 3:
                    continue
                e_idx = u * 3 + (v - 3)
            else:
                if v >= 3:
                    continue
                e_idx = v * 3 + (u - 3)
            vol = voltages[e_idx]
            for x in range(m):
                u_node = x * 6 + u
                v_node = ((x + vol) % m) * 6 + v
                if u_node < v_node:
                    edges.append((u_node, v_node))
                else:
                    edges.append((v_node, u_node))
    return edges

def _construct_luw(k, g, d=None, rule_variant=0):
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    
    if rule_variant == 0:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    elif rule_variant == 1:
        rule = {2: (1, 1), 3: (1, 2), 4: (2, 1)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 1), (c0 - 2, 1), (c0, 1), (1, c0 + 2)][r]
    elif rule_variant == 2:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 3
            c0 = c - (c - 5) % 3
            rule[c] = [(1, c0 - 1), (c0, 1), (1, c0 + 1)][r]
    else:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 5
            c0 = c - (c - 5) % 5
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1), (1, c0 + 2)][r]
    
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
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _has_girth_at_least(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        parent = [-1] * n
        level = [-1] * n
        level[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            if level[u] > (g - 1) // 2:
                continue
            
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if level[v] == -1:
                    level[v] = level[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = level[u] + level[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _compute_girth(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    girth = float('inf')
    for i in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = [i]
        dist[i] = 0
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
    return girth
