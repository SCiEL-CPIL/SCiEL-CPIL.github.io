import time
import random
import math
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    DEADLINE = start_time + 140.0

    if not (k == 3 and g == 16):
        return _luw_construct(k, g)

    # Try to construct a graph of size 936 using voltage lift
    # 936 = 2^3 * 3^2 * 13
    # Divisors: 1, 2, 3, 4, 6, 8, 9, 12, 13, 18, 24, 26, 36, 39, 52, 78, 104, 117, 156, 234, 312, 468, 936
    # Base size N_B must be even (cubic graph) and >= 4.
    # m = 936 / N_B.
    
    target_n = 936
    divisors = [12, 18, 24, 26, 36, 39, 52, 78] # Small bases are better for fewer constraints
    
    for N_B in divisors:
        if time.time() > DEADLINE:
            break
        m = target_n // N_B
        if m < 1: continue
        
        # Try to find a valid voltage assignment
        # Generate several random base graphs
        for attempt in range(50):
            if time.time() > DEADLINE:
                break
            base_edges = _generate_random_cubic_graph(N_B, attempt + N_B)
            if not base_edges:
                continue
            
            # Check if this base graph allows a valid lift
            # We need to assign voltages in Z_m
            # Constraints: for all cycles C with len < 16, lift_cycle_len >= 16
            
            # Get all cycles of length < 16 in base graph
            cycles = _get_short_cycles(N_B, base_edges, 15)
            
            # If no short cycles, any voltage assignment works (girth of base >= 16? Unlikely for small N_B)
            # Actually if base has girth >= 16, then m=1 would work, but m>1 here.
            # If base has short cycles, we need to avoid "bad" voltage sums.
            
            # Randomly assign voltages
            found = False
            for vol_attempt in range(100):
                if time.time() > DEADLINE:
                    break
                labels = {}
                for u, v in base_edges:
                    key = (min(u, v), max(u, v))
                    labels[key] = random.randint(0, m - 1)
                
                valid = True
                for cycle in cycles:
                    # cycle is a list of vertices [v0, v1, ..., v_{l-1}]
                    # Edges are (v0,v1), (v1,v2), ..., (v_{l-1}, v0)
                    # Sum of voltages
                    s = 0
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i+1) % len(cycle)]
                        key = (min(u, v), max(u, v))
                        l_val = labels[key]
                        # Direction matters for sum?
                        # Voltage of edge u->v is l. v->u is -l.
                        # We need to define orientation.
                        # Let's fix an orientation for each edge in base graph.
                        # If edge is (u,v) with u<v, voltage is l.
                        # If we traverse v->u, voltage is -l.
                        # In the cycle, we traverse edges in order.
                        # We need to know if we go u->v or v->u.
                        # This requires careful bookkeeping.
                        # Alternatively, just check if sum is 0 mod m?
                        # No, sum must be 0 for a cycle of length l in the lift.
                        # The condition for a cycle of length l in the lift is that the voltage sum is 0.
                        # So we need sum != 0.
                        
                        # Let's compute the signed sum.
                        # We need to know the direction.
                        # Let's store edges as (u, v, l) with u < v.
                        # When traversing cycle, if we go u->v, add l. If v->u, add -l.
                        
                        if u < v:
                            s += l_val
                        else:
                            s -= l_val
                    s %= m
                    
                    if s == 0:
                        valid = False
                        break
                    
                    # Also check length condition
                    # lift_len = l * m / gcd(s, m)
                    # We need lift_len >= 16
                    l_cyc = len(cycle)
                    g = math.gcd(s, m)
                    lift_len = l_cyc * m // g
                    if lift_len < 16:
                        valid = False
                        break
                
                if valid:
                    # Build the lift
                    n = N_B * m
                    cover_edges = []
                    cover_edge_set = set()
                    for u, v in base_edges:
                        key = (min(u, v), max(u, v))
                        l_val = labels[key]
                        for g_idx in range(m):
                            u1 = u * m + g_idx
                            v1 = v * m + (g_idx + l_val) % m
                            if u1 < v1:
                                e_key = (u1, v1)
                            else:
                                e_key = (v1, u1)
                            if e_key not in cover_edge_set:
                                cover_edge_set.add(e_key)
                                cover_edges.append((min(u1, v1), max(u1, v1)))
                    
                    # Verify
                    if len(cover_edges) != n * 3 // 2:
                        continue
                    
                    # Check girth
                    girth = _get_girth_nonbacktracking(n, cover_edges)
                    if girth >= g:
                        return n, cover_edges
            
            if found:
                break

    # Fallback to L-U-W
    return _luw_construct(k, g)

def _get_short_cycles(n, edges, max_len):
    """
    Find all simple cycles of length <= max_len in a graph.
    Returns a list of cycles, each cycle is a list of vertices.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    cycles = []
    seen = set()
    
    for start in range(n):
        # DFS
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > max_len:
                continue
            if len(path) >= 3 and node == start:
                # Found a cycle
                # Normalize cycle to avoid duplicates
                # Find min vertex
                min_v = min(path)
                min_idx = path.index(min_v)
                # Rotate
                rotated = path[min_idx:] + path[:min_idx]
                # Check reverse
                rev = [rotated[0]] + rotated[1:][::-1]
                key = tuple(sorted([tuple(rotated), tuple(rev)]))
                if key not in seen:
                    seen.add(key)
                    cycles.append(rotated)
                continue
            
            for neighbor in adj[node]:
                if neighbor in path:
                    if neighbor == start:
                        # Potential cycle
                        pass
                    else:
                        # Backtrack to earlier node, not a simple cycle extension
                        continue
                else:
                    stack.append((neighbor, path + [neighbor]))
                    
    return cycles

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    """
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

def _generate_random_cubic_graph(n, seed_offset=0):
    if n < 4 or n % 2 != 0:
        return None
    
    rng = random.Random(n * 1000 + seed_offset)
    
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    
    rng.shuffle(stubs)
    
    pairs = [(stubs[i], stubs[i+1]) for i in range(0, len(stubs), 2)]
    
    max_repair_attempts = 100
    for repair_iter in range(max_repair_attempts):
        has_loop = False
        has_multi = False
        edge_set = set()
        for u, v in pairs:
            if u == v:
                has_loop = True
                break
            key = (min(u,v), max(u,v))
            if key in edge_set:
                has_multi = True
                break
            edge_set.add(key)
        
        if not has_loop and not has_multi:
            break
        
        fixed = False
        for i in range(len(pairs)):
            u, v = pairs[i]
            if u == v:
                for j in range(len(pairs)):
                    if j == i: continue
                    a, b = pairs[j]
                    new_u, new_v = u, a
                    new_a, new_b = v, b
                    if new_u != new_v and new_a != new_b:
                        pairs[i] = (new_u, new_v)
                        pairs[j] = (new_a, new_b)
                        fixed = True
                        break
                if fixed: break
            else:
                key = (min(u,v), max(u,v))
                if key in edge_set:
                    for j in range(len(pairs)):
                        if j == i: continue
                        a, b = pairs[j]
                        new_u, new_v = u, a
                        new_a, new_b = v, b
                        if new_u != new_v and new_a != new_b:
                            pairs[i] = (new_u, new_v)
                            pairs[j] = (new_a, new_b)
                            fixed = True
                            break
                    if fixed: break
        if not fixed:
            return None
    
    edge_set = set()
    for u, v in pairs:
        if u == v: return None
        key = (min(u,v), max(u,v))
        if key in edge_set: return None
        edge_set.add(key)
    
    return pairs

def _get_girth_nonbacktracking(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    min_girth = float('inf')
    
    for start in range(n):
        dist = {}
        queue = deque()
        
        for neighbor in adj[start]:
            state = (neighbor, start)
            dist[state] = 1
            queue.append(state)
        
        while queue:
            node, prev = queue.popleft()
            length = dist[(node, prev)]
            
            if length >= min_girth:
                continue
            
            for neighbor in adj[node]:
                if neighbor == prev:
                    continue
                
                new_state = (neighbor, node)
                
                if neighbor == start:
                    cycle_len = length + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                elif new_state not in dist:
                    dist[new_state] = length + 1
                    queue.append(new_state)
        
        if min_girth <= 4:
            break
    
    return min_girth if min_girth != float('inf') else 0
