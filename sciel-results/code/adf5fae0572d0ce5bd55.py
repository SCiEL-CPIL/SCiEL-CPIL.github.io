import random
import time
import math
import numpy as np
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 140.0
    
    def remaining():
        return deadline - time.time()
    
    # Petersen Graph as base
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Generate all closed non-backtracking walks of length [3, g-1]
    max_L = g - 1
    walk_vectors = set()
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        while stack:
            current, prev, length, counts = stack.pop()
            if length >= max_L:
                continue
            for next_v in adj[current]:
                if next_v == prev:
                    continue
                e_idx = edge_index[(current, next_v)]
                u, v = pet_edges[e_idx]
                d = 1 if current == u else -1
                
                new_counts = counts.copy()
                new_counts[e_idx] += d
                new_l = length + 1
                
                if next_v == start_v and new_l >= 3:
                    walk_vectors.add(tuple(new_counts))
                
                stack.append((next_v, current, new_l, new_counts))
    
    if not walk_vectors:
        return _luw_fallback(k, g)
    
    # Filter out zero vectors and canonicalize
    unique_constraints = set()
    for c in walk_vectors:
        if all(x == 0 for x in c):
            continue
        first_non_zero = -1
        for i in range(15):
            if c[i] != 0:
                first_non_zero = i
                break
        if first_non_zero != -1:
            if c[first_non_zero] < 0:
                c = tuple(-x for x in c)
            unique_constraints.add(c)
    
    C = np.array(list(unique_constraints), dtype=np.int64)
    if C.size == 0:
        return _luw_fallback(k, g)
    
    num_constraints = len(C)
    
    # Girth verification
    def check_girth(n, edges, min_girth, time_limit):
        adj_local = [[] for _ in range(n)]
        for i, j in edges:
            adj_local[i].append(j)
            adj_local[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj_local[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < min_girth:
                return min_cycle
        return min_cycle
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Voltage search with simulated annealing + local repair
    def solve_voltage(m, max_iters=8000, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        v = rng.randint(1, m, size=15).astype(np.int64)
        
        def cost(v):
            residues = np.mod(C @ v, m)
            return int(np.sum(residues == 0))
        
        best_v = v.copy()
        best_cost = cost(v)
        
        if best_cost == 0:
            return v, 0
        
        T = 1.0
        cooling = 0.99995
        min_T = 0.005
        
        for it in range(max_iters):
            if remaining() < 5:
                break
            
            # Pick a random variable to change
            j = rng.randint(0, 14)
            old_val = v[j]
            v[j] = rng.randint(1, m - 1)
            
            new_cost = cost(v)
            delta = new_cost - best_cost
            
            if delta < 0 or (T > min_T and random.random() < math.exp(-delta / T)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_v = v.copy()
                if new_cost == 0:
                    return v, 0
            else:
                v[j] = old_val
            
            T *= cooling
        
        # Local repair: try to fix violated constraints
        if best_cost > 0:
            for repair_iter in range(200):
                if remaining() < 3:
                    break
                residues = np.mod(C @ best_v, m)
                violated = np.where(residues == 0)[0]
                if len(violated) == 0:
                    return best_v, 0
                # Pick a random violated constraint
                vi = violated[rng.randint(0, len(violated) - 1)]
                row = C[vi]
                # Try changing one variable in this row to fix it
                nonzeros = np.where(row != 0)[0]
                if len(nonzers) == 0:
                    continue
                j = nonzeros[rng.randint(0, len(nonzers) - 1)]
                old_val = best_v[j]
                # Try a few random values
                for _ in range(5):
                    new_val = rng.randint(1, m - 1)
                    best_v[j] = new_val
                    new_res = np.mod(C @ best_v, m)
                    new_cost = int(np.sum(new_res == 0))
                    if new_cost < best_cost:
                        best_cost = new_cost
                        if new_cost == 0:
                            return best_v, 0
                    else:
                        best_v[j] = old_val
                        break
                else:
                    best_v[j] = old_val
            
            return best_v, best_cost
        
        return best_v, best_cost
    
    best_n = float('inf')
    best_edges = None
    
    # Try primes in increasing order
    m = 53
    primes_tried = 0
    while m < 2000 and remaining() > 25:
        if is_prime(m):
            primes_tried += 1
            # More restarts for smaller m
            num_restarts = min(30, max(5, 100 // primes_tried))
            
            found = False
            for restart in range(num_restarts):
                if remaining() < 15:
                    break
                
                v, cost = solve_voltage(m, max_iters=5000, seed=restart * 1000 + m)
                
                if cost == 0:
                    # Construct graph
                    n = 10 * m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = int(v[e_idx])
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))
                    
                    # Verify
                    gc = check_girth(n, edges, g, deadline - 3)
                    if gc >= g:
                        best_n = n
                        best_edges = edges
                        found = True
                        break
            
            if found:
                break
        
        m += 1
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback: try composite m values with more restarts
    for m in range(52, 150):
        if remaining() < 15:
            break
        if is_prime(m):
            continue
        
        for restart in range(10):
            if remaining() < 10:
                break
            v, cost = solve_voltage(m, max_iters=3000, seed=restart * 7 + m)
            if cost == 0:
                n = 10 * m
                edges = []
                for u, v_base in pet_edges:
                    e_idx = edge_index[(u, v_base)]
                    vol = int(v[e_idx])
                    for layer in range(m):
                        u_lift = u * m + layer
                        v_lift = v_base * m + (layer + vol) % m
                        if u_lift < v_lift:
                            edges.append((u_lift, v_lift))
                        else:
                            edges.append((v_lift, u_lift))
                
                gc = check_girth(n, edges, g, deadline - 3)
                if gc >= g:
                    return n, edges
    
    return _luw_fallback(k, g)

def _luw_fallback(k, g):
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw

if __name__ == "__main__":
    n, e = construct(3, 16)
    print(f"n={n}")
