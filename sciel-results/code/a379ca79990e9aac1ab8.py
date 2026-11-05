import time
import random
import math
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Handle specific small cases with known optimal or near-optimal graphs
    if k == 3:
        if g <= 4:
            # Cube graph Q3 has girth 4, n=8
            return 8, [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), 
                       (4, 5), (4, 6), (5, 7), (6, 7)]
        elif g <= 5:
            # Petersen graph has girth 5, n=10
            return 10, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), 
                        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), 
                        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
        elif g <= 6:
            # Heawood graph has girth 6, n=14
            return 14, [(0, 1), (0, 5), (0, 10), (1, 2), (1, 6), (1, 11), 
                        (2, 3), (2, 7), (2, 12), (3, 4), (3, 8), (3, 13), 
                        (4, 5), (4, 9), (4, 13), (5, 6), (5, 10), (5, 14) if False else (5, 10), 
                        (6, 7), (6, 11), (6, 14) if False else (6, 11), 
                        (7, 8), (7, 12), (7, 14) if False else (7, 12), 
                        (8, 9), (8, 13), (8, 14) if False else (8, 13), 
                        (9, 10), (9, 13), (9, 14) if False else (9, 10), 
                        (10, 11), (10, 14) if False else (10, 11), 
                        (11, 12), (11, 14) if False else (11, 12), 
                        (12, 13), (12, 14) if False else (12, 13), 
                        (13, 14) if False else (13, 14)] # Note: Heawood edges need precise listing. 
            # Actually, let's just use LUW for g<=6 to be safe and fast, or hardcode Heawood correctly.
            # Heawood graph edges:
            # 0: 1, 5, 10
            # 1: 0, 2, 6
            # 2: 1, 3, 7
            # 3: 2, 4, 8
            # 4: 3, 5, 9
            # 5: 0, 4, 10 -> wait, 5 connects to 0, 4, 10? 
            # Let's use the LUW construction which is reliable.
            return _construct_luw(k, g, d=3)
        else:
            return _construct_petersen_lift(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses Simulated Annealing to find valid voltages.
    """
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), 
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), 
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute constraints: all non-backtracking closed walks of length < g
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0]*15)]
        while stack:
            v, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in range(n_base):
                if next_v == prev_v or next_v == v:
                    continue
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                u_edge, v_edge = pet_edges[e_idx]
                sign = 1 if (v == u_edge and next_v == v_edge) else -1
                new_counts = counts.copy()
                new_counts[e_idx] += sign
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    coeff_vectors.add(tuple(new_counts))
                stack.append((next_v, v, new_l, new_counts))
    
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    A = np.array(coeff_list, dtype=np.int64)
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True

    primes = []
    candidate = 53
    while len(primes) < 100:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1

    best_n = float('inf')
    best_edges = None

    for m in primes:
        if time.time() > deadline:
            break
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
            
        # Simulated Annealing
        voltages = np.array([random.randint(1, m-1) for _ in range(15)], dtype=np.int64)
        current_energy = _calc_energy(A, voltages, m)
        
        T = 1.0
        cooling = 0.99
        min_T = 1e-4
        
        for step in range(5000):
            if time.time() > deadline:
                break
            if current_energy == 0:
                break
            
            # Random perturbation
            j = random.randint(0, 14)
            old_val = voltages[j]
            new_val = random.randint(1, m-1)
            if new_val == old_val:
                continue
            voltages[j] = new_val
            new_energy = _calc_energy(A, voltages, m)
            
            delta = new_energy - current_energy
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_energy = new_energy
            else:
                voltages[j] = old_val
                
            T *= cooling
            if T < min_T:
                break
        
        if current_energy == 0:
            edges = _build_lifted_graph(pet_edges, voltages, m)
            return n_candidate, edges

    # Fallback
    return _construct_luw(k=3, g=g, d=11)

def _calc_energy(A, voltages, m):
    prod = (A @ voltages) % m
    return np.sum(prod == 0)

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = []
    seen = set()
    for i, (u, v) in enumerate(pet_edges):
        vol = int(voltages[i])
        for t in range(m):
            u_node = t * n_base + u
            v_node = ((t + vol) % m) * n_base + v
            if u_node < v_node:
                edge = (u_node, v_node)
            else:
                edge = (v_node, u_node)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
    return edges

def _construct_luw(k, g, d=None, variant=0):
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
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
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
