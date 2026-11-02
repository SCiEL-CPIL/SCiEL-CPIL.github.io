import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Handle the specific case k=5, g=9 with the voltage lift strategy
    if k == 5 and g == 9:
        result = _voltage_lift_k55()
        if result is not None:
            return result
        # Fallback to LUW if voltage lift fails or times out
        return _luw_construct(k, g)
    
    # General fallback to LUW
    return _luw_construct(k, g)

def _voltage_lift_k55():
    """
    Constructs a 5-regular graph of girth 9 using a voltage lift of K_{5,5} over Z_m.
    Uses local search to find valid voltages.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    # Base graph K_{5,5}
    # U = {0, 1, 2, 3, 4}, V = {5, 6, 7, 8, 9}
    # Edges (u, v) for u in U, v in V
    
    # Generate constraints for non-backtracking closed walks of length 4, 6, 8
    constraints = []
    
    # Length 4 walks: u1-v1-u2-v2-u1
    for u1 in range(5):
        for u2 in range(5):
            if u1 == u2: continue
            for v1 in range(5, 10):
                for v2 in range(5, 10):
                    if v1 == v2: continue
                    e1 = u1 * 5 + (v1 - 5)
                    e2 = u2 * 5 + (v1 - 5)
                    e3 = u2 * 5 + (v2 - 5)
                    e4 = u1 * 5 + (v2 - 5)
                    coeff = {e1: 1, e2: -1, e3: 1, e4: -1}
                    constraints.append(coeff)
                    
    # Length 6 walks: u1-v1-u2-v2-u3-v3-u1
    for u1 in range(5):
        for u2 in range(5):
            if u1 == u2: continue
            for u3 in range(5):
                if u1 == u3 or u2 == u3: continue
                for v1 in range(5, 10):
                    for v2 in range(5, 10):
                        if v1 == v2: continue
                        for v3 in range(5, 10):
                            if v1 == v3 or v2 == v3: continue
                            e1 = u1 * 5 + (v1 - 5)
                            e2 = u2 * 5 + (v1 - 5)
                            e3 = u2 * 5 + (v2 - 5)
                            e4 = u3 * 5 + (v2 - 5)
                            e5 = u3 * 5 + (v3 - 5)
                            e6 = u1 * 5 + (v3 - 5)
                            coeff = {e1: 1, e2: -1, e3: 1, e4: -1, e5: 1, e6: -1}
                            constraints.append(coeff)
                            
    # Length 8 walks: u1-v1-u2-v2-u3-v3-u4-v4-u1
    for u1 in range(5):
        for u2 in range(5):
            if u1 == u2: continue
            for u3 in range(5):
                if u1 == u3 or u2 == u3: continue
                for u4 in range(5):
                    if u1 == u4 or u2 == u4 or u3 == u4: continue
                    for v1 in range(5, 10):
                        for v2 in range(5, 10):
                            if v1 == v2: continue
                            for v3 in range(5, 10):
                                if v1 == v3 or v2 == v3: continue
                                for v4 in range(5, 10):
                                    if v1 == v4 or v2 == v4 or v3 == v4: continue
                                    e1 = u1 * 5 + (v1 - 5)
                                    e2 = u2 * 5 + (v1 - 5)
                                    e3 = u2 * 5 + (v2 - 5)
                                    e4 = u3 * 5 + (v2 - 5)
                                    e5 = u3 * 5 + (v3 - 5)
                                    e6 = u4 * 5 + (v3 - 5)
                                    e7 = u4 * 5 + (v4 - 5)
                                    e8 = u1 * 5 + (v4 - 5)
                                    coeff = {e1: 1, e2: -1, e3: 1, e4: -1, e5: 1, e6: -1, e7: 1, e8: -1}
                                    constraints.append(coeff)
                                    
    # Deduplicate constraints
    unique_constraints = set()
    for c in constraints:
        t = tuple(sorted(c.items()))
        unique_constraints.add(t)
        
    # Build matrix and list of edges per constraint for local search
    n_edges = 25
    n_constraints = len(unique_constraints)
    C = np.zeros((n_constraints, n_edges), dtype=np.int32)
    constraint_edges = [] # List of list of (edge_idx, coeff)
    for i, t in enumerate(unique_constraints):
        for eidx, coeff in t:
            C[i, eidx] = coeff
            constraint_edges.append((eidx, coeff))
        # Store the list of (edge_idx, coeff) for this constraint
        constraint_edges[i] = list(t)
        
    # Search for prime m and voltages
    # Use primes to ensure field properties
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    primes = [m for m in range(43, 200) if is_prime(m)]
    
    for m in primes:
        if time.time() > deadline - 10:
            break
            
        n_vertices = 10 * m
        
        # Local Search
        max_iterations = 50000
        # Initialize with random small values
        voltages = np.random.randint(0, m, size=n_edges, dtype=np.int32)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 2:
                break
                
            sums = C @ voltages
            sums_mod = sums % m
            
            # Find violated constraints (sum == 0)
            violated_indices = np.where(sums_mod == 0)[0]
            
            if len(violated_indices) == 0:
                # Valid assignment found
                edges = []
                for u in range(5):
                    for v in range(5, 10):
                        eidx = u * 5 + (v - 5)
                        vol = int(voltages[eidx])
                        for i in range(m):
                            vid1 = u * m + i
                            vid2 = v * m + ((i + vol) % m)
                            if vid1 < vid2:
                                edges.append((vid1, vid2))
                return n_vertices, edges
            
            # Pick a random violated constraint
            idx = violated_indices[random.randint(0, len(violated_indices) - 1)]
            # Pick a random edge in this constraint
            e_idx, coeff = constraint_edges[idx][random.randint(0, len(constraint_edges[idx]) - 1)]
            
            # We need to change voltages[e_idx] such that the sum for idx becomes non-zero.
            # Current sum is 0.
            # New sum = 0 + coeff * delta.
            # We need coeff * delta != 0 mod m.
            # Since m is prime and coeff != 0, we just need delta != 0.
            delta = random.randint(1, m - 1)
            voltages[e_idx] = (voltages[e_idx] + delta) % m
            
    return None

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction from the seed program."""
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
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
