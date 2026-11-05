import random
import time
import math
import numpy as np
from collections import deque

def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Petersen Graph
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
    ]
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Generate closed non-backtracking walks of length < g
    # We represent the walk as a coefficient vector over the edges.
    # A walk corresponds to a vector c in Z^15. The constraint is c . v != 0 mod m.
    coeff_vectors = set()
    
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        # BFS state: (current_v, prev_v, length, coeff_vector_list)
        # Using a list of 15 integers for the coefficient vector
        # Start: no edge traversed yet, but we are at start_v
        # To handle the "no previous edge" logic for non-backtracking, 
        # we start by taking the first step from start_v.
        
        # Initialize queue with states after 1 step
        queue = deque()
        for next_v, e_id in adj[start_v]:
            u, v = edges[e_id]
            sign = 1 if u == start_v else -1
            coeffs = [0] * 15
            coeffs[e_id] = sign
            queue.append((next_v, start_v, 1, coeffs))
            
        while queue:
            curr_v, prev_v, length, coeffs = queue.popleft()
            
            if length >= max_walk_len:
                continue
                
            # Check if closed
            if curr_v == start_v and length >= 3:
                coeff_vectors.add(tuple(coeffs))
                
            # Extend
            for next_v, e_id in adj[curr_v]:
                if next_v == prev_v:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_coeffs = coeffs[:]
                new_coeffs[e_id] += sign
                queue.append((next_v, curr_v, length + 1, new_coeffs))
                
    # Normalize constraints to reduce redundancy
    # If c is a solution, -c is also a solution (since we check c.v == 0).
    # We can pick a canonical representative.
    unique_coeffs = set()
    for c in coeff_vectors:
        # Skip zero vector (shouldn't happen for simple cycles, but just in case)
        if all(x == 0 for x in c):
            continue
        
        # Find first non-zero index
        first_non_zero = -1
        for i in range(15):
            if c[i] != 0:
                first_non_zero = i
                break
        
        if first_non_zero == -1:
            continue
            
        # If c[first_non_zero] < 0, negate the whole vector
        if c[first_non_zero] < 0:
            c = tuple(-x for x in c)
        unique_coeffs.add(c)
        
    coeff_list = list(unique_coeffs)
    
    if len(coeff_list) == 0:
        # This shouldn't happen for girth 16, but just in case
        return 10, edges # Invalid girth, but valid structure? No, Petersen has girth 5.
        # Actually, if no constraints, any voltage works. But Petersen itself has girth 5.
        # The constraints MUST catch the 5-cycles.
        # Let's assume coeff_list is not empty.

    # Build numpy matrix for vectorized evaluation
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(coeff_matrix)
    
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

    # Generate primes in ascending order
    # Moore bound 512 => m >= 52. 
    # Smallest prime >= 52 is 53.
    # We want to try small primes first to minimize n = 10*m.
    primes = []
    candidate = 2
    while len(primes) < 100: # Enough primes
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    # Filter primes >= 52
    primes = [p for p in primes if p >= 52]
    
    best_n = float('inf')
    best_edges = None
    best_m = None
    
    # Strategy: Try primes in ascending order.
    # For each prime m, try to find a valid voltage assignment.
    # Use random restarts with local repair.
    
    for m in primes:
        if time.time() > deadline - 5.0:
            break
            
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        # Number of attempts per prime
        # Smaller primes are harder to satisfy (more constraints relative to freedom),
        # but we want them most. 
        # Let's give a generous budget to smaller primes.
        max_attempts = 1000 if m < 100 else 500
        
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2.0:
                break
                
            # Random voltages
            # Ensure connectivity: gcd(v_1, ..., v_m, m) = 1
            # For prime m, any non-zero voltage ensures connectivity if the graph is connected?
            # Actually, for a voltage graph on Z_m, the lift is connected iff the subgroup generated by the voltages is Z_m.
            # Since m is prime, any single non-zero voltage generates Z_m.
            # So we just need to ensure not all voltages are 0.
            # But we also need to avoid 0 voltage on any edge? 
            # If an edge has voltage 0, it's a "vertical" edge. It doesn't break connectivity necessarily.
            # However, if we have a 0 voltage, it might create short cycles? 
            # The constraint check handles cycles.
            # Let's allow 0 voltages? 
            # If v_e = 0, then the edge connects (u, t) to (v, t).
            # If we have a cycle of such edges, it's a cycle in the base graph.
            # The constraints will catch it if the base cycle has length < g.
            # So 0 voltages are allowed.
            
            # But wait, if we use 0 voltages, we might not get a simple graph?
            # The lift of a simple graph is simple if the voltages are distinct?
            # No, the lift of a simple graph is always simple.
            # Vertices are (u, t). Edge (u, v) with voltage w connects (u, t) to (v, t+w).
            # Could we have (u, t) connected to (v, t+w) AND (v, t) connected to (u, t+w')?
            # This would be a multi-edge between { (u,t), (v,t+w) } and { (v,t), (u,t+w') }.
            # Since the base graph is simple, u!=v.
            # The only way to have multiple edges between the same pair of vertices in the lift is if the base graph has multiple edges (it doesn't) or if the mapping is not injective?
            # The mapping (u, t) -> u is a graph homomorphism.
            # If (u, t1) ~ (v, t2) and (u, t1') ~ (v, t2'), and {t1, t2} = {t1', t2'}, then the base edge is the same.
            # Since base edge is unique, the voltage is unique.
            # So (t2 - t1) mod m = w.
            # If t1=t1' and t2=t2', it's the same edge.
            # So the lift is simple.
            
            volts = [random.randint(0, m-1) for _ in range(15)]
            
            # Check if all zero? If all zero, it's m copies of Petersen. Girth 5.
            # Constraints will fail.
            
            # Local Repair Loop
            # We want to satisfy: for all c in coeff_list, c . v != 0 mod m.
            
            valid = False
            # We can do a few steps of repair, or just check random assignment.
            # Given the number of constraints (likely hundreds/thousands), random assignment might fail often.
            # Local repair: pick a violated constraint, change a voltage.
            
            for repair_step in range(50):
                if time.time() > deadline - 2.0:
                    break
                
                # Check validity using numpy
                # sums = (coeff_matrix @ volts) % m
                # violated_indices = np.where(sums == 0)[0]
                
                # Optimization: Only check a subset of constraints? No, must be exact.
                # But we can break early if we find a violation during repair?
                # For the final check, we need all.
                
                # Let's just do a full check every few steps or at the end?
                # Full check is O(N * 15). N ~ 1000. 15000 ops. Very fast.
                
                v_arr = np.array(volts, dtype=np.int64)
                sums = (coeff_matrix @ v_arr) % m
                violated_indices = np.where(sums == 0)[0]
                
                if len(violated_indices) == 0:
                    valid = True
                    break
                
                # Pick a random violated constraint
                viol_idx = random.choice(violated_indices.tolist())
                c = coeff_list[viol_idx]
                
                # Find edges with non-zero coefficient in this constraint
                candidates = [i for i in range(15) if c[i] != 0]
                if not candidates:
                    # Constraint is 0 . v = 0, always violated?
                    # This implies the walk is a zero vector in homology?
                    # In a simple graph, non-backtracking closed walks are not zero vectors unless they are back-and-forth (which is not non-backtracking).
                    # So candidates should not be empty.
                    continue
                    
                j = random.choice(candidates)
                
                # Change volt[j]
                # We need c[j] * new_volt[j] + sum_{i!=j} c[i] * volt[i] != 0
                # Let S = sum_{i!=j} c[i] * volt[i]
                # We need c[j] * v_j != -S
                # Pick a random v_j that satisfies this.
                # Since m is prime and c[j] != 0, there is exactly one forbidden value for v_j.
                # So we pick a random value and if it's the forbidden one, pick another.
                
                S = 0
                for i in range(15):
                    if i != j and c[i] != 0:
                        S += c[i] * volts[i]
                S %= m
                
                forbidden = 0
                if c[j] != 0:
                    # c[j] * v_j == -S  => v_j == -S * inv(c[j])
                    # Since m is prime, we can find inv.
                    # Or just check all values? m is small.
                    # Let's just pick a random value and check.
                    pass
                
                # Simpler: just pick a random new value and check if it fixes THIS constraint.
                # It might break others, but we'll check again.
                old_val = volts[j]
                new_val = random.randint(0, m-1)
                if new_val == old_val:
                    new_val = (new_val + 1) % m
                volts[j] = new_val
                
                # Quick check if this specific constraint is fixed
                # s_new = S + c[j] * new_val
                # if s_new % m != 0: continue (it is fixed)
                # else: try another value?
                # Actually, since we picked random, it has (m-1)/m chance of fixing it.
                # If it doesn't, we just proceed to next repair step.
                
            if valid:
                found = True
                break
        
        if found:
            best_m = m
            best_n = n_candidate
            lift_edges = []
            for i, (u, v) in enumerate(edges):
                vol = volts[i]
                u_edge, v_edge = edges[i]
                sign = 1 if u == u_edge else -1
                for t in range(m):
                    u_lift = u * m + t
                    v_lift = v * m + (t + sign * vol) % m
                    lift_edges.append((u_lift, v_lift))
                    
            best_edges = lift_edges
            # Since we are iterating primes in ascending order, the first found is the smallest n.
            # We can break here?
            # Yes, because we want to minimize n.
            break
            
    if best_edges is None:
        # Fallback to LUW if random lift failed
        return _luw_fallback(k, g)

    return best_n, best_edges

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
