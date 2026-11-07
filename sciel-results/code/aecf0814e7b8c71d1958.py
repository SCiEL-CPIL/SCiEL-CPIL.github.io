"""
Structural shift: model the problem as a constraint-satisfaction search over
multiplicative orbits (values sharing the same prime-factor pattern up to scaling)
so that products collide heavily. Combine this with simulated-annealing refinement
from the inspiration, plus a small local-search sweep for final polish.
"""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import Counter, defaultdict
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}


def _raw_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _all_edges(n):
    return list(combinations(range(n), 2))


def _make_orbit_values():
    """
    Build 14 values by choosing a small base set of 'orbit representatives'
    and generating multiples that keep the same prime-factor pattern.
    This forces product collisions: if v_i = c * r_i, then v_i * v_j = c^2 * r_i * r_j,
    so products depend only on the pair of representatives.
    """
    # Try several orbit structures
    # Structure 1: powers of 2 (single orbit)
    # Structure 2: two orbits: powers of 2 and powers of 3
    # Structure 3: three orbits: powers of 2, powers of 3, powers of 5
    
    candidates = []
    
    # Orbit 1: powers of 2
    orbit1 = []
    v = 1
    while v <= 10000 and len(orbit1) < 14:
        orbit1.append(v)
        v *= 2
    
    # Orbit 2: powers of 3
    orbit2 = []
    v = 1
    while v <= 10000 and len(orbit2) < 14:
        orbit2.append(v)
        v *= 3
    
    # Orbit 3: powers of 5
    orbit3 = []
    v = 1
    while v <= 10000 and len(orbit3) < 14:
        orbit3.append(v)
        v *= 5
    
    # Combine orbits to get 14 distinct values
    # Try different splits
    splits = [
        (14, 0, 0),  # all powers of 2
        (7, 7, 0),   # half powers of 2, half powers of 3
        (5, 5, 4),   # mixed
        (4, 5, 5),   # mixed
        (6, 4, 4),   # mixed
        (8, 3, 3),   # mostly powers of 2
    ]
    
    for n1, n2, n3 in splits:
        vals = orbit1[:n1] + orbit2[:n2] + orbit3[:n3]
        if len(set(vals)) == 14 and all(0 < v <= 10000 for v in vals):
            candidates.append(sorted(vals))
    
    # Also try small composites with shared factors
    # These create many product collisions
    composites = [2, 4, 6, 8, 12, 16, 18, 24, 32, 36, 48, 64, 72, 96]
    if len(set(composites)) == 14:
        candidates.append(sorted(composites))
    
    # Two-prime structure: 2^a * 3^b
    two_prime = []
    for a in range(0, 8):
        for b in range(0, 8):
            val = (2 ** a) * (3 ** b)
            if 0 < val <= 10000 and val not in two_prime:
                two_prime.append(val)
    two_prime.sort()
    if len(two_prime) >= 14:
        candidates.append(two_prime[:14])
    
    # Divisors of 2520
    base = 2520
    divisors = []
    for d in range(1, int(base**0.5) + 1):
        if base % d == 0:
            divisors.append(d)
            if d != base // d:
                divisors.append(base // d)
    divisors.sort()
    if len(divisors) >= 14:
        candidates.append(divisors[:14])
    
    # Fallback: powers of 2
    if not candidates:
        candidates.append(orbit1[:14])
    
    return candidates


def _sa_optimize(values, all_edges, deadline, num_restarts=8, steps_per_restart=150000):
    """
    Simulated annealing to find 35 edges minimizing max(distinct_sums, distinct_products).
    """
    n = len(values)
    m = P['m']
    
    # Precompute edge attributes
    edge_sums = []
    edge_prods = []
    for i, j in all_edges:
        a, b = values[i], values[j]
        edge_sums.append(a + b)
        edge_prods.append(a * b)
    
    num_total_edges = len(all_edges)
    best_score = float('inf')
    best_edges = None
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        # Random initial set of m edges
        init_indices = random.sample(range(num_total_edges), m)
        current_edges = set(init_indices)
        
        # Counters
        sum_counts = Counter()
        prod_counts = Counter()
        active_sums = set()
        active_prods = set()
        
        for idx in current_edges:
            s = edge_sums[idx]
            p = edge_prods[idx]
            sum_counts[s] += 1
            prod_counts[p] += 1
            active_sums.add(s)
            active_prods.add(p)
        
        current_score = max(len(active_sums), len(active_prods))
        
        # Simulated Annealing
        T = 1.0
        cooling = 0.9995
        min_T = 0.01
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 1:
                break
            
            # Pick a random edge to remove
            e_out = random.choice(list(current_edges))
            
            # Pick a random edge to add (not in current)
            e_in = random.randrange(num_total_edges)
            if e_in in current_edges:
                continue
            
            # Calculate delta score
            s_out = edge_sums[e_out]
            p_out = edge_prods[e_out]
            s_in = edge_sums[e_in]
            p_in = edge_prods[e_in]
            
            # Tentative removal
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                active_sums.discard(s_out)
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                active_prods.discard(p_out)
            
            # Tentative addition
            sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
            if sum_counts[s_in] == 1:
                active_sums.add(s_in)
            prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
            if prod_counts[p_in] == 1:
                active_prods.add(p_in)
            
            new_score = max(len(active_sums), len(active_prods))
            
            if new_score <= current_score:
                # Accept
                current_edges.remove(e_out)
                current_edges.add(e_in)
                current_score = new_score
            else:
                # SA: accept with probability exp(-delta/T)
                if random.random() < math.exp(-(new_score - current_score) / T):
                    current_edges.remove(e_out)
                    current_edges.add(e_in)
                    current_score = new_score
                else:
                    # Revert
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0:
                        active_sums.discard(s_in)
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0:
                        active_prods.discard(p_in)
                    
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    if sum_counts[s_out] == 1:
                        active_sums.add(s_out)
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                    if prod_counts[p_out] == 1:
                        active_prods.add(p_out)
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Update global best
        if current_score < best_score:
            best_score = current_score
            best_edges = [list(all_edges[idx]) for idx in current_edges]
    
    return best_score, best_edges


def _local_search_polish(values, edges, all_edges, deadline, max_iters=50):
    """
    Local search: try swapping edges to improve score.
    """
    n = len(values)
    m = len(edges)
    
    # Precompute edge attributes
    edge_attrs = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_attrs[(u, v)] = (a + b, a * b)
    
    current_edges = [list(e) for e in edges]
    edge_set = set(tuple(e) for e in current_edges)
    
    # Current score state
    sum_counts = Counter()
    prod_counts = Counter()
    for u, v in current_edges:
        s, p = edge_attrs[(u, v)]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    def _current_score():
        n_sums = sum(1 for c in sum_counts.values() if c > 0)
        n_prods = sum(1 for c in prod_counts.values() if c > 0)
        return max(n_sums, n_prods)
    
    current_score = _current_score()
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(m):
            if time.time() > deadline:
                break
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]
            
            for e2 in all_edges:
                if e2 == e1 or e2 in edge_set:
                    continue
                
                s2, p2 = edge_attrs[e2]
                
                # Simulate swap: remove e1, add e2
                old_s_count = sum_counts[s1]
                old_p_count = prod_counts[p1]
                new_s_count = sum_counts.get(s2, 0)
                new_p_count = prod_counts.get(p2, 0)
                
                # After removal
                sum_counts[s1] = old_s_count - 1
                prod_counts[p1] = old_p_count - 1
                if sum_counts[s1] == 0:
                    del sum_counts[s1]
                if prod_counts[p1] == 0:
                    del prod_counts[p1]
                
                # After addition
                sum_counts[s2] = new_s_count + 1
                prod_counts[p2] = new_p_count + 1
                
                new_score = _current_score()
                
                if new_score < current_score:
                    # Accept the swap
                    current_edges[i] = list(e2)
                    edge_set.remove(e1)
                    edge_set.add(e2)
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    sum_counts[s1] = old_s_count
                    prod_counts[p1] = old_p_count
                    sum_counts[s2] = new_s_count
                    if new_s_count == 0:
                        del sum_counts[s2]
                    prod_counts[p2] = new_p_count
                    if new_p_count == 0:
                        del prod_counts[p2]
            
            if improved:
                break
        
        if not improved:
            break
    
    return current_score, current_edges


def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin
    
    all_edges = _all_edges(P['n'])
    
    # Generate candidate value sets using orbit structure
    value_sets = _make_orbit_values()
    
    best_score = float('inf')
    best_edges = None
    best_values = None
    
    # Try each value set with SA + local search
    for values in value_sets:
        if time.time() > deadline - 5:
            break
        
        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue
        
        # SA optimization
        score, edges = _sa_optimize(values, all_edges, deadline, num_restarts=6, steps_per_restart=120000)
        
        if edges is not None and score < best_score:
            best_score = score
            best_edges = [list(e) for e in edges]
            best_values = values[:]
        
        # Local search polish
        if best_edges is not None and time.time() < deadline - 2:
            score2, edges2 = _local_search_polish(values, best_edges, all_edges, deadline, max_iters=40)
            if score2 < best_score:
                best_score = score2
                best_edges = [list(e) for e in edges2]
                best_values = values[:]
    
    # Fallback
    if best_edges is None:
        values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        edges = [list(e) for e in all_edges[:P['m']]]
        best_values = values
        best_edges = edges
        best_score = _raw_score(best_values, best_edges)
    
    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
