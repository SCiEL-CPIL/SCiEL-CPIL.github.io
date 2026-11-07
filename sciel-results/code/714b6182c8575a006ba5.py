"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    def get_counts(values, edges):
        sum_count = Counter()
        prod_count = Counter()
        for i, j in edges:
            sum_count[values[i] + values[j]] += 1
            prod_count[values[i] * values[j]] += 1
        return sum_count, prod_count
    
    def score_from_counts(sum_count, prod_count):
        return max(len(sum_count), len(prod_count))
    
    def generate_structured_values():
        """Generate values with built-in structure for collisions."""
        strategy = random.choice(['arith', 'multiplicative', 'mixed', 'powers', 'composite'])
        
        if strategy == 'arith':
            # Arithmetic progression with small step
            start = random.randint(1, 50)
            step = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
            vals = [start + i * step for i in range(n)]
            # Ensure distinct and in range
            if len(set(vals)) != n or any(v < 1 or v > P['maximum'] for v in vals):
                vals = list(range(1, n + 1))
            return vals
        
        elif strategy == 'multiplicative':
            # Numbers with many common factors
            base = random.choice([2, 3, 4, 5, 6])
            vals = []
            used = set()
            for _ in range(n):
                for attempt in range(100):
                    v = base * random.randint(1, P['maximum'] // base)
                    if v not in used and 1 <= v <= P['maximum']:
                        vals.append(v)
                        used.add(v)
                        break
                else:
                    vals.append(random.randint(1, P['maximum']))
            while len(set(vals)) < n:
                idx = random.randint(0, n - 1)
                vals[idx] = random.randint(1, P['maximum'])
            return vals
        
        elif strategy == 'mixed':
            # Mix of arithmetic and random
            half = n // 2
            start = random.randint(1, 30)
            step = random.choice([2, 3, 4, 5, 6])
            vals = [start + i * step for i in range(half)]
            for _ in range(n - half):
                vals.append(random.randint(1, P['maximum']))
            while len(set(vals)) < n:
                idx = random.randint(0, n - 1)
                vals[idx] = random.randint(1, P['maximum'])
            return vals
        
        elif strategy == 'powers':
            # Powers of small bases
            bases = [2, 3, 4, 5]
            vals = []
            used = set()
            for b in bases:
                for e in range(1, 20):
                    v = b ** e
                    if v not in used and 1 <= v <= P['maximum']:
                        vals.append(v)
                        used.add(v)
                        if len(vals) >= n:
                            break
                if len(vals) >= n:
                    break
            while len(vals) < n:
                vals.append(random.randint(1, P['maximum']))
            while len(set(vals)) < n:
                idx = random.randint(0, n - 1)
                vals[idx] = random.randint(1, P['maximum'])
            return vals
        
        else:  # composite
            # Highly composite numbers
            composites = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168,
                         180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312,
                         324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456,
                         468, 480, 492, 504, 516, 528, 540, 552, 564, 576, 588, 600]
            random.shuffle(composites)
            vals = composites[:n]
            while len(set(vals)) < n:
                idx = random.randint(0, n - 1)
                vals[idx] = random.randint(1, P['maximum'])
            return vals
    
    def greedy_edges_for_values(values):
        """Select edges to maximize collisions."""
        edge_sums = {}
        edge_prods = {}
        for i, j in all_edges:
            edge_sums[(i, j)] = values[i] + values[j]
            edge_prods[(i, j)] = values[i] * values[j]
        
        sum_freq = Counter(edge_sums.values())
        prod_freq = Counter(edge_prods.values())
        sum_count = Counter()
        prod_count = Counter()
        edges = []
        edge_set = set()
        remaining = set(all_edges)
        
        for _ in range(m):
            best_e = None
            best_key = None
            for e in remaining:
                s = edge_sums[e]
                p = edge_prods[e]
                new_s = 0 if sum_count[s] > 0 else 1
                new_p = 0 if prod_count[p] > 0 else 1
                # Prefer edges that create fewer new distinct values
                key = (new_s + new_p, -sum_freq[s], -prod_freq[p])
                if best_key is None or key < best_key:
                    best_key = key
                    best_e = e
            edges.append(list(best_e))
            edge_set.add(best_e)
            sum_count[edge_sums[best_e]] += 1
            prod_count[edge_prods[best_e]] += 1
            remaining.discard(best_e)
        
        return edges
    
    def simulated_annealing(init_values, init_edges, max_iters, deadline):
        """SA co-optimization of values and edges."""
        values = list(init_values)
        edges = [list(e) for e in init_edges]
        edge_set = set(tuple(e) for e in edges)
        
        sum_count, prod_count = get_counts(values, edges)
        cur_score = score_from_counts(sum_count, prod_count)
        best_score = cur_score
        best_values = list(values)
        best_edges = [list(e) for e in edges]
        
        # Temperature schedule
        T0 = 10.0
        Tend = 0.01
        alpha = (Tend / T0) ** (1.0 / max(max_iters, 1))
        T = T0
        
        # Track stagnation
        stagnation = 0
        max_stagnation = 200
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            T *= alpha
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Value swap
                v1, v2 = random.sample(range(n), 2)
                val1, val2 = values[v1], values[v2]
                if val1 == val2:
                    continue
                
                # Collect affected edges
                affected = []
                for idx, e in enumerate(edges):
                    if v1 in e or v2 in e:
                        affected.append(idx)
                
                # Remove old contributions
                old_data = []
                for idx in affected:
                    i, j = edges[idx]
                    s = values[i] + values[j]
                    p = values[i] * values[j]
                    old_data.append((idx, s, p))
                    sum_count[s] -= 1
                    if sum_count[s] == 0:
                        del sum_count[s]
                    prod_count[p] -= 1
                    if prod_count[p] == 0:
                        del prod_count[p]
                
                # Apply swap
                values[v1], values[v2] = val2, val1
                
                # Add new contributions
                for idx in affected:
                    i, j = edges[idx]
                    new_s = values[i] + values[j]
                    new_p = values[i] * values[j]
                    sum_count[new_s] += 1
                    prod_count[new_p] += 1
                
                new_score = score_from_counts(sum_count, prod_count)
                delta = new_score - cur_score
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(values)
                        best_edges = [list(e) for e in edges]
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    # Undo
                    for idx in affected:
                        i, j = edges[idx]
                        cur_s = values[i] + values[j]
                        cur_p = values[i] * values[j]
                        sum_count[cur_s] -= 1
                        if sum_count[cur_s] == 0:
                            del sum_count[cur_s]
                        prod_count[cur_p] -= 1
                        if prod_count[cur_p] == 0:
                            del prod_count[cur_p]
                    
                    for idx, s, p in old_data:
                        sum_count[s] += 1
                        prod_count[p] += 1
                    
                    values[v1], values[v2] = val1, val2
                    stagnation += 1
            
            elif move_type < 0.65:
                # Value replacement (change one value to a structured candidate)
                v_idx = random.randint(0, n - 1)
                old_val = values[v_idx]
                
                # Generate candidate values that might create collisions
                candidates = []
                # Try values that are sums/differences of existing values
                for other in values:
                    if other != old_val:
                        for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
                            cv = other + delta
                            if 1 <= cv <= P['maximum'] and cv != old_val and cv not in values:
                                candidates.append(cv)
                        # Try multiples
                        for mult in [2, 3, 4]:
                            cv = other * mult
                            if 1 <= cv <= P['maximum'] and cv != old_val and cv not in values:
                                candidates.append(cv)
                
                if not candidates:
                    candidates = [random.randint(1, P['maximum']) for _ in range(10)]
                
                nv = random.choice(candidates)
                if nv == old_val or nv in values:
                    continue
                
                # Collect affected edges
                affected = []
                for idx, e in enumerate(edges):
                    if v_idx in e:
                        affected.append(idx)
                
                old_data = []
                for idx in affected:
                    i, j = edges[idx]
                    s = values[i] + values[j]
                    p = values[i] * values[j]
                    old_data.append((idx, s, p))
                    sum_count[s] -= 1
                    if sum_count[s] == 0:
                        del sum_count[s]
                    prod_count[p] -= 1
                    if prod_count[p] == 0:
                        del prod_count[p]
                
                values[v_idx] = nv
                
                for idx in affected:
                    i, j = edges[idx]
                    new_s = values[i] + values[j]
                    new_p = values[i] * values[j]
                    sum_count[new_s] += 1
                    prod_count[new_p] += 1
                
                new_score = score_from_counts(sum_count, prod_count)
                delta = new_score - cur_score
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(values)
                        best_edges = [list(e) for e in edges]
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    for idx in affected:
                        i, j = edges[idx]
                        cur_s = values[i] + values[j]
                        cur_p = values[i] * values[j]
                        sum_count[cur_s] -= 1
                        if sum_count[cur_s] == 0:
                            del sum_count[cur_s]
                        prod_count[cur_p] -= 1
                        if prod_count[cur_p] == 0:
                            del prod_count[cur_p]
                    
                    for idx, s, p in old_data:
                        sum_count[s] += 1
                        prod_count[p] += 1
                    
                    values[v_idx] = old_val
                    stagnation += 1
            
            else:
                # Edge swap (2-edge swap)
                idx1, idx2 = random.sample(range(m), 2)
                e1 = tuple(edges[idx1])
                e2 = tuple(edges[idx2])
                
                # Find candidate replacement edges
                found = False
                attempts = 0
                ne1 = ne2 = None
                while not found and attempts < 100:
                    c1 = random.choice(all_edges)
                    c2 = random.choice(all_edges)
                    if c1 != c2 and c1 not in edge_set and c2 not in edge_set:
                        ne1, ne2 = c1, c2
                        found = True
                    attempts += 1
                
                if not found:
                    continue
                
                # Remove old edges
                for e in [e1, e2]:
                    s = values[e[0]] + values[e[1]]
                    p = values[e[0]] * values[e[1]]
                    sum_count[s] -= 1
                    if sum_count[s] == 0:
                        del sum_count[s]
                    prod_count[p] -= 1
                    if prod_count[p] == 0:
                        del prod_count[p]
                
                # Add new edges
                for e in [ne1, ne2]:
                    s = values[e[0]] + values[e[1]]
                    p = values[e[0]] * values[e[1]]
                    sum_count[s] += 1
                    prod_count[p] += 1
                
                new_score = score_from_counts(sum_count, prod_count)
                delta = new_score - cur_score
                
                if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                    edges[idx1] = list(ne1)
                    edges[idx2] = list(ne2)
                    edge_set.remove(e1)
                    edge_set.remove(e2)
                    edge_set.add(ne1)
                    edge_set.add(ne2)
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_values = list(values)
                        best_edges = [list(e) for e in edges]
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    # Undo
                    for e in [ne1, ne2]:
                        s = values[e[0]] + values[e[1]]
                        p = values[e[0]] * values[e[1]]
                        sum_count[s] -= 1
                        if sum_count[s] == 0:
                            del sum_count[s]
                        prod_count[p] -= 1
                        if prod_count[p] == 0:
                            del prod_count[p]
                    
                    for e in [e1, e2]:
                        s = values[e[0]] + values[e[1]]
                        p = values[e[0]] * values[e[1]]
                        sum_count[s] += 1
                        prod_count[p] += 1
                    
                    stagnation += 1
            
            # Restart if stuck
            if stagnation > max_stagnation:
                T = T0 * 0.5
                stagnation = 0
        
        return best_values, best_edges, best_score
    
    # Multiple restarts with different initial structures
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Generate diverse initial solutions
    initial_solutions = []
    
    # Structured initial values
    for _ in range(8):
        vals = generate_structured_values()
        if len(set(vals)) == n and all(1 <= v <= P['maximum'] for v in vals):
            edges = greedy_edges_for_values(vals)
            initial_solutions.append((vals, edges))
    
    # Simple sequential
    initial_solutions.append((list(range(1, n + 1)), greedy_edges_for_values(list(range(1, n + 1)))))
    
    # Random
    for _ in range(3):
        vals = [random.randint(1, P['maximum']) for _ in range(n)]
        while len(set(vals)) < n:
            idx = random.randint(0, n - 1)
            vals[idx] = random.randint(1, P['maximum'])
        edges = greedy_edges_for_values(vals)
        initial_solutions.append((vals, edges))
    
    # Run SA on each initial solution
    for init_vals, init_edges in initial_solutions:
        if time.time() > deadline - 5:
            break
        
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        max_iters = min(3000, int(remaining * 100))
        if max_iters < 100:
            break
        
        vals, edges, score = simulated_annealing(init_vals, init_edges, max_iters, deadline)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = vals
            best_overall_edges = edges
    
    # If we still have time, do more SA runs from the best solution
    if best_overall_values is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution
            vals = list(best_overall_values)
            edges = [list(e) for e in best_overall_edges]
            
            # Random value perturbations
            for _ in range(random.randint(2, 4)):
                idx = random.randint(0, n - 1)
                old = vals[idx]
                nv = random.randint(1, P['maximum'])
                while nv == old or nv in vals:
                    nv = random.randint(1, P['maximum'])
                vals[idx] = nv
            
            # Random edge perturbations
            for _ in range(random.randint(2, 3)):
                idx = random.randint(0, m - 1)
                edge_set = set(tuple(e) for e in edges)
                for _ in range(50):
                    new_e = random.choice(all_edges)
                    if new_e not in edge_set:
                        edges[idx] = list(new_e)
                        edge_set.add(new_e)
                        break
            
            if len(set(vals)) != n:
                continue
            
            remaining = deadline - time.time()
            max_iters = min(2000, int(remaining * 100))
            if max_iters < 100:
                break
            
            nv, ne, ns = simulated_annealing(vals, edges, max_iters, deadline)
            if ns < best_overall_score:
                best_overall_score = ns
                best_overall_values = nv
                best_overall_edges = ne
    
    # Fallback
    if best_overall_values is None:
        best_overall_values = list(range(1, n + 1))
        best_overall_edges = greedy_edges_for_values(best_overall_values)
        best_overall_score = evaluate(best_overall_values, best_overall_edges)
    
    final_values = [int(v) for v in best_overall_values]
    final_edges = [[int(i), int(j)] for i, j in best_overall_edges]
    
    # Validation
    assert len(final_values) == n
    assert len(set(final_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= P['maximum'] for v in final_values)
    assert len(final_edges) == m
    edge_tuples = set()
    for e in final_edges:
        i, j = e
        assert 0 <= i < n and 0 <= j < n and i != j
        t = tuple(sorted((i, j)))
        assert t not in edge_tuples
        edge_tuples.add(t)
    
    return {'values': final_values, 'edges': final_edges}

# EVOLVE-BLOCK-END
