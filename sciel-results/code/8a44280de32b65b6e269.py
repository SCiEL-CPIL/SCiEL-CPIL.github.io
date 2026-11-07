"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s
    
    n = P['n']
    m = P['m']
    max_val = P['maximum']
    
    all_pairs = list(combinations(range(n), 2))
    n_pairs = len(all_pairs)  # 91
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Initialize
    initial_values = list(range(1, n + 1))
    initial_edges = [list(e) for e in all_pairs[:m]]
    best_score = _score(initial_values, initial_edges)
    best_values = initial_values.copy()
    best_edges = initial_edges.copy()
    
    def _valid_values(vals):
        if len(vals) != n:
            return False
        if len(set(vals)) != n:
            return False
        if not all(isinstance(v, int) and 1 <= v <= max_val for v in vals):
            return False
        return True
    
    def _valid_edges(edges):
        if len(edges) != m:
            return False
        seen = set()
        for u, v in edges:
            if not (isinstance(u, int) and isinstance(v, int)):
                return False
            if not (0 <= u < n and 0 <= v < n):
                return False
            if u == v:
                return False
            t = (min(u, v), max(u, v))
            if t in seen:
                return False
            seen.add(t)
        return True
    
    def gen_arithmetic_values():
        """Generate values as an arithmetic progression to create sum collisions."""
        step = random.choice([1, 2, 3, 4, 5, 6])
        start = random.randint(1, max_val - step * (n - 1))
        vals = [start + i * step for i in range(n)]
        if not _valid_values(vals):
            vals = random.sample(range(1, max_val + 1), n)
        return vals
    
    def gen_common_factor_values():
        """Generate values all sharing a common factor for product collisions."""
        candidates = [12, 24, 36, 48, 60, 72, 84, 96, 120, 144, 180, 240, 360, 720]
        base = random.choice(candidates)
        vals = set()
        attempts = 0
        while len(vals) < n and attempts < 500:
            mult = random.randint(1, max_val // base)
            v = base * mult
            if 1 <= v <= max_val:
                vals.add(v)
            attempts += 1
        while len(vals) < n:
            vals.add(random.randint(1, max_val))
        return list(vals)
    
    def gen_two_factor_values():
        """Generate values as products of two small factors for rich product structure."""
        small_primes = [2, 3, 5, 7, 11]
        vals = set()
        attempts = 0
        while len(vals) < n and attempts < 500:
            a = random.choice(small_primes)
            b = random.choice(small_primes)
            v = a * b * random.randint(1, max_val // (a * b))
            if 1 <= v <= max_val:
                vals.add(v)
            attempts += 1
        while len(vals) < n:
            vals.add(random.randint(1, max_val))
        return list(vals)
    
    def gen_power_values():
        """Generate values as powers times a base."""
        base = random.choice([2, 3, 4, 5])
        vals = set()
        for i in range(n):
            v = base * (2 ** i)
            if v <= max_val:
                vals.add(v)
            else:
                vals.add(base * random.randint(1, max_val // base))
        while len(vals) < n:
            vals.add(random.randint(1, max_val))
        return list(vals)
    
    def gen_random_values():
        return random.sample(range(1, max_val + 1), n)
    
    def gen_edges_for_values(values, n_edges=m):
        """Generate edges that create sum/product collisions using greedy approach."""
        edge_set = set()
        edges = []
        
        sorted_indices = sorted(range(n), key=lambda i: values[i])
        
        # Build candidate edges with scores
        candidates = []
        for i, j in all_pairs:
            a, b = values[i], values[j]
            g = math.gcd(a, b)
            score = g * 10 - (a + b) / 100.0
            candidates.append((score, i, j))
        
        candidates.sort(key=lambda x: -x[0])
        
        for score, i, j in candidates:
            if len(edges) >= n_edges:
                break
            t = (min(i, j), max(i, j))
            if t not in edge_set:
                edge_set.add(t)
                edges.append([i, j])
        
        if len(edges) < n_edges:
            remaining = [e for e in all_pairs if (min(e[0], e[1]), max(e[0], e[1])) not in edge_set]
            random.shuffle(remaining)
            for e in remaining:
                if len(edges) >= n_edges:
                    break
                edges.append(list(e))
                edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        return edges
    
    def local_search_values(values, edges, max_iter=50):
        """Local search to improve values for fixed edges."""
        current_score = _score(values, edges)
        values_set = set(values)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            for i in range(n):
                old_val = values[i]
                values_set.discard(old_val)
                
                candidates = set()
                for _ in range(15):
                    candidates.add(random.randint(1, max_val))
                for v in values:
                    for delta in [-2, -1, 1, 2, -3, 3]:
                        nv = v + delta
                        if 1 <= nv <= max_val:
                            candidates.add(nv)
                for factor in [2, 3, 4, 5, 6, 8, 10, 12]:
                    candidates.add(factor * random.randint(1, max_val // factor))
                
                best_new_val = old_val
                best_new_score = current_score
                for new_val in candidates:
                    if new_val == old_val or new_val in values_set:
                        continue
                    values[i] = new_val
                    values_set.add(new_val)
                    new_score = _score(values, edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_val = new_val
                    values[i] = old_val
                    values_set.discard(new_val)
                
                if best_new_val != old_val:
                    values[i] = best_new_val
                    values_set.discard(old_val)
                    values_set.add(best_new_val)
                    current_score = best_new_score
                    improved = True
                else:
                    values_set.add(old_val)
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    values[i], values[j] = values[j], values[i]
                    new_score = _score(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        values[i], values[j] = values[j], values[i]
                    if time.time() > deadline:
                        return values, current_score
        
        return values, current_score
    
    def local_search_edges(values, edges, max_iter=30):
        """Local search to improve edges for fixed values using targeted candidate generation."""
        current_score = _score(values, edges)
        edge_set = set(tuple(sorted(e)) for e in edges)
        improved = True
        iter_count = 0
        
        # Precompute edge scores based on collision potential
        def edge_collision_score(e):
            u, v = e
            a, b = values[u], values[v]
            # Count how many other edges share the same sum or product
            sum_count = 0
            prod_count = 0
            for other in all_pairs:
                if other == e:
                    continue
                ou, ov = other
                oa, ob = values[ou], values[ov]
                if oa + ob == a + b:
                    sum_count += 1
                if oa * ob == a * b:
                    prod_count += 1
            return sum_count + prod_count
        
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            for idx in range(m):
                if time.time() > deadline:
                    return edges, current_score
                e_remove = tuple(sorted(edges[idx]))
                
                # Generate targeted candidates: prefer edges with high collision scores
                candidate_edges = [e for e in all_pairs if tuple(sorted(e)) not in edge_set]
                
                # Score and sort candidates
                scored_candidates = [(edge_collision_score(e), e) for e in candidate_edges]
                scored_candidates.sort(key=lambda x: -x[0])
                
                # Take top candidates with some randomness
                top_candidates = [e for _, e in scored_candidates[:20]]
                random.shuffle(top_candidates)
                
                best_e_add = None
                best_new_score = current_score
                for e_add in top_candidates:
                    new_edges = edges[:idx] + edges[idx+1:] + [list(e_add)]
                    new_score = _score(values, new_edges)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_e_add = e_add
                
                if best_e_add is not None:
                    edges = edges[:idx] + edges[idx+1:] + [list(best_e_add)]
                    edge_set.discard(e_remove)
                    edge_set.add(tuple(sorted(best_e_add)))
                    current_score = best_new_score
                    improved = True
        
        return edges, current_score
    
    def simulated_annealing(values, edges, max_iter=100):
        """Simulated annealing for joint optimization of values and edges."""
        current_score = _score(values, edges)
        best_score_sa = current_score
        best_values_sa = values.copy()
        best_edges_sa = edges.copy()
        
        temperature = 5.0
        cooling = 0.95
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Random perturbation
            new_values = values.copy()
            new_edges = edges.copy()
            
            # Perturb values
            if random.random() < 0.5:
                i = random.randint(0, n - 1)
                old = new_values[i]
                new_values[i] = random.randint(1, max_val)
                while new_values[i] == old or new_values[i] in set(new_values[:i] + new_values[i+1:]):
                    new_values[i] = random.randint(1, max_val)
            
            # Perturb edges
            if random.random() < 0.5:
                idx = random.randint(0, m - 1)
                edge_set = set(tuple(sorted(e)) for e in new_edges)
                available = [e for e in all_pairs if tuple(sorted(e)) not in edge_set]
                if available:
                    new_edges[idx] = list(random.choice(available))
            
            new_score = _score(new_values, new_edges)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.1)):
                values = new_values
                edges = new_edges
                current_score = new_score
                
                if current_score < best_score_sa:
                    best_score_sa = current_score
                    best_values_sa = values.copy()
                    best_edges_sa = edges.copy()
            
            temperature *= cooling
        
        return best_values_sa, best_edges_sa, best_score_sa
    
    # Run iterations with different strategies
    strategies = [gen_arithmetic_values, gen_common_factor_values, gen_two_factor_values, 
                  gen_power_values, gen_random_values, gen_arithmetic_values, gen_common_factor_values]
    
    for iteration in range(40):
        if time.time() > deadline:
            break
        
        # Choose strategy
        if iteration % 7 == 0:
            cur_values = strategies[iteration % len(strategies)]()
        else:
            cur_values = best_values.copy()
        
        cur_edges = gen_edges_for_values(cur_values) if iteration % 3 == 0 else best_edges.copy()
        
        # Alternate optimization
        for _ in range(10):
            if time.time() > deadline:
                break
            
            cur_values, vscore = local_search_values(cur_values, cur_edges, max_iter=40)
            
            if time.time() > deadline:
                break
            
            cur_edges, escore = local_search_edges(cur_values, cur_edges, max_iter=20)
            
            if time.time() > deadline:
                break
            
            cur_score = _score(cur_values, cur_edges)
            if cur_score < best_score:
                best_score = cur_score
                best_values = cur_values.copy()
                best_edges = cur_edges.copy()
        
        # Occasional simulated annealing
        if iteration % 5 == 0 and time.time() < deadline - 10:
            sa_values, sa_edges, sa_score = simulated_annealing(best_values.copy(), best_edges.copy(), max_iter=50)
            if sa_score < best_score:
                best_score = sa_score
                best_values = sa_values.copy()
                best_edges = sa_edges.copy()
        
        if best_score <= 14:
            break
    
    # Final polish: try more edge swaps
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline:
                break
            best_edges, escore = local_search_edges(best_values, best_edges, max_iter=10)
            new_score = _score(best_values, best_edges)
            if new_score < best_score:
                best_score = new_score
    
    # Final validation
    assert len(best_values) == n
    assert len(set(best_values)) == n
    assert all(isinstance(v, int) and 1 <= v <= max_val for v in best_values)
    assert len(best_edges) == m
    edge_tuples = set()
    for u, v in best_edges:
        assert isinstance(u, int) and isinstance(v, int)
        assert 0 <= u < n and 0 <= v < n
        assert u != v
        t = (min(u, v), max(u, v))
        assert t not in edge_tuples
        edge_tuples.add(t)
    
    return {'values': list(best_values), 'edges': [list(e) for e in best_edges]}
# EVOLVE-BLOCK-END
