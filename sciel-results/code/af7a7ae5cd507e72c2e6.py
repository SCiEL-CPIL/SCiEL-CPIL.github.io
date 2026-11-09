"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations
from collections import defaultdict, Counter

def _score(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = set()
    products = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        products.add(p)
    return max(len(sums), len(products))

def _collision_density(values):
    """Count how many sum/product pairs collide among all pairs of values."""
    n = len(values)
    sum_counts = Counter()
    prod_counts = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            sum_counts[values[i] + values[j]] += 1
            prod_counts[values[i] * values[j]] += 1
    sum_collisions = sum(c * (c - 1) // 2 for c in sum_counts.values())
    prod_collisions = sum(c * (c - 1) // 2 for c in prod_counts.values())
    return sum_collisions + prod_collisions

def _greedy_value_selection(candidate_pool, k=14, deadline=None):
    """
    Greedily select k values from candidate_pool that maximize collision density.
    Uses a collision graph where edge weight = number of shared sum/product relations.
    """
    if deadline is not None and time.time() > deadline:
        return candidate_pool[:k]
    
    n_pool = len(candidate_pool)
    if n_pool <= k:
        return list(candidate_pool)
    
    # Precompute pairwise collision contributions
    # For each pair (i,j), compute how many other pairs share their sum or product
    pair_sum = [[0] * n_pool for _ in range(n_pool)]
    pair_prod = [[0] * n_pool for _ in range(n_pool)]
    pair_sum_count = [[0] * n_pool for _ in range(n_pool)]
    pair_prod_count = [[0] * n_pool for _ in range(n_pool)]
    
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            s = candidate_pool[i] + candidate_pool[j]
            p = candidate_pool[i] * candidate_pool[j]
            pair_sum[i][j] = s
            pair_sum[j][i] = s
            pair_prod[i][j] = p
            pair_prod[j][i] = p
    
    # Count how many pairs have each sum/product
    sum_freq = Counter()
    prod_freq = Counter()
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            sum_freq[pair_sum[i][j]] += 1
            prod_freq[pair_prod[i][j]] += 1
    
    # Collision contribution of pair (i,j) = freq(sum[i,j]) + freq(prod[i,j])
    pair_collision = [[0] * n_pool for _ in range(n_pool)]
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            c = sum_freq[pair_sum[i][j]] + prod_freq[pair_prod[i][j]]
            pair_collision[i][j] = c
            pair_collision[j][i] = c
    
    # Greedy selection: start with the pair having highest collision, expand
    selected = []
    remaining = set(range(n_pool))
    
    # Find best starting pair
    best_pair_score = -1
    best_pair = (0, 1)
    for i in range(min(n_pool, 50)):
        for j in range(i + 1, min(n_pool, 50)):
            if pair_collision[i][j] > best_pair_score:
                best_pair_score = pair_collision[i][j]
                best_pair = (i, j)
    
    selected.append(best_pair[0])
    selected.append(best_pair[1])
    remaining.discard(best_pair[0])
    remaining.discard(best_pair[1])
    
    while len(selected) < k and remaining:
        if deadline is not None and time.time() > deadline:
            break
        
        # For each candidate in remaining, compute marginal gain
        best_candidate = None
        best_gain = -1
        
        for cand in remaining:
            gain = 0
            for sel in selected:
                gain += pair_collision[cand][sel]
            
            if gain > best_gain:
                best_gain = gain
                best_candidate = cand
        
        if best_candidate is None:
            break
        
        selected.append(best_candidate)
        remaining.discard(best_candidate)
    
    # If we still need more, add from remaining
    while len(selected) < k and remaining:
        selected.append(random.choice(list(remaining)))
        remaining.discard(selected[-1])
    
    result = [candidate_pool[i] for i in selected]
    result.sort()
    return result

def _build_candidate_pool(deadline=None):
    """
    Build a diverse candidate pool of integers <= 10000 with rich multiplicative/additive structure.
    """
    if deadline is not None and time.time() > deadline:
        return list(range(1, 101))
    
    pool = set()
    
    # Small integers (rich in additive collisions)
    for i in range(1, 101):
        pool.add(i)
    
    # Powers of 2 (rich in multiplicative structure)
    for i in range(1, 14):
        pool.add(2 ** i)
    
    # Powers of 3
    for i in range(1, 9):
        pool.add(3 ** i)
    
    # Highly composite numbers and their divisors
    for N in [12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 360, 480, 720, 840, 1200, 1680, 2400, 3360, 4200, 5040]:
        for i in range(1, int(N ** 0.5) + 1):
            if N % i == 0:
                if i <= 10000:
                    pool.add(i)
                if N // i <= 10000 and N // i != i:
                    pool.add(N // i)
    
    # Products of small primes
    small_primes = [2, 3, 5, 7, 11, 13]
    for p1 in small_primes:
        for p2 in small_primes:
            if p1 <= p2 and p1 * p2 <= 10000:
                pool.add(p1 * p2)
    
    # Arithmetic progressions
    for start in range(1, 20):
        for step in range(1, 10):
            for i in range(14):
                v = start + i * step
                if v <= 10000:
                    pool.add(v)
    
    # Geometric-like sequences
    for base in range(2, 20):
        for i in range(1, 8):
            v = base ** i
            if v <= 10000:
                pool.add(v)
    
    # Multiples of small numbers
    for base in range(1, 30):
        for mult in range(1, 341):
            v = base * mult
            if v <= 10000:
                pool.add(v)
    
    pool = sorted(pool)
    
    # Limit pool size for tractability
    if len(pool) > 500:
        # Keep a diverse subset
        random.seed(42)
        # Always keep small numbers and powers
        keep = set()
        for v in pool:
            if v <= 100:
                keep.add(v)
            # Keep powers of 2, 3
            x = v
            is_power = False
            for base in [2, 3, 5]:
                y = v
                while y % base == 0 and y > 1:
                    y //= base
                if y == 1:
                    is_power = True
                    break
            if is_power and v <= 10000:
                keep.add(v)
            # Keep highly composite
            for N in [12, 24, 36, 48, 60, 72, 84, 96, 120, 180, 240, 360, 480, 720, 840, 1200, 1680, 2400, 3360, 4200, 5040]:
                if N % v == 0 or v % N == 0:
                    keep.add(v)
                    break
        if len(keep) >= 200:
            pool = sorted(keep)
        else:
            pool = pool[:500]
    
    return pool

def _greedy_edge_selection(values, m=35, deadline=None):
    """Select m edges greedily to maximize sum/product collisions."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sum and product for each edge
    edge_data = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data[(i, j)] = (s, p)
    
    selected_edges = []
    sum_counts = Counter()
    prod_counts = Counter()
    
    remaining = list(all_edges)
    
    for _ in range(m):
        if deadline is not None and time.time() > deadline:
            break
        
        best_edge = None
        best_gain = -1
        
        for edge in remaining:
            i, j = edge
            s, p = edge_data[edge]
            
            gain = 0
            if s in sum_counts:
                gain += sum_counts[s]
            if p in prod_counts:
                gain += prod_counts[p]
            
            if gain > best_gain:
                best_gain = gain
                best_edge = edge
        
        if best_edge is None:
            break
        
        selected_edges.append(best_edge)
        s, p = edge_data[best_edge]
        sum_counts[s] += 1
        prod_counts[p] += 1
        remaining.remove(best_edge)
    
    return [list(e) for e in selected_edges]

def _local_search_edges(values, edges, deadline=None, max_iters=5000):
    """Local search to improve edge selection."""
    n = len(values)
    m = len(edges)
    all_edges = list(combinations(range(n), 2))
    edge_set = set(tuple(sorted(e)) for e in edges)
    
    current_score = _score(values, edges)
    best_edges = [list(e) for e in edges]
    best_score = current_score
    
    # Precompute all edge data
    edge_data = {}
    for i, j in all_edges:
        edge_data[(i, j)] = (values[i] + values[i], values[i] * values[j])
    # Fix: recompute properly
    edge_data = {}
    for i, j in all_edges:
        edge_data[(i, j)] = (values[i] + values[j], values[i] * values[j])
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        if not edges:
            break
        
        idx = random.randint(0, len(edges) - 1)
        old_edge = tuple(sorted(edges[idx]))
        
        candidates = [e for e in all_edges if e not in edge_set]
        if not candidates:
            break
        
        new_edge = random.choice(candidates)
        
        edges[idx] = list(new_edge)
        edge_set.discard(old_edge)
        edge_set.add(new_edge)
        
        new_score = _score(values, edges)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_edges = [list(e) for e in edges]
        else:
            edges[idx] = list(old_edge)
            edge_set.discard(new_edge)
            edge_set.add(old_edge)
    
    return best_edges, best_score

def _local_search_values(values, edges, deadline=None, max_iters=3000):
    """Local search to improve values while keeping edges."""
    n = len(values)
    current_score = _score(values, edges)
    best_values = list(values)
    best_score = current_score
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb a random value
        idx = random.randint(0, n - 1)
        old_val = values[idx]
        
        # Try nearby values
        delta = random.choice([-3, -2, -1, 1, 2, 3, -5, 5, -10, 10])
        new_val = old_val + delta
        if new_val < 1 or new_val > 10000:
            continue
        
        # Check distinctness
        if new_val in values and new_val != old_val:
            continue
        
        values[idx] = new_val
        new_score = _score(values, edges)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_values = list(values)
        else:
            values[idx] = old_val
    
    return best_values, best_score

def construct():
    n = 14
    m = 35
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    # Build candidate pool
    pool = _build_candidate_pool(deadline)
    
    # Strategy 1: Collision-density-based value selection
    if time.time() < deadline - 10:
        values = _greedy_value_selection(pool, 14, deadline)
        if len(set(values)) != 14 or max(values) > 10000:
            values = sorted(random.sample([v for v in pool if 1 <= v <= 10000], min(14, len(pool))))
        
        edges = _greedy_edge_selection(values, m, deadline)
        if len(edges) < m:
            all_edges = list(combinations(range(n), 2))
            edge_set = set(tuple(sorted(e)) for e in edges)
            for e in all_edges:
                if len(edges) >= m:
                    break
                if e not in edge_set:
                    edges.append(list(e))
                    edge_set.add(e)
        
        score = _score(values, edges)
        
        if time.time() < deadline - 5:
            improved_edges, improved_score = _local_search_edges(values, edges, deadline)
            if improved_score < score:
                score = improved_score
                edges = improved_edges
        
        if time.time() < deadline - 3:
            improved_values, improved_score = _local_search_values(values, edges, deadline)
            if improved_score < score:
                score = improved_score
                values = improved_values
        
        if score < best_score:
            best_score = score
            best_values = values[:]
            best_edges = edges[:]
    
    # Strategy 2: Try multiple random subsets from pool
    if time.time() < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            subset = random.sample(pool, min(14, len(pool)))
            subset.sort()
            
            edges = _greedy_edge_selection(subset, m, deadline)
            if len(edges) < m:
                all_edges = list(combinations(range(n), 2))
                edge_set = set(tuple(sorted(e)) for e in edges)
                for e in all_edges:
                    if len(edges) >= m:
                        break
                    if e not in edge_set:
                        edges.append(list(e))
                        edge_set.add(e)
            
            score = _score(subset, edges)
            
            if time.time() < deadline - 5:
                improved_edges, improved_score = _local_search_edges(subset, edges, deadline)
                if improved_score < score:
                    score = improved_score
                    edges = improved_edges
            
            if time.time() < deadline - 3:
                improved_values, improved_score = _local_search_values(subset, edges, deadline)
                if improved_score < score:
                    score = improved_score
                    subset = improved_values
            
            if score < best_score:
                best_score = score
                best_values = subset[:]
                best_edges = edges[:]
    
    # Strategy 3: Divisor-based values (from original)
    if time.time() < deadline - 10:
        for N in [5040, 7560, 10080, 15120, 20160, 25200, 27720]:
            if time.time() > deadline - 5:
                break
            
            divisors = []
            for i in range(1, int(N ** 0.5) + 1):
                if N % i == 0:
                    if i <= 10000:
                        divisors.append(i)
                    if N // i <= 10000 and N // i != i:
                        divisors.append(N // i)
            divisors = sorted(set(divisors))
            
            if len(divisors) >= 14:
                values = divisors[:14]
                if max(values) > 10000:
                    values = [d for d in divisors if d <= 10000][:14]
                
                if len(values) == 14 and max(values) <= 10000:
                    edges = _greedy_edge_selection(values, m, deadline)
                    if len(edges) < m:
                        all_edges = list(combinations(range(n), 2))
                        edge_set = set(tuple(sorted(e)) for e in edges)
                        for e in all_edges:
                            if len(edges) >= m:
                                break
                            if e not in edge_set:
                                edges.append(list(e))
                                edge_set.add(e)
                    
                    score = _score(values, edges)
                    
                    if time.time() < deadline - 3:
                        improved_edges, improved_score = _local_search_edges(values, edges, deadline)
                        if improved_score < score:
                            score = improved_score
                            edges = improved_edges
                    
                    if score < best_score:
                        best_score = score
                        best_values = values[:]
                        best_edges = edges[:]
    
    # Final fallback
    if best_values is None:
        best_values = list(range(1, 15))
        best_edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
