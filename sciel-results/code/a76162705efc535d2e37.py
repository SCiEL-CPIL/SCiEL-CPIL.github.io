"""Search-based program: minimize max(distinct_sums, distinct_products) over 35 edges of 14 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_score(values, edges):
    """Compute the raw score: max(#distinct sums, #distinct products) over edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _all_edges(n):
    """Generate all C(n,2) possible edges."""
    return list(itertools.combinations(range(n), 2))

def _try_complete_graph(values, all_edges):
    """Try all possible subsets of m edges from the complete graph (too many, so use heuristics)."""
    pass

def _greedy_edge_selection(values, target_m):
    """Greedy: pick edges one by one to minimize growth of distinct sums/products."""
    n = len(values)
    all_e = list(itertools.combinations(range(n), 2))
    random.shuffle(all_e)
    
    edges = []
    sums = set()
    prods = set()
    
    for e in all_e:
        if len(edges) >= target_m:
            break
        u, v = e
        s = values[u] + values[v]
        p = values[u] * values[v]
        new_sums = len(sums | {s})
        new_prods = len(prods | {p})
        # Prefer edges that add fewer new distinct values
        cost = max(new_sums, new_prods)
        if not edges or cost <= max(len(sums), len(prods)):
            edges.append(e)
            sums.add(s)
            prods.add(p)
    
    # If we don't have enough edges, fill in with remaining
    if len(edges) < target_m:
        existing = set(edges)
        for e in all_e:
            if e not in existing:
                edges.append(e)
                if len(edges) >= target_m:
                    break
    
    return edges[:target_m]

def _random_search(values, target_m, iterations=2000):
    """Randomly sample edge subsets and keep the best."""
    n = len(values)
    all_e = list(itertools.combinations(range(n), 2))
    best_edges = None
    best_score = float('inf')
    
    for _ in range(iterations):
        edges = random.sample(all_e, target_m)
        score = _raw_score(values, edges)
        if score < best_score:
            best_score = score
            best_edges = edges
            if best_score <= 10:
                break
    
    return best_edges, best_score

def _optimize_values_with_edges(edges, max_iter=5000):
    """Given a fixed edge set, try to find values that minimize the score."""
    n = len(edges[0][0]) if edges else 14
    n = 14  # fixed
    best_vals = None
    best_score = float('inf')
    
    for _ in range(max_iter):
        vals = [random.randint(1, P['maximum']) for _ in range(n)]
        score = _raw_score(vals, edges)
        if score < best_score:
            best_score = score
            best_vals = vals[:]
            if best_score <= 8:
                break
    
    return best_vals, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Leave 15s safety margin
    
    n = P['n']
    m = P['m']
    all_e = list(itertools.combinations(range(n), 2))
    
    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Try structured value sets with greedy edge selection
    value_sets = []
    
    # Small consecutive values
    value_sets.append(list(range(1, n + 1)))
    # Powers of 2
    value_sets.append([2**i for i in range(n)])
    # Fibonacci-like
    fib = [1, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    value_sets.append(fib[:n])
    # Primes
    primes = []
    x = 2
    while len(primes) < n:
        is_p = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
        x += 1
    value_sets.append(primes[:n])
    
    # Random value sets
    for _ in range(20):
        value_sets.append(sorted(random.sample(range(1, P['maximum'] + 1), n)))
    
    for vals in value_sets:
        if time.time() > deadline:
            break
        if len(set(vals)) < n:
            continue
        if max(vals) > P['maximum']:
            continue
        
        # Greedy edge selection
        edges = _greedy_edge_selection(vals, m)
        score = _raw_score(vals, edges)
        if score < best_score:
            best_score = score
            best_result = {'values': vals[:], 'edges': [list(e) for e in edges]}
        
        # Also try random edge selection with this value set
        if time.time() > deadline:
            break
        r_edges, r_score = _random_search(vals, m, iterations=500)
        if r_score < best_score:
            best_score = r_score
            best_result = {'values': vals[:], 'edges': [list(e) for e in r_edges]}
    
    # Strategy 2: Joint optimization - random values + random edges
    for _ in range(100):
        if time.time() > deadline:
            break
        vals = sorted(random.sample(range(1, P['maximum'] + 1), n))
        edges = random.sample(all_e, m)
        score = _raw_score(vals, edges)
        if score < best_score:
            best_score = score
            best_result = {'values': vals[:], 'edges': [list(e) for e in edges]}
    
    # Strategy 3: Local search on the best result
    if best_result and time.time() < deadline - 5:
        vals = best_result['values'][:]
        edges = [tuple(e) for e in best_result['edges']]
        cur_score = best_score
        
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Try perturbing values
            new_vals = vals[:]
            idx = random.randint(0, n - 1)
            new_vals[idx] = random.randint(1, P['maximum'])
            if len(set(new_vals)) == n and max(new_vals) <= P['maximum']:
                s = _raw_score(new_vals, edges)
                if s < cur_score:
                    vals = new_vals
                    cur_score = s
            
            # Try swapping an edge
            new_edges = edges[:]
            ei = random.randint(0, len(edges) - 1)
            new_edge = random.choice(all_e)
            if new_edge not in new_edges:
                new_edges[ei] = new_edge
                s = _raw_score(vals, new_edges)
                if s < cur_score:
                    edges = new_edges
                    cur_score = s
        
        if cur_score < best_score:
            best_score = cur_score
            best_result = {'values': vals[:], 'edges': [list(e) for e in edges]}
    
    # Ensure valid output
    if best_result is None:
        # Fallback
        vals = list(range(1, n + 1))
        edges = [list(e) for e in all_e[:m]]
        best_result = {'values': vals, 'edges': edges}
    
    # Final validation
    vals = best_result['values']
    edges = best_result['edges']
    
    # Ensure all types are correct
    best_result['values'] = [int(v) for v in vals]
    best_result['edges'] = [[int(a), int(b)] for a, b in edges]
    
    return best_result
# EVOLVE-BLOCK-END
