"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

_START_TIME = time.time()
_DEADLINE = _START_TIME + 82  # 8s safety margin before 90s kill

def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _fast_hill_climb(values, m, n, initial_edges=None, max_iter=200000):
    """Fast hill climbing with maintained counters (no rebuild per step)."""
    all_edges = list(combinations(range(n), 2))
    edge_sums = {}
    edge_prods = {}
    for e in all_edges:
        u, v = e
        edge_sums[e] = values[u] + values[v]
        edge_prods[e] = values[u] * values[v]
    
    if initial_edges is not None:
        edges = set(initial_edges)
    else:
        edges = set(random.sample(all_edges, m))
    
    # Maintain counters
    sum_count = Counter()
    prod_count = Counter()
    for e in edges:
        sum_count[edge_sums[e]] += 1
        prod_count[edge_prods[e]] += 1
    
    # Clean up zeros
    for k in list(sum_count.keys()):
        if sum_count[k] == 0:
            del sum_count[k]
    for k in list(prod_count.keys()):
        if prod_count[k] == 0:
            del prod_count[k]
    
    current_score = max(len(sum_count), len(prod_count))
    best_score = current_score
    best_edges = set(edges)
    
    # Precompute edge indices for faster lookup
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    for it in range(max_iter):
        if time.time() > _DEADLINE:
            break
        if current_score <= 2:
            break
        
        # Pick a random edge to remove
        u, v = random.choice(list(edges))
        old_e = (u, v)
        old_s = edge_sums[old_e]
        old_p = edge_prods[old_e]
        
        # Pick a random edge to add (not currently in edges)
        # Use rejection sampling for speed
        while True:
            new_e = random.choice(all_edges)
            if new_e not in edges:
                break
        new_s = edge_sums[new_e]
        new_p = edge_prods[new_e]
        
        # Compute delta in score
        # Remove old edge
        delta_sums = 0
        delta_prods = 0
        
        # Removing old_s
        if sum_count[old_s] == 1:
            delta_sums -= 1
        # Adding new_s
        if new_s not in sum_count or sum_count[new_s] == 0:
            delta_sums += 1
        
        # Removing old_p
        if prod_count[old_p] == 1:
            delta_prods -= 1
        # Adding new_p
        if new_p not in prod_count or prod_count[new_p] == 0:
            delta_prods += 1
        
        new_num_sums = len(sum_count) + delta_sums
        new_num_prods = len(prod_count) + delta_prods
        new_score = max(new_num_sums, new_num_prods)
        
        if new_score <= current_score:
            # Apply the swap
            edges.discard(old_e)
            edges.add(new_e)
            sum_count[old_s] -= 1
            if sum_count[old_s] == 0:
                del sum_count[old_s]
            sum_count[new_s] = sum_count.get(new_s, 0) + 1
            prod_count[old_p] -= 1
            if prod_count[old_p] == 0:
                del prod_count[old_p]
            prod_count[new_p] = prod_count.get(new_p, 0) + 1
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(edges)
    
    return best_score, [list(e) for e in best_edges]

def _greedy_edges(values, m, n):
    """Greedy: pick edges that minimize new distinct sums/products."""
    all_edges = list(combinations(range(n), 2))
    
    sum_count = Counter()
    prod_count = Counter()
    edges = []
    
    # First pass: compute all edge sums and products
    edge_data = []
    for e in all_edges:
        u, v = e
        s = values[u] + values[v]
        p = values[u] * values[v]
        edge_data.append((e, s, p))
    
    # Greedy: repeatedly pick the edge that adds fewest new distinct values
    for _ in range(m):
        if time.time() > _DEADLINE:
            break
        best_idx = -1
        best_cost = float('inf')
        for i, (e, s, p) in enumerate(edge_data):
            if e in [x[0] for x in edges]:
                continue
            new_s = 0 if s in sum_count else 1
            new_p = 0 if p in prod_count else 1
            cost = new_s + new_p
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        if best_idx == -1:
            break
        e, s, p = edge_data[best_idx]
        edges.append((e, s, p))
        sum_count[s] += 1
        prod_count[p] += 1
    
    return [list(e) for e, _, _ in edges]

def _try_value_set(values, m, n):
    """Try a value set with greedy init then fast hill climbing."""
    if time.time() > _DEADLINE - 2:
        return None
    
    # Get greedy initial solution
    greedy_edges = _greedy_edges(values, m, n)
    if not greedy_edges:
        return None
    
    # Convert to tuple set for fast lookup
    greedy_set = [tuple(e) for e in greedy_edges]
    
    # Run fast hill climbing from greedy result
    hc_score, hc_edges = _fast_hill_climb(values, m, n, initial_edges=greedy_set, max_iter=150000)
    
    return hc_score, values, hc_edges

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m
    best_result = None
    
    # Candidate value sets - focus on creating collisions
    candidates = []
    
    # Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # Even numbers
    candidates.append(list(range(2, 2 * n + 1, 2)))
    
    # Highly composite
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    # Multiples of 2 and 3 mixed
    candidates.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24])
    
    # Small numbers with many shared factors
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 20, 24, 27, 32])
    
    # Arithmetic progressions
    for start in range(1, 7):
        for step in range(1, 5):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= max_val and len(set(vals)) == n:
                candidates.append(vals)
    
    # Geometric-ish (powers of 2 scaled)
    candidates.append([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 5120, 8192])
    
    # Random sets with bias toward small values
    random.seed(42)
    for trial in range(30):
        # Mix: some very small, some small
        base = sorted(random.sample(range(1, 30), min(n, 20)))
        if len(base) < n:
            # Pad with larger values
            extra = random.sample(range(30, 100), n - len(base))
            base = sorted(base + extra)
        else:
            base = base[:n]
        if len(set(base)) == n and all(isinstance(v, int) for v in base):
            candidates.append(base)
    
    # More structured: multiples of small primes
    for prime in [2, 3, 5, 7]:
        vals = []
        i = 1
        while len(vals) < n:
            v = prime * i
            if v <= max_val and v not in vals:
                vals.append(v)
            i += 1
        if len(vals) == n:
            candidates.append(vals)
    
    # Mixed: some multiples of 2, some of 3
    for _ in range(15):
        vals = set()
        while len(vals) < n:
            r = random.random()
            if r < 0.5:
                vals.add(random.randint(1, 20) * 2)
            elif r < 0.8:
                vals.add(random.randint(1, 15) * 3)
            else:
                vals.add(random.randint(1, 40))
        vals = sorted(v for v in vals if 1 <= v <= max_val)[:n]
        if len(vals) == n and len(set(vals)) == n:
            candidates.append(vals)
    
    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates
    
    for vals in candidates:
        if time.time() > _DEADLINE - 3:
            break
        if len(set(vals)) != n or min(vals) < 1 or max(vals) > max_val:
            continue
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            continue
        
        result = _try_value_set(vals, m, n)
        if result is None:
            continue
        score, values, edges = result
        if score < best_score:
            best_score = score
            best_result = (values, edges)
            if best_score <= 2:
                break
    
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
