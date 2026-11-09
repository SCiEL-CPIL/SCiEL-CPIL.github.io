"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict, Counter

_DEADLINE = time.time() + 85  # safety margin before 90s kill

def _cluster_edges_by_sum_prod(values, n, m, edge_sp):
    """
    Deterministic edge selection based on clustering by sum and product values.
    Edges that share the same sum or product are preferred, as they contribute
    less to the distinct count.
    """
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    # Build frequency maps for sums and products
    sum_freq = {}
    prod_freq = {}
    for e in all_edges:
        s, p = edge_sp[e]
        sum_freq[s] = sum_freq.get(s, 0) + 1
        prod_freq[p] = prod_freq.get(p, 0) + 1
    
    # Score each edge: prefer edges with high-frequency sums AND products
    edge_scores = []
    for e in all_edges:
        s, p = edge_sp[e]
        sf = sum_freq.get(s, 0)
        pf = prod_freq.get(p, 0)
        # Combined score: geometric mean or product of frequencies
        score = sf * pf  # product rewards edges where both sum and product are common
        edge_scores.append((score, e))
    
    # Sort by score descending
    edge_scores.sort(key=lambda x: -x[0])
    
    # Greedily select edges, tracking which sums/prods are already "covered"
    selected = []
    selected_sums = set()
    selected_prods = set()
    
    # First pass: take edges with highest combined frequency
    for score, e in edge_scores:
        if len(selected) >= m:
            break
        s, p = edge_sp[e]
        # Always accept if it doesn't add new distinct values
        if s in selected_sums and p in selected_prods:
            selected.append(e)
        elif s in selected_sums or p in selected_prods:
            # Accept if it covers at least one already-seen value
            selected.append(e)
    
    # If we still need more edges, fill from remaining sorted list
    if len(selected) < m:
        selected_set = set(selected)
        for score, e in edge_scores:
            if len(selected) >= m:
                break
            if e not in selected_set:
                selected.append(e)
                selected_set.add(e)
    
    return selected[:m]

def _score_detailed(values, edges):
    """Return (sums_set, prods_set, score)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return sums, prods, max(len(sums), len(prods))

def _greedy_edge_selection(values, n, m, edge_sp, time_budget=10.0):
    """
    Deterministic greedy: iteratively add the edge that minimizes the increase
    in max(distinct_sums, distinct_prods). Uses a priority queue for efficiency.
    """
    import heapq
    
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    
    current_edges = set()
    sum_counts = Counter()
    prod_counts = Counter()
    sums_set = set()
    prods_set = set()
    
    # Precompute sum and product for each edge
    edge_sums = {e: edge_sp[e][0] for e in all_edges}
    edge_prods = {e: edge_sp[e][1] for e in all_edges}
    
    for step in range(m):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 5:
            break
        
        best_edge = None
        best_score = m + 1  # worst case
        best_ds = 0
        best_dp = 0
        
        # For each candidate edge not already selected, compute marginal impact
        for e in all_edges:
            if e in current_edges:
                continue
            
            s = edge_sums[e]
            p = edge_prods[e]
            
            # Marginal increase in distinct sums
            ds = 1 if s not in sums_set else 0
            # Marginal increase in distinct products
            dp = 1 if p not in prods_set else 0
            
            new_num_sums = len(sums_set) + ds
            new_num_prods = len(prods_set) + dp
            new_score = max(new_num_sums, new_num_prods)
            
            if new_score < best_score:
                best_score = new_score
                best_edge = e
                best_ds = ds
                best_dp = dp
                if best_score <= 3:
                    break
        
        if best_edge is None:
            # Fallback: pick any unselected edge
            for e in all_edges:
                if e not in current_edges:
                    best_edge = e
                    s = edge_sums[best_edge]
                    p = edge_prods[best_edge]
                    best_ds = 1 if s not in sums_set else 0
                    best_dp = 1 if p not in prods_set else 0
                    break
        
        if best_edge is not None:
            current_edges.add(best_edge)
            s = edge_sums[best_edge]
            p = edge_prods[best_edge]
            sum_counts[s] += 1
            prod_counts[p] += 1
            sums_set.add(s)
            prods_set.add(p)
    
    return max(len(sums_set), len(prods_set)), current_edges

def _fast_anneal(values, n, m, edge_sp, time_budget=5.0):
    """
    Fast simulated annealing on edge selection.
    """
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    # Initialize with random set of m edges
    current_edges = set(random.sample(all_edges, m))
    
    current_sums_set, current_prods_set, current_score = _score_detailed(values, list(current_edges))
    best_score = current_score
    best_edges = set(current_edges)
    
    sum_counts = Counter()
    prod_counts = Counter()
    for e in current_edges:
        s, p = edge_sp[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    T = 8.0
    T_min = 0.005
    cooling = 0.9995
    max_iter = 200000
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 2:
            break
        if current_score <= 3:
            break
        
        # Pick random edge to remove
        remove_edge = random.choice(list(current_edges))
        
        # Pick random edge to add
        add_edge = all_edges[random.randint(0, total_edges - 1)]
        
        if add_edge == remove_edge:
            continue
        if add_edge in current_edges:
            continue

        rs, rp = edge_sp[remove_edge]
        as_, ap = edge_sp[add_edge]
        
        delta_sums = 0
        if sum_counts[rs] == 1:
            delta_sums -= 1
        if sum_counts[as_] == 0:
            delta_sums += 1
        
        delta_prods = 0
        if prod_counts[rp] == 1:
            delta_prods -= 1
        if prod_counts[ap] == 0:
            delta_prods += 1
        
        new_num_sums = len(current_sums_set) + delta_sums
        new_num_prods = len(current_prods_set) + delta_prods
        new_score = max(new_num_sums, new_num_prods)
        
        delta_score = new_score - current_score
        
        if delta_score < 0 or (T > T_min and random.random() < math.exp(-delta_score / T)):
            current_edges.discard(remove_edge)
            current_edges.add(add_edge)
            
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
            
            sum_counts[as_] += 1
            prod_counts[ap] += 1
            
            if sum_counts[rs] == 0:
                current_sums_set.discard(rs)
            if sum_counts[as_] == 1:
                current_sums_set.add(as_)
            
            if prod_counts[rp] == 0:
                current_prods_set.discard(rp)
            if prod_counts[ap] == 1:
                current_prods_set.add(ap)
            
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        
        T = max(T_min, T * cooling)
    
    return best_score, best_edges

def _polish_edges(values, n, m, edge_sp, initial_edges, time_budget=3.0):
    """
    Final high-precision polish: try all single-edge swaps greedily,
    then do a few rounds of local search with tight acceptance.
    Fixed: properly restore state after tentative removal.
    """
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    current_edges = set(initial_edges)
    current_sums_set, current_prods_set, current_score = _score_detailed(values, list(current_edges))
    best_score = current_score
    best_edges = set(current_edges)
    
    sum_counts = Counter()
    prod_counts = Counter()
    for e in current_edges:
        s, p = edge_sp[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    # Multiple passes of greedy improvement
    for _pass in range(15):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 2:
            break
        if current_score <= 3:
            break
        
        improved = False
        # Try removing each edge and adding the best replacement
        for rem in list(current_edges):
            if time.time() - start_time > time_budget:
                break
            if time.time() > _DEADLINE - 2:
                break
            
            rs, rp = edge_sp[rem]
            
            # Tentatively remove: save old state
            old_sum_count = sum_counts.get(rs, 0)
            old_prod_count = prod_counts.get(rp, 0)
            
            # Apply removal
            sum_counts[rs] = old_sum_count - 1
            prod_counts[rp] = old_prod_count - 1
            
            if sum_counts[rs] == 0:
                current_sums_set.discard(rs)
                sum_counts.pop(rs, None)
            if prod_counts[rp] == 0:
                current_prods_set.discard(rp)
                prod_counts.pop(rp, None)
            
            temp_sums = len(current_sums_set)
            temp_prods = len(current_prods_set)
            
            best_add = None
            best_new_score = current_score  # we want to minimize
            
            # Try all candidate additions for thoroughness (n=14, so 91 edges max)
            for add in all_edges:
                if add in current_edges or add == rem:
                    continue
                as_, ap = edge_sp[add]
                
                ns = temp_sums + (1 if as_ not in current_sums_set else 0)
                np_ = temp_prods + (1 if ap not in current_prods_set else 0)
                ns_score = max(ns, np_)
                
                if ns_score < best_new_score:
                    best_new_score = ns_score
                    best_add = add
            
            # Restore removal: put back the removed edge's contribution
            if best_add is not None and best_new_score < current_score:
                # Apply the swap: keep removal, add new edge
                as_, ap = edge_sp[best_add]
                
                # Add new edge
                sum_counts[as_] = sum_counts.get(as_, 0) + 1
                prod_counts[ap] = prod_counts.get(ap, 0) + 1
                
                if sum_counts[as_] == 1:
                    current_sums_set.add(as_)
                if prod_counts[ap] == 1:
                    current_prods_set.add(ap)
                
                # Update edge set
                current_edges.discard(rem)
                current_edges.add(best_add)
                
                current_score = best_new_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(current_edges)
            else:
                # No improvement: restore the removed edge
                sum_counts[rs] = old_sum_count
                prod_counts[rp] = old_prod_count
                current_sums_set.add(rs)
                current_prods_set.add(rp)
        
        if not improved:
            break
    
    return best_score, best_edges

def _generate_value_candidates(n, max_val=10000):
    """Generate a focused set of high-quality candidates."""
    candidates = []
    seen_keys = set()
    
    def add_candidate(vals):
        if len(vals) != n or len(set(vals)) != n:
            return
        if any(v < 1 or v > max_val for v in vals):
            return
        key = tuple(sorted(vals))
        if key not in seen_keys:
            seen_keys.add(key)
            candidates.append(sorted(vals))
    
    # 1. Consecutive small integers
    add_candidate(list(range(1, n + 1)))
    
    # 2. Smooth numbers
    smooth_sets = [
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
        [1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 64, 96, 144, 192],
    ]
    for s in smooth_sets:
        if len(s) == n:
            add_candidate(s)
    
    # 3. Geometric progressions with integer rounding
    for base in [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]:
        vals = []
        for i in range(n):
            v = max(1, int(round(base ** i)))
            vals.append(v)
        # Ensure distinct
        seen = set()
        unique = []
        for v in vals:
            while v in seen or v < 1:
                v += 1
            seen.add(v)
            unique.append(v)
        if len(unique) == n and all(v <= max_val for v in unique):
            add_candidate(unique)
    
    # 4. Random small numbers with structure
    random.seed(42)
    for trial in range(30):
        # Mix of small consecutive and some smooth numbers
        base_vals = list(range(1, n + 1))
        # Perturb some values
        for _ in range(random.randint(2, 6)):
            idx = random.randint(0, n - 1)
            base_vals[idx] = random.randint(1, 50)
        add_candidate(base_vals)
    
    # 5. Values with many common divisors (good for product collisions)
    # Use multiples of small numbers
    for base_mult in [2, 3, 4, 6]:
        vals = [base_mult * (i + 1) for i in range(n)]
        if all(v <= max_val for v in vals):
            add_candidate(vals)
    
    # 6. Arithmetic progressions
    for start in [1, 2, 3]:
        for step in [1, 2, 3, 4, 5]:
            vals = [start + i * step for i in range(n)]
            if all(v <= max_val for v in vals):
                add_candidate(vals)
    
    # 7. Values that are products of small primes
    small_primes = [2, 3, 5, 7]
    random.seed(123)
    for trial in range(20):
        vals = []
        for i in range(n):
            # Pick 2-3 small primes and multiply
            k = random.randint(2, 3)
            chosen = random.sample(small_primes, k)
            v = 1
            for p in chosen:
                v *= p
            vals.append(v)
        # Add some variation
        for i in range(n):
            if random.random() < 0.3:
                vals[i] = vals[i] * random.choice([2, 3, 4])
        add_candidate(vals)
    
    # 8. Random from 1-100
    random.seed(999)
    for trial in range(15):
        vals = sorted(random.sample(range(1, 101), n))
        add_candidate(vals)
    
    # 9. Random from 1-200
    for trial in range(10):
        vals = sorted(random.sample(range(1, 201), n))
        add_candidate(vals)
    
    # 10. Powers of 2 (extreme)
    add_candidate([2**i for i in range(n)])
    
    # 11. Mixed: some powers of 2, some smooth
    for trial in range(10):
        vals = []
        for i in range(n):
            if random.random() < 0.5:
                vals.append(2**random.randint(0, 10))
            else:
                vals.append(random.choice([1,2,3,4,5,6,7,8,9,10,12,15,16,18,20,24,25,27,30,32,36,40,48,54,60,64,72,80,81,90,96,100]))
        add_candidate(vals)
    
    return candidates

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m
    best_result = None
    
    candidates = _generate_value_candidates(n, max_val)
    
    # Sort candidates by a quick heuristic: prefer those with more value overlap
    # (smaller range, more common factors)
    def candidate_heuristic(vals):
        # Smaller range is generally better for sum collisions
        rng = max(vals) - min(vals)
        # Count common divisors
        from math import gcd
        common_div_count = 0
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                g = gcd(vals[i], vals[j])
                if g > 1:
                    common_div_count += 1
        return (rng, -common_div_count)
    
    candidates.sort(key=candidate_heuristic)
    
    num_candidates = len(candidates)
    
    # Allocate time: first few candidates get more time, later ones less
    # Total budget: 75s (leaving 10s for polish + safety)
    total_budget = 75.0
    # Give top candidates more time
    weights = []
    for i in range(num_candidates):
        if i < 5:
            weights.append(2.0)
        elif i < 10:
            weights.append(1.5)
        else:
            weights.append(1.0)
    total_weight = sum(weights)
    
    for idx, vals in enumerate(candidates):
        if time.time() > _DEADLINE - 10:
            break
        
        time_per_candidate = total_budget * weights[idx] / total_weight
        
        # Precompute edge info
        all_edges = list(combinations(range(n), 2))
        edge_sp = {}
        for e in all_edges:
            u, v = e
            edge_sp[e] = (vals[u] + vals[v], vals[u] * vals[v])
        
        # Run greedy first (deterministic, fast)
        greedy_budget = min(5.0, time_per_candidate * 0.3)
        greedy_score, greedy_edges = _greedy_edge_selection(vals, n, m, edge_sp, time_budget=greedy_budget)
        
        # Also try clustering-based initialization
        cluster_edges = _cluster_edges_by_sum_prod(vals, n, m, edge_sp)
        cluster_score = _score_detailed(vals, cluster_edges)[2]
        
        # Run annealing from greedy initialization
        anneal_budget = time_per_candidate * 0.4
        score, edges = _fast_anneal(vals, n, m, edge_sp, time_budget=anneal_budget)
        
        # Run annealing from clustering initialization
        cluster_anneal_budget = time_per_candidate * 0.2
        cluster_anneal_score, cluster_anneal_edges = _fast_anneal(vals, n, m, edge_sp, time_budget=cluster_anneal_budget)
        
        # Also try annealing from random init
        random_score, random_edges = _fast_anneal(vals, n, m, edge_sp, time_budget=time_per_candidate * 0.1)
        
        # Take the best of all initializations and anneals
        candidates_score = [(score, edges), (greedy_score, greedy_edges), 
                           (cluster_score, set(cluster_edges)),
                           (cluster_anneal_score, cluster_anneal_edges),
                           (random_score, random_edges)]
        
        score, edges = min(candidates_score, key=lambda x: x[0])
        
        # Polish pass
        if score < best_score:
            polish_budget = min(3.0, time_per_candidate * 0.4)
            if time.time() < _DEADLINE - 5:
                score, edges = _polish_edges(vals, n, m, edge_sp, edges, time_budget=polish_budget)
        
        if score < best_score:
            best_score = score
            best_result = (vals, [list(e) for e in edges])
            if best_score <= 3:
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
