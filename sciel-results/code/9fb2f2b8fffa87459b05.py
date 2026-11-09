"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict, Counter

_DEADLINE = time.time() + 82  # safety margin before 90s kill

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

def _greedy_edge_selection(values, n, m, edge_sp, time_budget=2.0):
    """
    Greedy heuristic: iteratively select the edge that minimizes the growth
    in distinct sums and products. This provides a strong initial solution.
    """
    start_time = time.time()
    
    # Track current sums and products
    current_sums = set()
    current_prods = set()
    sum_counts = Counter()
    prod_counts = Counter()
    
    selected_edges = []
    used_vertices = set()
    
    # Precompute all edges with their sum and product
    all_edges = list(combinations(range(n), 2))
    edge_info = [(e, edge_sp[e][0], edge_sp[e][1]) for e in all_edges]
    
    for _ in range(m):
        if time.time() - start_time > time_budget:
            break
        
        best_edge = None
        best_delta = float('inf')
        best_new_sums = None
        best_new_prods = None
        
        # Sample a subset of candidate edges for speed
        if len(edge_info) > 50:
            candidates = random.sample(edge_info, 50)
        else:
            candidates = edge_info
        
        for e, s, p in candidates:
            u, v = e
            # Skip if edge already selected (shouldn't happen but safe)
            if e in selected_edges:
                continue
            
            # Calculate how many new sums and products this edge would add
            new_sums = 0
            new_prods = 0
            
            if s not in current_sums:
                new_sums = 1
            if p not in current_prods:
                new_prods = 1
            
            # We want to minimize the increase in max(|sums|, |prods|)
            # Approximate delta as max(new_sums, new_prods) with some weighting
            # Actually, we should look at the resulting max
            new_num_sums = len(current_sums) + new_sums
            new_num_prods = len(current_prods) + new_prods
            new_score = max(new_num_sums, new_num_prods)
            
            # Also consider if this edge connects already-used vertices (might be better for structure)
            # But for now, just minimize new_score
            if new_score < best_delta:
                best_delta = new_score
                best_edge = e
                best_new_sums = s
                best_new_prods = p
        
        if best_edge is None:
            # Fallback: pick random
            best_edge = random.choice(all_edges)
            best_new_sums = edge_sp[best_edge][0]
            best_new_prods = edge_sp[best_edge][1]
        
        # Add the edge
        selected_edges.append(best_edge)
        current_sums.add(best_new_sums)
        current_prods.add(best_new_prods)
        sum_counts[best_new_sums] += 1
        prod_counts[best_new_prods] += 1
    
    return selected_edges, current_sums, current_prods, sum_counts, prod_counts

def _fast_anneal(values, n, m, edge_sp, time_budget=5.0, init_edges=None):
    """
    Fast simulated annealing on edge selection.
    Can start from a greedy initialization or random.
    """
    start_time = time.time()
    all_edges = list(combinations(range(n), 2))
    total_edges = len(all_edges)
    
    # Initialize
    if init_edges is not None and len(init_edges) == m:
        current_edges = set(init_edges)
        current_sums_set = set()
        current_prods_set = set()
        sum_counts = Counter()
        prod_counts = Counter()
        for e in current_edges:
            s, p = edge_sp[e]
            current_sums_set.add(s)
            current_prods_set.add(p)
            sum_counts[s] += 1
            prod_counts[p] += 1
        current_score = max(len(current_sums_set), len(current_prods_set))
    else:
        current_edges = set(random.sample(all_edges, m))
        current_sums_set, current_prods_set, current_score = _score_detailed(values, list(current_edges))
        sum_counts = Counter()
        prod_counts = Counter()
        for e in current_edges:
            s, p = edge_sp[e]
            sum_counts[s] += 1
            prod_counts[p] += 1
    
    best_score = current_score
    best_edges = set(current_edges)
    
    T = 3.0
    T_min = 0.01
    cooling = 0.9995
    
    for it in range(100000):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE - 3:
            break
        if current_score <= 3:
            break
        
        # Pick random edge to remove
        remove_edge = random.choice(list(current_edges))
        
        # Pick random edge to add
        add_edge = all_edges[random.randint(0, total_edges - 1)]
        
        if add_edge == remove_edge or add_edge in current_edges:
            continue

        rs, rp = edge_sp[remove_edge]
        as_, ap = edge_sp[add_edge]
        
        # Compute delta
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
            # Apply move
            current_edges.discard(remove_edge)
            current_edges.add(add_edge)
            
            # Update counts
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
            
            sum_counts[as_] += 1
            prod_counts[ap] += 1
            
            # Update sets
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

def _generate_value_candidates(n, max_val=10000):
    """Generate a smaller, more focused set of high-quality candidates."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, n + 1)))
    
    # 2. Smooth numbers (highly composite)
    smooth_14 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    if len(smooth_14) == n:
        candidates.append(smooth_14)
        
    # 3. Another smooth set
    smooth_14b = [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
    if len(smooth_14b) == n:
        candidates.append(smooth_14b)
        
    # 4. Powers of 2
    candidates.append([2**i for i in range(n)])
    
    # 5. Random small numbers
    random.seed(42)
    for _ in range(15):
        vals = sorted(random.sample(range(1, 50), n))
        if len(set(vals)) == n:
            candidates.append(vals)
            
    # 6. Geometric progression
    base = 1.5
    vals = [max(1, int(base**i)) for i in range(n)]
    seen = set()
    unique_vals = []
    for v in vals:
        while v in seen:
            v += 1
        seen.add(v)
        unique_vals.append(v)
    if len(unique_vals) == n and all(v <= max_val for v in unique_vals):
        candidates.append(sorted(unique_vals))

    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m
    best_result = None
    
    candidates = _generate_value_candidates(n, max_val)
    num_candidates = len(candidates)
    
    # Allocate time: 15s for greedy init per candidate, 40s for annealing
    # Total budget ~80s
    time_per_candidate = 80.0 / num_candidates
    
    for idx, vals in enumerate(candidates):
        if time.time() > _DEADLINE - 5:
            break
        
        # Precompute edge info
        all_edges = list(combinations(range(n), 2))
        edge_sp = {}
        for e in all_edges:
            u, v = e
            edge_sp[e] = (vals[u] + vals[v], vals[u] * vals[v])
        
        # Step 1: Greedy initialization
        greedy_budget = min(2.0, time_per_candidate * 0.2)
        greedy_edges, g_sums, g_prods, g_sum_counts, g_prod_counts = _greedy_edge_selection(
            vals, n, m, edge_sp, time_budget=greedy_budget
        )
        
        # Step 2: Annealing starting from greedy
        anneal_budget = min(time_per_candidate * 0.7, 30.0)
        score, edges = _fast_anneal(vals, n, m, edge_sp, time_budget=anneal_budget, init_edges=greedy_edges)
        
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
