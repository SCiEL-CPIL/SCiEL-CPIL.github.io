import time
import random
from itertools import combinations
from collections import Counter

DEADLINE = 85.0

def _evaluate_edge_set(values, edges):
    """Compute the raw value: max(distinct sums, distinct products)."""
    sums = Counter()
    prods = Counter()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums[s] += 1
        prods[p] += 1
    
    distinct_sums = sum(1 for v in sums.values() if v > 0)
    distinct_prods = sum(1 for v in prods.values() if v > 0)
    
    return max(distinct_sums, distinct_prods)

def _optimize_edges(values, m, deadline):
    """
    Optimize the set of m edges for the given values.
    Uses Simulated Annealing.
    """
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    
    # Precompute sums and products for all edges
    edge_data = []
    for e in all_edges:
        i, j = e
        s = values[i] + values[j]
        p = values[i] * values[j]
        edge_data.append((e, s, p))
        
    # Index edge data for quick lookup
    # We need to maintain counts of sums and products for the current edge set
    # and be able to add/remove an edge quickly.
    
    def _get_score(counters):
        distinct_sums = sum(1 for v in counters[0].values() if v > 0)
        distinct_prods = sum(1 for v in counters[1].values() if v > 0)
        return max(distinct_sums, distinct_prods)

    # Initialize multiple starting points
    best_score = float('inf')
    best_edges = None
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy Add
    # Start with empty, add edges that minimize cost increase
    current_edges = set()
    sum_counts = Counter()
    prod_counts = Counter()
    
    # Sort edges by sum then product to bias towards small values initially?
    # Or just random order?
    edge_indices = list(range(len(all_edges)))
    
    # We will run a few restarts of SA
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Initialize state
        if restart == 0:
            # Greedy initialization
            current_edges = set()
            sum_counts = Counter()
            prod_counts = Counter()
            remaining = list(range(len(all_edges)))
            rng.shuffle(remaining)
            
            # Add m edges greedily
            for _ in range(m):
                best_idx = -1
                best_cost = float('inf')
                # Check a subset of remaining to speed up? No, m is small, remaining is ~90.
                # Just check all remaining for the first few, then random?
                # Let's check all remaining for simplicity as n is small.
                for idx in remaining:
                    e, s, p = edge_data[idx]
                    # Tentative add
                    old_s_count = sum_counts[s]
                    old_p_count = prod_counts[p]
                    
                    sum_counts[s] += 1
                    prod_counts[p] += 1
                    
                    cost = _get_score((sum_counts, prod_counts))
                    
                    # Tentative remove
                    sum_counts[s] -= 1
                    prod_counts[p] -= 1
                    if sum_counts[s] == 0: del sum_counts[s]
                    if prod_counts[p] == 0: del prod_counts[p]
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = idx
                        
                if best_idx != -1:
                    e, s, p = edge_data[best_idx]
                    current_edges.add(best_idx)
                    sum_counts[s] += 1
                    prod_counts[p] += 1
                    remaining.remove(best_idx)
                    
        elif restart == 1:
            # Random initialization
            current_edges = set(rng.sample(range(len(all_edges)), m))
            sum_counts = Counter()
            prod_counts = Counter()
            for idx in current_edges:
                e, s, p = edge_data[idx]
                sum_counts[s] += 1
                prod_counts[p] += 1
                
        else:
            # Perturbation of best found so far
            if best_edges is None:
                continue
            current_edges = set(best_edges)
            sum_counts = Counter()
            prod_counts = Counter()
            for idx in current_edges:
                e, s, p = edge_data[idx]
                sum_counts[s] += 1
                prod_counts[p] += 1
            
            # Randomly swap k edges
            k = 3
            for _ in range(k):
                if not current_edges: continue
                in_list = list(current_edges)
                out_list = [i for i in range(len(all_edges)) if i not in current_edges]
                if not out_list: continue
                idx_in = rng.choice(in_list)
                idx_out = rng.choice(out_list)
                
                # Remove idx_in
                e, s, p = edge_data[idx_in]
                sum_counts[s] -= 1
                prod_counts[p] -= 1
                if sum_counts[s] == 0: del sum_counts[s]
                if prod_counts[p] == 0: del prod_counts[p]
                current_edges.remove(idx_in)
                
                # Add idx_out
                e, s, p = edge_data[idx_out]
                sum_counts[s] += 1
                prod_counts[p] += 1
                current_edges.add(idx_out)

        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        alpha = 0.99
        current_score = _get_score((sum_counts, prod_counts))
        
        if current_score < best_score:
            best_score = current_score
            best_edges = list(current_edges)
            
        iters = 0
        max_iters = 2000
        while iters < max_iters:
            if time.time() > deadline:
                break
            
            # Pick a random edge in, random edge out
            in_list = list(current_edges)
            out_list = [i for i in range(len(all_edges)) if i not in current_edges]
            if not in_list or not out_list:
                break
                
            idx_in = rng.choice(in_list)
            idx_out = rng.choice(out_list)
            
            e_in, s_in, p_in = edge_data[idx_in]
            e_out, s_out, p_out = edge_data[idx_out]
            
            # Calculate delta
            # Remove in
            old_s_in = sum_counts[s_in]
            old_p_in = prod_counts[p_in]
            
            # Add out
            old_s_out = sum_counts[s_out]
            old_p_out = prod_counts[p_out]
            
            # Determine change in distinct counts
            # Distinct sums change:
            # -1 if old_s_in == 1 and (s_out == s_in or old_s_out > 0) -> wait, logic is tricky.
            # Let's just update counters and compute score. It's fast enough.
            
            sum_counts[s_in] -= 1
            if sum_counts[s_in] == 0: del sum_counts[s_in]
            prod_counts[p_in] -= 1
            if prod_counts[p_in] == 0: del prod_counts[p_in]
            
            sum_counts[s_out] += 1
            prod_counts[p_out] += 1
            
            new_score = _get_score((sum_counts, prod_counts))
            
            if new_score < current_score or (rng.random() < pow(2.718, -(new_score - current_score) / T)):
                current_score = new_score
                current_edges.remove(idx_in)
                current_edges.add(idx_out)
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(current_edges)
            else:
                # Revert
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0: del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0: del prod_counts[p_out]
                
                sum_counts[s_in] += 1
                prod_counts[p_in] += 1
                
            T *= alpha
            if T < T_min:
                T = T_min
            iters += 1
            
    if best_edges is None:
        # Fallback
        best_edges = list(range(m))
        
    return [all_edges[idx] for idx in best_edges], best_score

def construct():
    start_time = time.time()
    n = 14
    m = 35
    
    # Candidate value sets
    candidates = [
        list(range(1, 15)),
        list(range(2, 16)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 24],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        [1, 2, 3, 4, 6, 9, 12, 16, 18, 24, 27, 32, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 36, 48],
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 18, 20, 24, 27],
        [1, 2, 3, 4, 6, 9, 12, 15, 18, 20, 24, 27, 30, 36],
        # Add some random dense sets
    ]
    
    # Generate a few random dense sets
    rng = random.Random(123)
    for _ in range(10):
        base = rng.randint(1, 5)
        step = rng.randint(1, 2)
        vals = [base + i*step for i in range(n)]
        if len(set(vals)) == n and max(vals) <= 10000:
            candidates.append(vals)
            
    best_score = float('inf')
    best_result = None
    
    for values in candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        deadline_for_this = min(start_time + DEADLINE - 2, time.time() + 8.0)
        edges, score = _optimize_edges(values, m, deadline_for_this)
        
        if score is not None and score < best_score:
            best_score = score
            best_result = {
                'values': [int(v) for v in values],
                'edges': [[int(i), int(j)] for i, j in edges]
            }
            
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result = {'values': values, 'edges': edges}
        
    return best_result
