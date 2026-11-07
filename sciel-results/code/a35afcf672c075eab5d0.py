import time
import random
import math
from itertools import combinations
from collections import Counter

DEADLINE = 82.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def _evaluate(values, edges):
    """Calculate the raw score for given values and edges."""
    if not edges:
        return 0
    sum_counts = Counter()
    prod_counts = Counter()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    distinct_sums = len(sum_counts)
    distinct_prods = len(prod_counts)
    return max(distinct_sums, distinct_prods)

def _polish_edges(edges_set, all_edges, all_sums, all_prods, m, deadline):
    """
    Exhaustive 1-swap and 2-swap polish to find local optimum.
    Returns improved edges_set and score.
    """
    n = len(all_edges)
    num_edges = len(all_edges)
    
    # Build initial counters
    sum_counts = Counter()
    prod_counts = Counter()
    for idx in edges_set:
        sum_counts[all_sums[idx]] += 1
        prod_counts[all_prods[idx]] += 1
    
    current_score = max(len(sum_counts), len(prod_counts))
    improved = True
    
    # 1-swap polish: try removing one edge and adding one other
    while improved:
        improved = False
        if time.time() > deadline:
            break
        in_list = list(edges_set)
        for idx_in in in_list:
            if time.time() > deadline:
                break
            s_in = all_sums[idx_in]
            p_in = all_prods[idx_in]
            
            # Tentative removal
            sum_counts[s_in] -= 1
            if sum_counts[s_in] == 0:
                del sum_counts[s_in]
            prod_counts[p_in] -= 1
            if prod_counts[p_in] == 0:
                del prod_counts[p_in]
            
            best_idx_out = -1
            best_new_score = current_score
            
            for idx_out in range(num_edges):
                if idx_out in edges_set:
                    continue
                s_out = all_sums[idx_out]
                p_out = all_prods[idx_out]
                
                # Tentative addition
                sum_counts[s_out] += 1
                prod_counts[p_out] += 1
                
                new_score = max(len(sum_counts), len(prod_counts))
                
                # Revert tentative addition
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_idx_out = idx_out
            
            # Revert tentative removal
            sum_counts[s_in] += 1
            prod_counts[p_in] += 1
            
            if best_idx_out >= 0:
                # Commit the swap
                edges_set.remove(idx_in)
                edges_set.add(best_idx_out)
                
                s_in = all_sums[idx_in]
                p_in = all_prods[idx_in]
                s_out = all_sums[best_idx_out]
                p_out = all_prods[best_idx_out]
                
                sum_counts[s_in] -= 1
                if sum_counts[s_in] == 0:
                    del sum_counts[s_in]
                prod_counts[p_in] -= 1
                if prod_counts[p_in] == 0:
                    del prod_counts[p_in]
                
                sum_counts[s_out] += 1
                prod_counts[p_out] += 1
                
                current_score = best_new_score
                improved = True
    
    return edges_set, current_score


def _optimize_edges(values, deadline):
    """
    Optimize the set of 35 edges for a fixed set of values using Simulated Annealing + polish.
    """
    n = len(values)
    m = 35
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)
    
    # Pre-compute sums and products for all edges
    all_sums = [values[i] + values[j] for i, j in all_edges]
    all_prods = [values[i] * values[j] for i, j in all_edges]
    
    rng = random.Random(42)
    
    best_score = float('inf')
    best_edges_set = None
    
    # Run multiple restarts to avoid local minima
    num_restarts = 4
    time_per_restart = max(0.5, (deadline - time.time()) / (num_restarts + 1))
    
    for r in range(num_restarts):
        if time.time() > deadline - 2.0:  # Leave time for polish
            break
            
        if r == 0:
            sum_freq = Counter(all_sums)
            sorted_indices = sorted(range(num_edges), key=lambda idx: -sum_freq[all_sums[idx]])
            edges_set = set(sorted_indices[:m])
        elif r == 1:
            prod_freq = Counter(all_prods)
            sorted_indices = sorted(range(num_edges), key=lambda idx: -prod_freq[all_prods[idx]])
            edges_set = set(sorted_indices[:m])
        elif r == 2:
            # Greedy by combined score
            combined_freq = Counter()
            for idx in range(num_edges):
                combined_freq[all_sums[idx]] += 1
                combined_freq[all_prods[idx]] += 1
            sorted_indices = sorted(range(num_edges), key=lambda idx: -(combined_freq[all_sums[idx]] + combined_freq[all_prods[idx]]))
            edges_set = set(sorted_indices[:m])
        else:
            edges_set = set(rng.sample(range(num_edges), m))
        
        # Build initial counters
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in edges_set:
            sum_counts[all_sums[idx]] += 1
            prod_counts[all_prods[idx]] += 1
            
        current_score = max(len(sum_counts), len(prod_counts))
        
        if current_score < best_score:
            best_score = current_score
            best_edges_set = set(edges_set)
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        alpha = 0.995
        max_iters = 15000
        iters = 0
        no_improve = 0
        patience = 1500
        sa_deadline = min(deadline - 2.0, time.time() + time_per_restart)
        
        while iters < max_iters:
            if time.time() > sa_deadline:
                break
            
            in_list = list(edges_set)
            idx_in = rng.choice(in_list)
            
            while True:
                idx_out = rng.randint(0, num_edges - 1)
                if idx_out not in edges_set:
                    break
            
            s_in = all_sums[idx_in]
            p_in = all_prods[idx_in]
            s_out = all_sums[idx_out]
            p_out = all_prods[idx_out]
            
            sum_counts[s_in] -= 1
            if sum_counts[s_in] == 0:
                del sum_counts[s_in]
            prod_counts[p_in] -= 1
            if prod_counts[p_in] == 0:
                del prod_counts[p_in]
            
            sum_counts[s_out] += 1
            prod_counts[p_out] += 1
            
            new_distinct_sums = len(sum_counts)
            new_distinct_prods = len(prod_counts)
            new_score = max(new_distinct_sums, new_distinct_prods)
            
            delta = new_score - current_score
            
            accept = False
            if delta <= 0:
                accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                edges_set.remove(idx_in)
                edges_set.add(idx_out)
                current_score = new_score
                no_improve = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges_set = set(edges_set)
            else:
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0:
                    del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0:
                    del prod_counts[p_out]
                
                sum_counts[s_in] += 1
                prod_counts[p_in] += 1
                no_improve += 1
                
                if no_improve > patience:
                    break
            
            T *= alpha
            if T < T_min:
                T = T_min
            iters += 1
            
    # Apply polish to the best solution found
    if best_edges_set is not None and time.time() < deadline - 0.5:
        best_edges_set, polished_score = _polish_edges(
            best_edges_set, all_edges, all_sums, all_prods, m, deadline - 0.5
        )
        if polished_score < best_score:
            best_score = polished_score
    
    if best_edges_set is None:
        best_edges_set = set(range(m))
        best_score = _evaluate(values, [all_edges[i] for i in best_edges_set])
        
    return [list(all_edges[idx]) for idx in best_edges_set], best_score

def _generate_value_candidates():
    """Generate a diverse set of candidate value lists."""
    candidates = []
    
    # 1. Consecutive small integers
    candidates.append(list(range(1, 15)))
    candidates.append(list(range(2, 16)))
    candidates.append(list(range(1, 10)) + [11, 12, 14, 15, 16])
    
    # 2. Smooth numbers (highly composite) - more collision-friendly
    smooth1 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    candidates.append(smooth1)
    
    smooth2 = [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27]
    candidates.append(smooth2)
    
    smooth3 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    candidates.append(smooth3)
    
    smooth4 = [1, 2, 3, 4, 6, 8, 9, 12, 14, 16, 18, 20, 21, 24]
    candidates.append(smooth4)
    
    smooth5 = [1, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
    candidates.append(smooth5)
    
    # 3. Multiples of 2
    candidates.append([2*i for i in range(1, 15)])
    
    # 4. Multiples of 3
    candidates.append([3*i for i in range(1, 15)])
    
    # 5. Multiples of 4
    candidates.append([4*i for i in range(1, 15)])
    
    # 6. Mixed small primes
    candidates.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    
    # 7. Dense small values with high collision potential
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16])
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18])
    candidates.append([2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20])
    
    # 8. Random dense sets
    rng = random.Random(123)
    for seed in [123, 456, 789, 101, 202, 303, 404, 505, 606, 707]:
        rng2 = random.Random(seed)
        vals = set()
        while len(vals) < 14:
            if rng2.random() < 0.8:
                vals.add(rng2.randint(1, 18))
            else:
                vals.add(rng2.randint(1, 40))
        if len(vals) == 14:
            candidates.append(sorted(vals))
            
    # 9. Specific patterns
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    candidates.append([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
    candidates.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    return candidates

def construct():
    start = time.time()
    n = 14
    m = 35
    
    candidates = _generate_value_candidates()
    
    best_score_global = float('inf')
    best_result_global = None
    
    # Sort candidates by sum of values (smaller tends to give more collisions)
    candidates.sort(key=lambda v: sum(v))
    
    num_candidates = len(candidates)
    
    for ci, values in enumerate(candidates):
        if time.time() - start > DEADLINE - 5:
            break
            
        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        
        # Allocate time: earlier candidates get more time
        remaining = DEADLINE - (time.time() - start)
        remaining_candidates = num_candidates - ci
        time_slice = min(5.0, max(1.0, remaining / remaining_candidates))
        deadline_for_this = time.time() + time_slice
        
        edges, score = _optimize_edges(values, deadline_for_this)
        
        if score < best_score_global:
            best_score_global = score
            best_result_global = {
                'values': [int(v) for v in values],
                'edges': [[int(i), int(j)] for i, j in edges]
            }
            
        # If we've found a good score, reduce time for remaining candidates
        if best_score_global <= 15:
            time_slice = min(time_slice, 2.0)
        
    if best_result_global is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result_global = {'values': values, 'edges': edges}
        
    return best_result_global
