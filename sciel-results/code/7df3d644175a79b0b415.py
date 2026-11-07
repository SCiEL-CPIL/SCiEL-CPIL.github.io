import time
import random
import math
from itertools import combinations
from collections import Counter

DEADLINE = 85.0
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

def _optimize_edges(values, deadline):
    """
    Optimize the set of 35 edges for a fixed set of values using Simulated Annealing.
    """
    n = len(values)
    m = 35
    all_edges = list(combinations(range(n), 2))
    num_edges = len(all_edges)
    
    # Pre-compute sums and products for all edges
    all_sums = [values[i] + values[j] for i, j in all_edges]
    all_prods = [values[i] * values[j] for i, j in all_edges]
    
    # Initial state: Greedy selection based on "collision potential"
    # We want edges that share sums/prods with other likely edges.
    # A simple heuristic: sort edges by sum, then pick, but that's biased.
    # Better: Random start is often fine for SA, but let's try a few starts.
    
    rng = random.Random(42)
    
    best_score = float('inf')
    best_edges_set = None
    
    # Run multiple restarts to avoid local minima
    num_restarts = 5
    time_per_restart = max(1.0, (deadline - time.time()) / num_restarts)
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize edges set
        # Strategy 1: Random
        # Strategy 2: Greedy by sum frequency (pick edges with most common sums first)
        
        if r == 0:
            # Greedy initialization
            # Count sum frequencies in the whole graph
            sum_freq = Counter(all_sums)
            # Sort edges by how "popular" their sum is
            sorted_indices = sorted(range(num_edges), key=lambda idx: -sum_freq[all_sums[idx]])
            edges_set = set(sorted_indices[:m])
        elif r == 1:
            # Greedy by product frequency
            prod_freq = Counter(all_prods)
            sorted_indices = sorted(range(num_edges), key=lambda idx: -prod_freq[all_prods[idx]])
            edges_set = set(sorted_indices[:m])
        else:
            # Random
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
        max_iters = 20000
        iters = 0
        no_improve = 0
        patience = 2000
        
        while iters < max_iters:
            if time.time() > deadline:
                break
            
            # Pick an edge in the set to remove
            in_list = list(edges_set)
            idx_in = rng.choice(in_list)
            
            # Pick an edge not in the set to add
            # Efficient way: pick random until not in set (set lookup is O(1))
            while True:
                idx_out = rng.randint(0, num_edges - 1)
                if idx_out not in edges_set:
                    break
            
            s_in = all_sums[idx_in]
            p_in = all_prods[idx_in]
            s_out = all_sums[idx_out]
            p_out = all_prods[idx_out]
            
            # Tentative removal
            sum_counts[s_in] -= 1
            if sum_counts[s_in] == 0:
                del sum_counts[s_in]
            prod_counts[p_in] -= 1
            if prod_counts[p_in] == 0:
                del prod_counts[p_in]
            
            # Tentative addition
            sum_counts[s_out] += 1
            prod_counts[p_out] += 1
            
            new_distinct_sums = len(sum_counts)
            new_distinct_prods = len(prod_counts)
            new_score = max(new_distinct_sums, new_distinct_prods)
            
            delta = new_score - current_score
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = True
            else:
                # Metropolis criterion
                if T > T_min:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        accept = True
            
            if accept:
                # Commit move
                edges_set.remove(idx_in)
                edges_set.add(idx_out)
                current_score = new_score
                no_improve = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges_set = set(edges_set)
            else:
                # Revert move
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
            
    if best_edges_set is None:
        # Fallback
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
    
    # 2. Smooth numbers (highly composite)
    smooth1 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    candidates.append(smooth1)
    
    smooth2 = [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27]
    candidates.append(smooth2)
    
    smooth3 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    candidates.append(smooth3)
    
    # 3. Multiples of 2
    candidates.append([2*i for i in range(1, 15)])
    
    # 4. Multiples of 3
    candidates.append([3*i for i in range(1, 15)])
    
    # 5. Mixed small primes
    candidates.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
    
    # 6. Random dense sets
    rng = random.Random(123)
    for _ in range(10):
        vals = set()
        while len(vals) < 14:
            # Bias towards smaller numbers
            if rng.random() < 0.7:
                vals.add(rng.randint(1, 20))
            else:
                vals.add(rng.randint(1, 50))
        if len(vals) == 14:
            candidates.append(sorted(vals))
            
    # 7. Specific patterns
    candidates.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    candidates.append([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
    
    return candidates

def construct():
    start = time.time()
    n = 14
    m = 35
    
    candidates = _generate_value_candidates()
    
    best_score_global = float('inf')
    best_result_global = None
    
    # Sort candidates by a quick heuristic: sum of values (smaller is usually better for collisions)
    # But don't rely on it too much. Just iterate.
    
    for values in candidates:
        if time.time() - start > DEADLINE - 10:
            break
            
        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # Allocate time based on remaining time
        # We have ~17 candidates. Let's give each a fair share, but prioritize earlier ones?
        # Actually, just use a fixed time slice per candidate to ensure we test all.
        time_slice = min(4.0, (DEADLINE - (time.time() - start)) / (len(candidates) + 1))
        deadline_for_this = time.time() + time_slice
        
        edges, score = _optimize_edges(values, deadline_for_this)
        
        if score < best_score_global:
            best_score_global = score
            best_result_global = {
                'values': [int(v) for v in values],
                'edges': [[int(i), int(j)] for i, j in edges]
            }
            
        # If we find a very good score (e.g., < 20), we might want to spend more time on it?
        # But for now, let's just move to the next candidate to ensure diversity.
        
    if best_result_global is None:
        # Fallback
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
        best_result_global = {'values': values, 'edges': edges}
        
    return best_result_global
