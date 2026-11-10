import time
import random
from collections import Counter
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Precompute all possible edges (indices)
    all_edges = list(combinations(range(n), 2))
    
    # Candidate value sets:
    # 1. Small integers
    # 2. Highly composite numbers (rich in factors -> product collisions)
    # 3. Arithmetic progressions (sum collisions)
    # 4. Random
    
    def is_valid_values(vals):
        return len(vals) == n and len(set(vals)) == n and all(1 <= v <= max_val for v in vals)
    
    candidate_sets = []
    
    # Set 1: Small integers
    candidate_sets.append(list(range(1, n + 1)))
    
    # Set 2: Highly composite numbers (first 14 distinct)
    # 1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24
    hc = [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24]
    candidate_sets.append(hc)
    
    # Set 3: Mix of small and composite
    mix = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    candidate_sets.append(mix)
    
    # Set 4: Arithmetic progression
    candidate_sets.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    
    # Set 5: Random small
    rng_init = random.Random(42)
    for _ in range(20):
        vals = rng_init.sample(range(1, 50), n)
        candidate_sets.append(vals)
        
    # Set 6: Random larger
    for _ in range(20):
        vals = rng_init.sample(range(1, 200), n)
        candidate_sets.append(vals)

    best_score = float('inf')
    best_values = None
    best_edges = None
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    def optimize_edges(values):
        """
        Greedy + Local Search to find 35 edges minimizing max(distinct_sums, distinct_prods).
        """
        # Precompute sum and product for all 91 edges
        edge_sums = [values[i] + values[j] for i, j in all_edges]
        edge_prods = [values[i] * values[j] for i, j in all_edges]
        
        # Greedy Initialization
        # We want to pick edges that add the fewest new distinct sums/prods.
        # Start with empty sets.
        current_edges = set()
        sum_counts = Counter()
        prod_counts = Counter()
        
        # To make greedy effective, we can shuffle the candidate edges if there are ties,
        # but a simple greedy picking the one with minimum new additions is good.
        # However, picking an edge that creates a NEW sum/prod is bad if we can pick one that doesn't.
        # Initially, all edges create new sums/prods.
        # So we just pick arbitrary 35? No, we want to structure them.
        
        # Better Greedy:
        # 1. Pick an edge.
        # 2. Pick the next edge that shares a sum or product with existing edges.
        # If no such edge, pick one that minimizes new unique labels.
        
        remaining = list(range(len(all_edges)))
        random.shuffle(remaining) # Randomize to escape bad local minima in greedy
        
        for _ in range(m):
            if not remaining: break
            best_idx = -1
            best_new_sum = 1
            best_new_prod = 1
            best_tie = -1
            
            for idx in remaining:
                s = edge_sums[idx]
                p = edge_prods[idx]
                
                new_s = 0 if s in sum_counts else 1
                new_p = 0 if p in prod_counts else 1
                
                # We want to minimize the increase in max(distinct_sums, distinct_prods)
                # Approximation: minimize new_s + new_p
                # Tie break: prefer edges that have higher frequency potential?
                # For now, just min new_s + new_p. If tie, keep first.
                
                score = new_s + new_p
                if score < best_new_sum + best_new_prod:
                    best_new_sum = new_s
                    best_new_prod = new_p
                    best_idx = idx
                    best_tie = 0
                elif score == best_new_sum + best_new_prod:
                    # Tie break: prefer edges whose sum/prod are already common?
                    # Or just random.
                    if random.random() < 0.1:
                        best_idx = idx
                        best_new_sum = new_s
                        best_new_prod = new_p
            
            if best_idx != -1:
                current_edges.add(best_idx)
                sum_counts[edge_sums[best_idx]] += 1
                prod_counts[edge_prods[best_idx]] += 1
                remaining.remove(best_idx)
        
        # Local Search: Swap edges to improve
        current_raw = max(len(sum_counts), len(prod_counts))
        
        unselected = set(range(len(all_edges))) - current_edges
        unselected_list = list(unselected)
        
        improved = True
        while improved:
            improved = False
            # Try all swaps? 35 * 56 = 1960 swaps. Fast enough.
            for e_out in list(current_edges):
                s_out = edge_sums[e_out]
                p_out = edge_prods[e_out]
                
                # Temporarily remove e_out
                sum_counts[s_out] -= 1
                if sum_counts[s_out] == 0: del sum_counts[s_out]
                prod_counts[p_out] -= 1
                if prod_counts[p_out] == 0: del prod_counts[p_out]
                
                best_e_in = -1
                best_score_in = current_raw
                
                for e_in in unselected_list:
                    s_in = edge_sums[e_in]
                    p_in = edge_prods[e_in]
                    
                    # Check if adding e_in improves
                    new_sum_count = len(sum_counts) + (1 if s_in not in sum_counts else 0)
                    new_prod_count = len(prod_counts) + (1 if p_in not in prod_counts else 0)
                    new_raw = max(new_sum_count, new_prod_count)
                    
                    if new_raw < best_score_in:
                        best_score_in = new_raw
                        best_e_in = e_in
                
                if best_e_in != -1:
                    # Apply swap
                    current_edges.remove(e_out)
                    current_edges.add(best_e_in)
                    unselected_list.remove(best_e_in)
                    unselected_list.append(e_out)
                    
                    # Update counts
                    sum_counts[edge_sums[best_e_in]] += 1
                    prod_counts[edge_prods[best_e_in]] += 1
                    
                    current_raw = best_score_in
                    improved = True
                else:
                    # Revert removal of e_out
                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
        
        return list(current_edges), current_raw, sum_counts, prod_counts

    # Main Loop
    for vals in candidate_sets:
        if time.time() > deadline:
            break
            
        if not is_valid_values(vals):
            continue
            
        edges, score, _, _ = optimize_edges(vals)
        
        if score < best_score:
            best_score = score
            best_values = list(vals)
            best_edges = edges
            
            # Early exit if perfect or near perfect
            if best_score <= 10:
                break

    # Value Perturbation Local Search
    if best_values is not None:
        current_values = list(best_values)
        current_edges = best_edges
        current_score = best_score
        
        rng = random.Random(999)
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline and no_improve < max_no_improve:
            i = rng.randint(0, n - 1)
            old_val = current_values[i]
            
            # Generate candidates for new value
            candidates = []
            # Neighbors
            for d in range(-10, 11):
                nv = old_val + d
                if 1 <= nv <= max_val and nv != old_val and nv not in current_values:
                    candidates.append(nv)
            # Random
            candidates.append(rng.randint(1, max_val))
            
            found = False
            for nv in candidates:
                if nv in current_values:
                    continue
                    
                current_values[i] = nv
                
                edges, score, _, _ = optimize_edges(current_values)
                
                if score < current_score:
                    current_score = score
                    current_edges = edges
                    best_score = score
                    best_values = list(current_values)
                    best_edges = edges
                    no_improve = 0
                    found = True
                    break
                else:
                    current_values[i] = old_val
            
            if not found:
                no_improve += 1

    if best_values is None:
        best_values = list(range(1, n + 1))
        best_edges, best_score, _, _ = optimize_edges(best_values)

    # Final check and formatting
    # Ensure types are int
    final_values = [int(v) for v in best_values]
    final_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
