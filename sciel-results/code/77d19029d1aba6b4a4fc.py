import time
import random
import math
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    edge_indices = {e: i for i, e in enumerate(all_edges)}
    
    def score_of_mask(mask, edge_sums, edge_prods):
        """
        Calculate distinct sums and prods for a given edge mask.
        mask: list of booleans or set of edge indices.
        """
        sums = set()
        prods = set()
        for i in range(len(all_edges)):
            if mask[i]:
                sums.add(edge_sums[i])
                prods.add(edge_prods[i])
        return max(len(sums), len(prods))

    def optimize_edges(values):
        """
        Find 35 edges for given values minimizing max(distinct_sums, distinct_prods).
        """
        # Precompute sums and products for all 91 edges
        edge_sums = [0] * 91
        edge_prods = [0] * 91
        for i, (u, v) in enumerate(all_edges):
            edge_sums[i] = values[u] + values[v]
            edge_prods[i] = values[u] * values[v]
        
        # Get unique sums and prods to map to smaller indices? 
        # Not strictly necessary for sets, but good for counting.
        
        def get_score_from_counts(sum_counts, prod_counts, n_distinct_s, n_distinct_p):
            return max(n_distinct_s, n_distinct_p)

        def greedy_init():
            mask = [False] * 91
            sum_counts = defaultdict(int)
            prod_counts = defaultdict(int)
            n_distinct_s = 0
            n_distinct_p = 0
            selected = 0
            
            # Sort edges by a heuristic? 
            # Let's just pick the one that adds fewest new distincts.
            # To speed up, we can pre-sort edges by frequency of sum/prod in the whole graph?
            # No, dynamic.
            
            for _ in range(m):
                best_i = -1
                best_new_s = float('inf')
                best_new_p = float('inf')
                best_score = float('inf')
                
                # Optimization: Only consider edges not yet selected
                candidates = [i for i in range(91) if not mask[i]]
                
                for i in candidates:
                    s = edge_sums[i]
                    p = edge_prods[i]
                    new_s = 0 if sum_counts[s] > 0 else 1
                    new_p = 0 if prod_counts[p] > 0 else 1
                    # We want to minimize the resulting max count
                    # current max is max(n_distinct_s, n_distinct_p)
                    # new max will be max(n_distinct_s + new_s, n_distinct_p + new_p)
                    # We want to minimize this.
                    # If both are 0, it's best.
                    # If one is 0, one is 1, it depends on which side is larger.
                    # Simple heuristic: minimize (new_s + new_p), then break ties by randomness.
                    cost = new_s + new_p
                    if cost < best_score:
                        best_score = cost
                        best_i = i
                        best_new_s = new_s
                        best_new_p = new_p
                    elif cost == best_score:
                        # Tie-breaking: prefer if it doesn't increase the larger side
                        cur_max = max(n_distinct_s, n_distinct_p)
                        new_max_s = n_distinct_s + new_s
                        new_max_p = n_distinct_p + new_p
                        # If current best also has same cost, pick the one that keeps max lower
                        # This logic is getting complex. Let's just use random tie-breaking for speed.
                        if random.random() < 0.5:
                            best_i = i
                            best_new_s = new_s
                            best_new_p = new_p

                if best_i == -1:
                    break
                    
                mask[best_i] = True
                s = edge_sums[best_i]
                p = edge_prods[best_i]
                if sum_counts[s] == 0:
                    n_distinct_s += 1
                sum_counts[s] += 1
                if prod_counts[p] == 0:
                    n_distinct_p += 1
                prod_counts[p] += 1
                selected += 1
                
            return mask, sum_counts, prod_counts, n_distinct_s, n_distinct_p

        # Initial greedy
        mask, sum_counts, prod_counts, n_distinct_s, n_distinct_p = greedy_init()
        current_score = max(n_distinct_s, n_distinct_p)
        best_score = current_score
        best_mask = list(mask)
        
        # Local Search
        # We will perform swaps: remove one selected, add one unselected.
        # Or 1-move: replace one edge with another.
        
        no_improve = 0
        it = 0
        max_iters = 2000
        
        while it < max_iters and time.time() < deadline - 10.0:
            it += 1
            # Pick a random selected edge to remove
            selected_indices = [i for i in range(91) if mask[i]]
            if not selected_indices:
                break
            i_out = random.choice(selected_indices)
            
            # Pick a random unselected edge to add
            unselected_indices = [i for i in range(91) if not mask[i]]
            if not unselected_indices:
                break
            i_in = random.choice(unselected_indices)
            
            # Tentative change
            s_out = edge_sums[i_out]
            p_out = edge_prods[i_out]
            s_in = edge_sums[i_in]
            p_in = edge_prods[i_in]
            
            # Update counts
            sum_counts[s_out] -= 1
            prod_counts[p_out] -= 1
            sum_counts[s_in] += 1
            prod_counts[p_in] += 1
            
            # Update distinct counts
            if sum_counts[s_out] == 0:
                n_distinct_s -= 1
            if sum_counts[s_in] == 1:
                n_distinct_s += 1
                
            if prod_counts[p_out] == 0:
                n_distinct_p -= 1
            if prod_counts[p_in] == 1:
                n_distinct_p += 1
            
            new_score = max(n_distinct_s, n_distinct_p)
            
            if new_score < current_score:
                current_score = new_score
                mask[i_out] = False
                mask[i_in] = True
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_mask = list(mask)
            else:
                # Revert
                sum_counts[s_out] += 1
                prod_counts[p_out] += 1
                sum_counts[s_in] -= 1
                prod_counts[p_in] -= 1
                
                if sum_counts[s_out] == 1:
                    n_distinct_s += 1
                if sum_counts[s_in] == 0:
                    n_distinct_s -= 1
                    
                if prod_counts[p_out] == 1:
                    n_distinct_p += 1
                if prod_counts[p_in] == 0:
                    n_distinct_p -= 1
                    
                no_improve += 1
                # Simulated Annealing style acceptance?
                # For now, just strict improvement or random restart if stuck.
                if no_improve > 100 and random.random() < 0.05:
                    # Accept worse move to escape local min
                    mask[i_out] = False
                    mask[i_in] = True
                    current_score = new_score
                    no_improve = 0
                    # Note: counts are currently in the "new" state because we reverted them?
                    # Wait, I reverted the counts. So I need to apply the change again.
                    sum_counts[s_out] -= 1
                    prod_counts[p_out] -= 1
                    sum_counts[s_in] += 1
                    prod_counts[p_in] += 1
                    if sum_counts[s_out] == 0:
                        n_distinct_s -= 1
                    if sum_counts[s_in] == 1:
                        n_distinct_s += 1
                    if prod_counts[p_out] == 0:
                        n_distinct_p -= 1
                    if prod_counts[p_in] == 1:
                        n_distinct_p += 1
                    current_score = max(n_distinct_s, n_distinct_p)

        # Convert best_mask to edge list
        best_edges = [all_edges[i] for i in range(91) if best_mask[i]]
        return best_edges, best_score

    def generate_value_candidates():
        candidates = []
        
        # 1. Arithmetic Progressions
        for a in range(1, 20):
            for d in range(1, 10):
                vals = [a + i * d for i in range(n)]
                if len(set(vals)) == n and all(v <= 10000 for v in vals):
                    candidates.append(vals)
        
        # 2. "Sum-Constant" Pairs
        # 7 pairs summing to S
        for S in range(20, 100, 5):
            vals = []
            for i in range(7):
                a = random.randint(1, S - 1)
                b = S - a
                if a != b and a not in vals and b not in vals:
                    vals.extend([a, b])
            if len(vals) == n:
                candidates.append(vals)
            else:
                # Fill remaining
                while len(vals) < n:
                    v = random.randint(1, 100)
                    if v not in vals:
                        vals.append(v)
                candidates.append(vals)
                
        # 3. Small Random Integers (1-50)
        for _ in range(50):
            vals = random.sample(range(1, 50), n)
            candidates.append(vals)
            
        # 4. Small Random Integers (1-100)
        for _ in range(30):
            vals = random.sample(range(1, 100), n)
            candidates.append(vals)

        # 5. Mixed Small/Large
        for _ in range(20):
            small = random.sample(range(1, 20), 7)
            large = random.sample(range(50, 200), 7)
            vals = small + large
            random.shuffle(vals)
            if len(set(vals)) == n:
                candidates.append(vals)
                
        return candidates

    best_score_overall = float('inf')
    best_vals_overall = None
    best_edges_overall = None
    
    value_candidates = generate_value_candidates()
    random.shuffle(value_candidates)
    
    # Process candidates
    for vals in value_candidates:
        if time.time() > deadline - 15.0:
            break
            
        # Ensure distinct and valid
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
            
        edges, score = optimize_edges(vals)
        
        if score < best_score_overall:
            best_score_overall = score
            best_vals_overall = list(vals)
            best_edges_overall = list(edges)
            
        # If we hit a very low score, we might spend more time polishing?
        # For now, just move to next candidate.

    # Fallback
    if best_vals_overall is None:
        best_vals_overall = list(range(1, 15))
        best_edges_overall = list(all_edges[:m])

    final_values = [int(v) for v in best_vals_overall]
    final_edges = [[int(u), int(v)] for u, v in best_edges_overall]
    
    return {'values': final_values, 'edges': final_edges}
