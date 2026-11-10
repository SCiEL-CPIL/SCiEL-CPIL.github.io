import time
import random
from collections import Counter
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 14
    m = 35
    maximum = 10000
    
    # Precompute all pairs
    all_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.append((i, j))
    
    # Candidate value sets: small, factor-rich
    # Adding more structured sets
    candidate_values = [
        [2,3,4,5,6,8,9,10,12,15,16,18,20,24],
        [1,2,3,4,5,6,7,8,9,10,12,15,18,20],
        [2,4,6,8,10,12,14,16,18,20,24,30,36,40],
        [3,4,5,6,8,9,10,12,15,16,18,20,24,30],
        [1,2,3,4,6,8,9,12,15,18,20,24,30,36],
        [2,4,6,9,10,12,15,16,18,20,24,27,30,36],
        [1,2,4,6,8,9,10,12,15,16,18,20,24,30],
        [2,3,6,8,9,10,12,15,16,18,20,24,27,30],
        # New structured sets
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Powers of 2
        [1, 3, 9, 27, 81, 243, 729, 2187, 6561, 10000, 5000, 2500, 1250, 625], # Mixed powers
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43], # Primes
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], # Consecutive
        [2, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 9000, 9500], # Growth
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # All 1s (invalid distinct, but good for structure if relaxed? No, must be distinct)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 9999],
    ]
    
    # Also generate some random small value sets
    random.seed(12345)
    for _ in range(50):
        vals = random.sample(range(1, 50), n)
        candidate_values.append(vals)
    
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    def evaluate_edges(values, edge_indices):
        """Given values and a set of edge indices (into all_pairs), compute raw value."""
        sums = set()
        prods = set()
        for idx in edge_indices:
            i, j = all_pairs[idx]
            v = values[i]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    def greedy_edge_selection(values, seed=0):
        """Greedy: pick edges that minimize new distinct sums+prods."""
        rng = random.Random(seed)
        # Precompute sum/prod for each pair
        pair_sum = [values[i] + values[j] for i, j in all_pairs]
        pair_prod = [values[i] * values[j] for i, j in all_pairs]
        
        used_sums = set()
        used_prods = set()
        selected = []
        remaining = list(range(len(all_pairs)))
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_idx = -1
            best_score = float('inf')
            best_tie = -1
            for k in remaining:
                s = pair_sum[k]
                p = pair_prod[k]
                new_sums = len(used_sums) + (1 if s not in used_sums else 0)
                new_prods = len(used_prods) + (1 if p not in used_prods else 0)
                score = max(new_sums, new_prods)
                if score < best_score:
                    best_score = score
                    best_idx = k
                    best_tie = 0
                elif score == best_score:
                    # Tie-break: prefer pairs whose sum/prod are already common
                    tie = (1 if s in used_sums else 0) + (1 if p in used_prods else 0)
                    if tie > best_tie:
                        best_tie = tie
                        best_idx = k
            if best_idx == -1:
                break
            selected.append(best_idx)
            used_sums.add(pair_sum[best_idx])
            used_prods.add(pair_prod[best_idx])
            remaining.remove(best_idx)
        
        return selected
    
    def simulated_annealing(values, initial_edges, max_iters=50000, seed=0):
        """Simulated Annealing with 1-swap and 2-swap moves."""
        rng = random.Random(seed)
        edge_set = set(initial_edges)
        
        # Precompute
        pair_sum = [values[i] + values[j] for i, j in all_pairs]
        pair_prod = [values[i] * values[j] for i, j in all_pairs]
        
        # Maintain counts
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in edge_set:
            sum_counts[pair_sum[idx]] += 1
            prod_counts[pair_prod[idx]] += 1
        
        def get_raw():
            return max(len(sum_counts), len(prod_counts))
        
        current_raw = get_raw()
        best_raw_so_far = current_raw
        best_edge_set = set(edge_set)
        
        T = 5.0
        T_min = 0.1
        cooling = 0.9999
        
        # Use a set for unselected to avoid list.remove errors and duplicates
        unselected = set(range(len(all_pairs))) - edge_set
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if not edge_set or not unselected:
                continue

            # Choose move type: 1-swap (80%) or 2-swap (20%)
            if rng.random() < 0.8:
                # 1-swap
                remove_idx = rng.choice(list(edge_set))
                # Try a few random candidates for add to increase acceptance chance
                candidates = rng.sample(list(unselected), min(5, len(unselected)))
                best_candidate = None
                best_delta = float('inf')
                best_new_raw = current_raw
                
                for add_idx in candidates:
                    if add_idx == remove_idx:
                        continue
                    
                    # Simulate
                    rs, rp = pair_sum[remove_idx], pair_prod[remove_idx]
                    as_, ap = pair_sum[add_idx], pair_prod[add_idx]
                    
                    # Modify
                    sum_counts[rs] -= 1
                    if sum_counts[rs] == 0: del sum_counts[rs]
                    sum_counts[as_] += 1
                    
                    prod_counts[rp] -= 1
                    if prod_counts[rp] == 0: del prod_counts[rp]
                    prod_counts[ap] += 1
                    
                    new_raw = get_raw()
                    delta = new_raw - current_raw
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_candidate = add_idx
                        best_new_raw = new_raw
                    
                    # Revert this candidate
                    sum_counts[rs] += 1
                    if sum_counts[as_] == 1: del sum_counts[as_]
                    else: sum_counts[as_] -= 1
                    
                    prod_counts[rp] += 1
                    if prod_counts[ap] == 1: del prod_counts[ap]
                    else: prod_counts[ap] -= 1
                
                if best_candidate is None:
                    continue
                
                # Apply best_candidate
                add_idx = best_candidate
                rs, rp = pair_sum[remove_idx], pair_prod[remove_idx]
                as_, ap = pair_sum[add_idx], pair_prod[add_idx]
                
                sum_counts[rs] -= 1
                if sum_counts[rs] == 0: del sum_counts[rs]
                sum_counts[as_] += 1
                
                prod_counts[rp] -= 1
                if prod_counts[rp] == 0: del prod_counts[rp]
                prod_counts[ap] += 1
                
                new_raw = best_new_raw
                delta = new_raw - current_raw
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    # Accept
                    edge_set.remove(remove_idx)
                    edge_set.add(add_idx)
                    # Update unselected set
                    unselected.add(remove_idx)
                    unselected.remove(add_idx)
                    
                    current_raw = new_raw
                    if current_raw < best_raw_so_far:
                        best_raw_so_far = current_raw
                        best_edge_set = set(edge_set)
                else:
                    # Revert
                    sum_counts[rs] += 1
                    if sum_counts[as_] == 1: del sum_counts[as_]
                    else: sum_counts[as_] -= 1
                    
                    prod_counts[rp] += 1
                    if prod_counts[ap] == 1: del prod_counts[ap]
                    else: prod_counts[ap] -= 1
            else:
                # 2-swap (remove 2, add 2)
                if len(edge_set) < 2 or len(unselected) < 2:
                    continue
                
                rem1, rem2 = rng.sample(list(edge_set), 2)
                add1, add2 = rng.sample(list(unselected), 2)
                
                # Ensure no overlap (should be guaranteed by sets, but safe)
                if rem1 in (add1, add2) or rem2 in (add1, add2):
                    continue
                
                # Apply
                # Remove rem1, rem2
                s1, p1 = pair_sum[rem1], pair_prod[rem1]
                s2, p2 = pair_sum[rem2], pair_prod[rem2]
                
                sum_counts[s1] -= 1
                if sum_counts[s1] == 0: del sum_counts[s1]
                prod_counts[p1] -= 1
                if prod_counts[p1] == 0: del prod_counts[p1]
                
                sum_counts[s2] -= 1
                if sum_counts[s2] == 0: del sum_counts[s2]
                prod_counts[p2] -= 1
                if prod_counts[p2] == 0: del prod_counts[p2]
                
                # Add add1, add2
                sa1, pa1 = pair_sum[add1], pair_prod[add1]
                sa2, pa2 = pair_sum[add2], pair_prod[add2]
                
                sum_counts[sa1] += 1
                prod_counts[pa1] += 1
                sum_counts[sa2] += 1
                prod_counts[pa2] += 1
                
                new_raw = get_raw()
                delta = new_raw - current_raw
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    # Accept
                    edge_set.discard(rem1)
                    edge_set.discard(rem2)
                    edge_set.add(add1)
                    edge_set.add(add2)
                    
                    # Update unselected set
                    unselected.add(rem1)
                    unselected.add(rem2)
                    unselected.discard(add1)
                    unselected.discard(add2)
                    
                    current_raw = new_raw
                    if current_raw < best_raw_so_far:
                        best_raw_so_far = current_raw
                        best_edge_set = set(edge_set)
                else:
                    # Revert
                    # Remove add1, add2
                    sum_counts[sa1] -= 1
                    if sum_counts[sa1] == 0: del sum_counts[sa1]
                    prod_counts[pa1] -= 1
                    if prod_counts[pa1] == 0: del prod_counts[pa1]
                    
                    sum_counts[sa2] -= 1
                    if sum_counts[sa2] == 0: del sum_counts[sa2]
                    prod_counts[pa2] -= 1
                    if prod_counts[pa2] == 0: del prod_counts[pa2]
                    
                    # Add rem1, rem2
                    sum_counts[s1] += 1
                    prod_counts[p1] += 1
                    sum_counts[s2] += 1
                    prod_counts[p2] += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        return list(best_edge_set), best_raw_so_far
    
    # Try each candidate value set
    for vi, values in enumerate(candidate_values):
        if time.time() > deadline:
            break
        
        if len(set(values)) != n or any(v < 1 or v > maximum for v in values):
            continue
        
        # Multiple greedy starts + SA
        for restart in range(5):
            if time.time() > deadline:
                break
            
            greedy_edges = greedy_edge_selection(values, seed=restart * 100 + vi)
            sa_edges, sa_raw = simulated_annealing(values, greedy_edges, max_iters=20000, seed=restart * 1000 + vi)
            
            if sa_raw < best_raw:
                best_raw = sa_raw
                best_values = list(values)
                best_edges = [list(all_pairs[idx]) for idx in sa_edges]
                
                if best_raw <= 15:
                    return {'values': best_values, 'edges': best_edges}
    
    # Fallback
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges = [list(e) for e in __import__('itertools').combinations(range(n), 2)[:m]]
    
    return {'values': best_values, 'edges': best_edges}
