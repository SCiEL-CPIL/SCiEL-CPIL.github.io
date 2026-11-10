import time
import random
import math
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 14
    m = 35

    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    P = len(all_pairs)  # 91

    # Candidate value sets
    # 1. Consecutive integers (High collision in both sums and products for small numbers)
    # 2. Highly composite small numbers
    # 3. Powers of 2 (Low product distinct count, but high sum distinct count)
    # 4. Mixed
    
    candidate_values = [
        list(range(1, 15)),
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 18, 20, 24, 30, 36],
        [2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36, 40],
        [3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30],
        [1, 2, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30],
        [2, 3, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 27, 30],
        # Powers of 2
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        # Arithmetic Progression
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        # Mixed
        [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 144, 192],
    ]

    def evaluate_edges(values, edge_indices):
        """Calculate raw score for a given set of edge indices."""
        sums = set()
        prods = set()
        for idx in edge_indices:
            i, j = all_pairs[idx]
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def global_assignment_optimize(values, K, seed, time_limit=2.0):
        """
        Use Hungarian algorithm to assign m edges to K sum-buckets and K prod-buckets
        simultaneously, minimizing the number of distinct sums and products.
        
        Strategy: For a given K, we want to find edges such that at most K distinct
        sum values and at most K distinct product values are used.
        
        We formulate this as: pick K sum-values S1..SK and K prod-values P1..PK,
        then assign each of the m edges to a (Si, Pj) pair such that the edge's
        actual sum is in {S1..SK} and product is in {P1..PK}.
        
        Simplified approach: 
        1. Enumerate candidate sum sets and prod sets of size K
        2. For each combo, build assignment matrix and solve
        """
        start = time.time()
        n_pairs = len(all_pairs)
        
        # Precompute sum and product for each pair
        pair_sums = np.array([values[i] + values[j] for i, j in all_pairs], dtype=np.int64)
        pair_prods = np.array([values[i] * values[j] for i, j in all_pairs], dtype=np.int64)
        
        # Get unique sums and products
        unique_sums = np.unique(pair_sums)
        unique_prods = np.unique(pair_prods)
        
        # For Hungarian: we need to assign m edges to "slots"
        # Each slot corresponds to a (sum_value, prod_value) pair
        # But we only want at most K distinct sums and K distinct prods.
        
        # Alternative: iterate over subsets of sum values and prod values
        # This is combinatorially expensive, so we use a heuristic:
        # Pick the K most common sum values and K most common prod values
        
        # Better approach: Use a relaxed formulation
        # Create a cost matrix where rows are edges (n_pairs) and columns are 
        # "assignment slots". Each slot has a sum label and prod label.
        # We want to select m edges and assign them to slots such that
        # the number of distinct sum labels <= K and distinct prod labels <= K.
        
        # Simplified: Try all combinations of K sum values from top candidates
        # and K prod values from top candidates, then solve assignment.
        
        # First, identify candidate sum and prod values
        sum_counts = Counter(pair_sums.tolist())
        prod_counts = Counter(pair_prods.tolist())
        
        # Top sum candidates (by frequency)
        top_sums = [s for s, c in sum_counts.most_common(min(2*K, len(unique_sums)))]
        top_prods = [p for p, c in prod_counts.most_common(min(2*K, len(unique_prods)))]
        
        best_raw = K + 1  # worse than K
        best_edge_set = None
        
        # Try combinations of K sums from top_sums and K prods from top_prods
        # To limit search, try a few random subsets
        rng = random.Random(seed)
        
        n_sum_trials = min(20, len(top_sums))
        n_prod_trials = min(20, len(top_prods))
        
        for _ in range(min(50, n_sum_trials * n_prod_trials)):
            if time.time() - start > time_limit:
                break
            
            # Pick K sum values
            if len(top_sums) <= K:
                sum_subset = top_sums
            else:
                sum_subset = rng.sample(top_sums, K)
            
            # Pick K prod values
            if len(top_prods) <= K:
                prod_subset = top_prods
            else:
                prod_subset = rng.sample(top_prods, K)
            
            sum_set = set(sum_subset)
            prod_set = set(prod_subset)
            
            # Find edges where sum is in sum_set AND prod is in prod_set
            valid_edges = []
            for idx in range(n_pairs):
                if pair_sums[idx] in sum_set and pair_prods[idx] in prod_set:
                    valid_edges.append(idx)
            
            if len(valid_edges) < m:
                continue
            
            # We have valid edges. Now we need to select exactly m of them.
            # Since all valid edges use only K sums and K prods, any m of them
            # will have at most K distinct sums and K distinct prods.
            # So just pick m edges from valid_edges.
            
            # But we want to minimize the actual distinct count, which might be < K.
            # For now, pick m edges and evaluate.
            
            # Pick m edges - try to maximize collisions
            # Sort by frequency of their sum and prod
            edge_scores = []
            for idx in valid_edges:
                s = pair_sums[idx]
                p = pair_prods[idx]
                score = sum_counts[s] + prod_counts[p]
                edge_scores.append((score, idx))
            
            edge_scores.sort(reverse=True)
            selected = [idx for _, idx in edge_scores[:m]]
            
            raw = evaluate_edges(values, selected)
            if raw < best_raw:
                best_raw = raw
                best_edge_set = selected
            
            if best_raw <= K:
                break
        
        return best_raw, best_edge_set

    def greedy_init(values, seed):
        """Greedy initialization: pick edges that minimize new distinct labels."""
        rng = random.Random(seed)
        ps = [values[i] + values[j] for i, j in all_pairs]
        pp = [values[i] * values[j] for i, j in all_pairs]
        
        used_s = set()
        used_p = set()
        selected = []
        remaining = list(range(P))
        rng.shuffle(remaining)
        
        for _ in range(m):
            best_idx = -1
            best_score = float('inf')
            best_tie = -1
            for k in remaining:
                s = ps[k]
                p = pp[k]
                # If we add this edge, how many new distinct sums/prods do we get?
                new_s = 1 if s not in used_s else 0
                new_p = 1 if p not in used_p else 0
                # We want to minimize the resulting max count
                # Approximate score: current max + potential increase
                # Better: simulate the set size
                ns = len(used_s) + new_s
                np_ = len(used_p) + new_p
                score = max(ns, np_)
                # Tie breaker: prefer edges that add to existing sets (new_s=0, new_p=0)
                tie = new_s + new_p
                
                if score < best_score or (score == best_score and tie < best_tie):
                    best_score = score
                    best_tie = tie
                    best_idx = k
            
            selected.append(best_idx)
            used_s.add(ps[best_idx])
            used_p.add(pp[best_idx])
            remaining.remove(best_idx)
        
        return selected

    def simulated_annealing(values, initial_edges, max_iters, seed):
        """Optimize edges using Simulated Annealing."""
        rng = random.Random(seed)
        ps = [values[i] + values[j] for i, j in all_pairs]
        pp = [values[i] * values[j] for i, j in all_pairs]

        edge_set = set(initial_edges)
        unselected = set(range(P)) - edge_set
        
        sum_counts = Counter()
        prod_counts = Counter()
        for idx in edge_set:
            sum_counts[ps[idx]] += 1
            prod_counts[pp[idx]] += 1

        def get_raw():
            return max(len(sum_counts), len(prod_counts))

        current_raw = get_raw()
        best_raw = current_raw
        best_edge_set = set(edge_set)

        T = 3.0
        T_min = 0.01
        cooling = 0.9999

        for it in range(max_iters):
            if time.time() > deadline:
                break

            if rng.random() < 0.8:
                # 1-swap
                if not edge_set or not unselected: continue
                remove_idx = rng.choice(list(edge_set))
                add_idx = rng.choice(list(unselected))
                if remove_idx == add_idx: continue

                rs, rp = ps[remove_idx], pp[remove_idx]
                as_, ap = ps[add_idx], pp[add_idx]

                # Apply
                sum_counts[rs] -= 1
                if sum_counts[rs] == 0: del sum_counts[rs]
                prod_counts[rp] -= 1
                if prod_counts[rp] == 0: del prod_counts[rp]
                
                sum_counts[as_] += 1
                prod_counts[ap] += 1

                new_raw = get_raw()
                delta = new_raw - current_raw
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    edge_set.remove(remove_idx)
                    edge_set.add(add_idx)
                    unselected.add(remove_idx)
                    unselected.remove(add_idx)
                    current_raw = new_raw
                    if current_raw < best_raw:
                        best_raw = current_raw
                        best_edge_set = set(edge_set)
                else:
                    # Revert
                    sum_counts[as_] -= 1
                    if sum_counts[as_] == 0: del sum_counts[as_]
                    prod_counts[ap] -= 1
                    if prod_counts[ap] == 0: del prod_counts[ap]
                    
                    sum_counts[rs] += 1
                    prod_counts[rp] += 1
            else:
                # 2-swap
                if len(edge_set) < 2 or len(unselected) < 2: continue
                rem1, rem2 = rng.sample(list(edge_set), 2)
                add1, add2 = rng.sample(list(unselected), 2)
                
                s1, p1 = ps[rem1], pp[rem1]
                s2, p2 = ps[rem2], pp[rem2]
                sa1, pa1 = ps[add1], pp[add1]
                sa2, pa2 = ps[add2], pp[add2]

                # Apply
                sum_counts[s1] -= 1
                if sum_counts[s1] == 0: del sum_counts[s1]
                prod_counts[p1] -= 1
                if prod_counts[p1] == 0: del prod_counts[p1]
                
                sum_counts[s2] -= 1
                if sum_counts[s2] == 0: del sum_counts[s2]
                prod_counts[p2] -= 1
                if prod_counts[p2] == 0: del prod_counts[p2]
                
                sum_counts[sa1] += 1
                prod_counts[pa1] += 1
                sum_counts[sa2] += 1
                prod_counts[pa2] += 1

                new_raw = get_raw()
                delta = new_raw - current_raw
                
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    edge_set.discard(rem1)
                    edge_set.discard(rem2)
                    edge_set.add(add1)
                    edge_set.add(add2)
                    unselected.add(rem1)
                    unselected.add(rem2)
                    unselected.discard(add1)
                    unselected.discard(add2)
                    current_raw = new_raw
                    if current_raw < best_raw:
                        best_raw = current_raw
                        best_edge_set = set(edge_set)
                else:
                    # Revert
                    sum_counts[sa1] -= 1
                    if sum_counts[sa1] == 0: del sum_counts[sa1]
                    prod_counts[pa1] -= 1
                    if prod_counts[pa1] == 0: del prod_counts[pa1]
                    
                    sum_counts[sa2] -= 1
                    if sum_counts[sa2] == 0: del sum_counts[sa2]
                    prod_counts[pa2] -= 1
                    if prod_counts[pa2] == 0: del prod_counts[pa2]
                    
                    sum_counts[s1] += 1
                    prod_counts[p1] += 1
                    sum_counts[s2] += 1
                    prod_counts[p2] += 1

            T *= cooling
            if T < T_min:
                T = T_min

        return list(best_edge_set), best_raw

    best_raw = float('inf')
    best_values = None
    best_edges = None

    # Sort candidates by a heuristic lower bound
    def lower_bound(values):
        ps = set()
        pp = set()
        for i, j in all_pairs:
            ps.add(values[i] + values[j])
            pp.add(values[i] * values[j])
        return max(len(ps), len(pp))

    scored_candidates = []
    for vi, values in enumerate(candidate_values):
        if len(set(values)) != n or any(v < 1 or v > 10000 for v in values):
            continue
        lb = lower_bound(values)
        scored_candidates.append((lb, vi, values))
    
    scored_candidates.sort(key=lambda x: x[0])

    # Phase 1: Global assignment optimization for each candidate
    for lb, vi, values in scored_candidates:
        if time.time() > deadline:
            break
        
        # Try K from 25 down to 15 (or until we find a good solution)
        for K in range(30, 14, -1):
            if time.time() > deadline:
                break
            
            # Time budget per (values, K) combo
            remaining = deadline - time.time()
            time_limit = min(3.0, remaining * 0.3)
            
            if time_limit < 0.5:
                break
            
            ga_raw, ga_edges = global_assignment_optimize(values, K, seed=vi * 1000 + K, time_limit=time_limit)
            
            if ga_raw < best_raw and ga_edges is not None:
                best_raw = ga_raw
                best_values = [int(v) for v in values]
                best_edges = [[int(i), int(j)] for i, j in (all_pairs[idx] for idx in ga_edges)]
            
            # If we found a solution with raw <= K, we can stop increasing K
            if best_raw <= K:
                break
        
        # If we have a very good result, stop
        if best_raw <= 18:
            break

    # Phase 2: Local search polish on the best result
    if best_values is not None and best_edges is not None:
        # Convert edges to indices
        edge_idx_set = set()
        edge_to_idx = {}
        for idx, (i, j) in enumerate(all_pairs):
            edge_to_idx[(i, j)] = idx
        
        for e in best_edges:
            key = (e[0], e[1]) if e[0] < e[1] else (e[1], e[0])
            if key in edge_to_idx:
                edge_idx_set.add(edge_to_idx[key])
        
        # SA polish
        if len(edge_idx_set) == m:
            initial_list = list(edge_idx_set)
            sa_edges, sa_raw = simulated_annealing(best_values, initial_list, 30000, seed=42)
            if sa_raw < best_raw:
                best_raw = sa_raw
                best_edges = [[int(i), int(j)] for i, j in (all_pairs[idx] for idx in sa_edges)]
    
    if best_values is None:
        # Fallback
        best_values = list(range(1, n + 1))
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n)][:m]

    # Final validation and type conversion
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]

    return {'values': best_values, 'edges': best_edges}
