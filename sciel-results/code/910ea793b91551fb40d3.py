"""Strategy: Deterministic MST-based edge selection with value-set local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 14
    m = 35
    
    def evaluate(vals):
        """Find best 35 edges for given values using MST-like greedy approach."""
        # Pre-compute sums and products for all pairs
        all_pairs = list(itertools.combinations(range(n), 2))
        pair_sums = {}
        pair_prods = {}
        for u, v in all_pairs:
            s = vals[u] + vals[v]
            p = vals[u] * vals[v]
            pair_sums[(u, v)] = s
            pair_prods[(u, v)] = p
        
        # Greedy selection:
        # We want to pick 35 edges such that the set of sums and set of prods are small.
        # This is hard. 
        # Approximation: 
        # 1. Count frequency of each sum and product across all pairs.
        # 2. Edges with high frequency sums/prods are "cheap" to add.
        # 3. Build a graph where weight of edge (u,v) is 
        #    (1 if sum not seen yet else 0) + (1 if prod not seen yet else 0) ?
        #    No, that depends on order.
        
        # Better heuristic:
        # Sort all pairs by (freq_sum, freq_prod) descending?
        # Or: Start with empty set. Repeatedly add edge that minimizes increase in distinct sums/prods.
        # If tie, prefer higher frequency.
        
        selected = set()
        current_sums = set()
        current_prods = set()
        
        # Pre-compute counts of sums and prods in the full graph
        sum_counts = {}
        prod_counts = {}
        for e in all_pairs:
            s = pair_sums[e]
            p = pair_prods[e]
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        # Sort candidates by "potential"
        # An edge is good if its sum and product are already common or already selected.
        
        # To make this fast, we'll do a few passes.
        # Pass 1: Pick edges with highest sum_freq and prod_freq.
        
        # Actually, a simpler robust heuristic:
        # Random restarts of greedy selection.
        
        best_raw = float('inf')
        best_edges = None
        
        # Try a few different greedy orderings
        orderings = [
            lambda e: (sum_counts[pair_sums[e]], prod_counts[pair_prods[e]]),
            lambda e: (sum_counts[pair_sums[e]] + prod_counts[pair_prods[e]]),
            lambda e: (1/sum_counts[pair_sums[e]], 1/prod_counts[pair_prods[e]]) # Prefer rare? No, prefer common to merge.
        ]
        
        # Also try random orderings
        for _ in range(5):
            orderings.append(lambda e, _r=random.random: _r())
        
        for key_func in orderings:
            if time.time() > deadline - 1:
                break
                
            # Sort all pairs by key_func (descending for frequency, ascending for random?)
            # For frequency, we want high frequency first.
            # For random, any order.
            
            # We need a stable, fast sort.
            # Let's just sort by a tuple.
            # If key_func returns (freq_s, freq_p), we sort descending.
            
            # To handle "descending" vs "ascending" uniformly, we can negate if needed.
            # Let's assume key_func returns a value where HIGHER is better.
            
            sorted_pairs = sorted(all_pairs, key=key_func, reverse=True)
            
            selected = set()
            current_sums = set()
            current_prods = set()
            
            for e in sorted_pairs:
                if len(selected) == m:
                    break
                s = pair_sums[e]
                p = pair_prods[e]
                selected.add(e)
                current_sums.add(s)
                current_prods.add(p)
            
            if len(selected) == m:
                raw = max(len(current_sums), len(current_prods))
                if raw < best_raw:
                    best_raw = raw
                    best_edges = set(selected)
        
        # Local search on the best found set
        if best_edges:
            current_edges = set(best_edges)
            current_sums = set(pair_sums[e] for e in current_edges)
            current_prods = set(pair_prods[e] for e in current_edges)
            current_raw = max(len(current_sums), len(current_prods))
            
            # Simple local search: swap edges
            # Try swapping one edge with another to reduce raw.
            # Since m is small (35), and N is small (14), we can afford some computation.
            
            # We'll do a limited number of swaps.
            for _ in range(100):
                if time.time() > deadline - 1:
                    break
                improved = False
                # Pick a random edge in current
                e_in = random.choice(list(current_edges))
                # Pick a random edge NOT in current
                e_out = random.choice(all_pairs)
                while e_out in current_edges:
                    e_out = random.choice(all_pairs)
                
                # Check if swap improves
                new_edges = (current_edges - {e_in}) | {e_out}
                new_sums = set(pair_sums[e] for e in new_edges)
                new_prods = set(pair_prods[e] for e in new_edges)
                new_raw = max(len(new_sums), len(new_prods))
                
                if new_raw < current_raw:
                    current_edges = new_edges
                    current_sums = new_sums
                    current_prods = new_prods
                    current_raw = new_raw
                    improved = True
                    best_raw = current_raw
                    best_edges = set(current_edges)
                
                if not improved and random.random() < 0.1:
                    # Accept worse moves occasionally?
                    # For now, stick to strict improvement to be safe and fast.
                    pass
        
        return best_raw, best_edges

    # 1. Generate diverse candidate value sets
    candidates = []
    
    # Strategy A: Small consecutive integers
    for start in range(1, 10):
        for step in range(1, 5):
            vals = [start + i*step for i in range(n)]
            if all(v <= 10000 and v > 0 for v in vals) and len(set(vals)) == n:
                candidates.append(vals)
                
    # Strategy B: Geometric progressions
    for base in range(2, 10):
        for ratio in [2, 3, 4, 5, 6]:
            vals = []
            v = base
            for _ in range(n):
                if v > 10000: break
                vals.append(v)
                v *= ratio
            if len(vals) == n:
                candidates.append(vals)
                
    # Strategy C: Mixed small primes/numbers to encourage product collisions
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    
    # Strategy D: Random search for "collision-rich" sets
    for _ in range(50):
        if time.time() > deadline - 10: break
        vals = sorted(random.sample(range(1, 200), n))
        if len(set(vals)) == n:
            candidates.append(vals)
            
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    best_raw_overall = float('inf')
    best_result = None

    # Evaluate each candidate
    for vals in unique_candidates:
        if time.time() > deadline - 5:
            break
            
        raw, edges = evaluate(vals)
        
        if raw < best_raw_overall:
            best_raw_overall = raw
            best_result = {
                'values': [int(x) for x in vals],
                'edges': [[int(u), int(v)] for u, v in edges]
            }
            best_result['edges'].sort()

    # If no result found, fallback
    if best_result is None:
        vals = list(range(1, 15))
        edges = list(itertools.combinations(range(n), 2))[:m]
        best_result = {
            'values': [int(x) for x in vals],
            'edges': [[int(u), int(v)] for u, v in edges]
        }

    return best_result
# EVOLVE-BLOCK-END
