"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    best_result = None
    best_score = float('inf')
    
    # Strategy: 
    # 1. Generate candidate value sets that are "collision-friendly".
    #    Small values create more sum/product collisions.
    #    Arithmetic Progressions (AP) create many sum collisions (e.g., 1+4=2+3).
    #    However, APs create many distinct products if spread out.
    #    Let's try sets with small numbers, potentially with duplicates allowed? 
    #    No, "fourteen distinct positive integers".
    #    So we must pick 14 distinct ints.
    #    To minimize distinct sums, we want the set to be "additively degenerate".
    #    Small consecutive integers: {1..14} have sums in [3..27]. 25 distinct sums.
    #    Can we do better?
    #    If we use {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 
    #    sums are 3..27. Count = 25.
    #    Products: 1*2=2 ... 13*14=182. Many distinct products.
    #    
    #    Alternative: Use a set that minimizes product variety.
    #    If values are close, products are close but distinct.
    #    If values are powers of 2, products are distinct powers (bad for product count, good for sum count? No, sums of powers of 2 are unique in binary -> 2^14 sums, very bad).
    #    
    #    Let's analyze the trade-off.
    #    Max distinct sums for 14 distinct ints is at most (max+min-1) if they form an AP.
    #    For {1..14}, sums = 25.
    #    For {1,2,4,8,...}, sums = 2^14 (huge).
    #    So APs are good for sums.
    #    
    #    What about products for AP {1..14}?
    #    Products range from 2 to 182.
    #    How many distinct products?
    #    Let's check a few.
    #    1*2=2, 1*3=3, ..., 1*14=14.
    #    2*2 not allowed (distinct vertices).
    #    2*3=6, 2*4=8, 2*5=10...
    #    3*4=12, 3*5=15...
    #    There will be many collisions (e.g., 2*6=12, 3*4=12).
    #    But likely > 25 distinct products.
    #    
    #    Let's try to minimize the MAX of the two counts.
    #    If we can get sums down to 25, can we get products down to 25?
    #    Or can we get both to ~30?
    #    
    #    Let's try a few specific value sets and optimize edges for them.
    
    candidate_value_sets = []
    
    # Set 1: Consecutive integers 1..14
    candidate_value_sets.append(list(range(1, 15)))
    
    # Set 2: Consecutive integers 2..15
    candidate_value_sets.append(list(range(2, 16)))
    
    # Set 3: Small integers with a gap to break some product patterns?
    # Try {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} again but maybe shift?
    # Shifting doesn't change sum counts, only product counts increase.
    # So 1..14 is better for products than 2..15.
    
    # Set 4: Try to cluster values to increase product collisions.
    # e.g. {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    # What if we use {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20}?
    # This breaks the AP sum property. Sums will be more distinct.
    # AP is best for minimizing sum distinctness.
    
    # Let's stick with APs of small numbers.
    # Try different starting points?
    # {1..14} is best for products (smaller values).
    # Try {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    
    # Are there non-AP sets with fewer distinct sums?
    # For a set of size n, the minimum number of distinct sums is 2n-3 (for AP).
    # So 2*14-3 = 25 is the theoretical minimum for distinct sums.
    # So we cannot go below 25 distinct sums.
    # Therefore, the answer is at least 25.
    # Can we achieve 25 distinct sums AND 25 distinct products?
    # If yes, score is 25.
    # If products are > 25, score is determined by products.
    
    # Let's evaluate {1..14} with optimal edges.
    # To minimize distinct products, we should pick edges that reuse product values.
    # This is a graph problem: given a set of values, find 35 edges that minimize distinct products.
    
    # Heuristic for Edge Selection:
    # 1. Calculate all possible products for all pairs.
    # 2. Group pairs by product value.
    # 3. We need to select 35 pairs.
    # 4. To minimize distinct products, we want to pick as many pairs as possible from the same product group.
    # 5. However, we also care about sums.
    # 6. Let's prioritize minimizing the max(count_sums, count_prods).
    
    # Since min sums is 25, let's see if we can keep prods <= 25.
    # If not, we minimize prods.
    
    # Let's implement a greedy edge selector for a fixed value set.
    
    def optimize_edges(values, max_time=None):
        n = len(values)
        all_pairs = list(combinations(range(n), 2))
        
        # Precompute sums and products for all pairs
        pair_sums = {}
        pair_prods = {}
        for i, j in all_pairs:
            s = values[i] + values[j]
            p = values[i] * values[j]
            pair_sums[(i, j)] = s
            pair_prods[(i, j)] = p
        
        # Group pairs by product value
        prod_groups = {}
        for (i, j) in all_pairs:
            p = pair_prods[(i, j)]
            if p not in prod_groups:
                prod_groups[p] = []
            prod_groups[p].append((i, j))
        
        # Group pairs by sum value
        sum_groups = {}
        for (i, j) in all_pairs:
            s = pair_sums[(i, j)]
            if s not in sum_groups:
                sum_groups[s] = []
            sum_groups[s].append((i, j))
        
        def greedy_construct(values, all_pairs, pair_sums, pair_prods, prod_groups, sum_groups, m):
            """Greedy edge construction with tie-breaking."""
            current_edges = []
            current_sums = set()
            current_prods = set()
            used_pairs = set()
            
            for _ in range(m):
                best_pair = None
                best_cost = float('inf')
                best_tie_breaker = 0
                best_sum_freq = 0
                
                for i, j in all_pairs:
                    if (i, j) in used_pairs:
                        continue
                    
                    s = pair_sums[(i, j)]
                    p = pair_prods[(i, j)]
                    
                    cost = 0
                    if s not in current_sums:
                        cost += 1
                    if p not in current_prods:
                        cost += 1
                    
                    # Tie breaker: prefer pairs with higher product frequency, then sum frequency
                    prod_count = len(prod_groups[p])
                    sum_count = len(sum_groups[s])
                    
                    if (cost < best_cost or 
                        (cost == best_cost and prod_count > best_tie_breaker) or
                        (cost == best_cost and prod_count == best_tie_breaker and sum_count > best_sum_freq)):
                        best_cost = cost
                        best_pair = (i, j)
                        best_tie_breaker = prod_count
                        best_sum_freq = sum_count
                
                if best_pair is None:
                    break
                
                used_pairs.add(best_pair)
                current_edges.append(best_pair)
                current_sums.add(pair_sums[best_pair])
                current_prods.add(pair_prods[best_pair])
            
            return current_edges
        
        def local_search(edges, values, all_pairs, pair_sums, pair_prods, max_time=None):
            """Local search: try swapping edges to reduce distinct sums/products."""
            if not edges:
                return edges
            
            edges = list(edges)
            edge_set = set(edges)
            
            def compute_score(edges):
                sums = set()
                prods = set()
                for i, j in edges:
                    sums.add(pair_sums[(i, j)])
                    prods.add(pair_prods[(i, j)])
                return max(len(sums), len(prods))
            
            current_score = compute_score(edges)
            best_edges = list(edges)
            best_score = current_score
            
            deadline = time.time() + max_time if max_time else time.time() + 5.0
            
            while time.time() < deadline:
                improved = False
                # Try single edge swaps
                for idx in range(len(edges)):
                    old_edge = edges[idx]
                    old_s = pair_sums[old_edge]
                    old_p = pair_prods[old_edge]
                    
                    # Compute current sums/prods without this edge
                    sums_wo = set()
                    prods_wo = set()
                    for e in edges:
                        if e != old_edge:
                            sums_wo.add(pair_sums[e])
                            prods_wo.add(pair_prods[e])
                    
                    # Try replacing with each unused edge
                    for new_edge in all_pairs:
                        if new_edge in edge_set:
                            continue
                        
                        new_s = pair_sums[new_edge]
                        new_p = pair_prods[new_edge]
                        
                        new_sums = sums_wo | {new_s}
                        new_prods = prods_wo | {new_p}
                        new_score = max(len(new_sums), len(new_prods))
                        
                        if new_score < best_score:
                            # Accept the swap
                            edges[idx] = new_edge
                            edge_set.remove(old_edge)
                            edge_set.add(new_edge)
                            best_score = new_score
                            best_edges = list(edges)
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if not improved:
                    # Try 2-opt style: swap two edges
                    for i in range(len(edges)):
                        for j in range(i + 1, len(edges)):
                            e1 = edges[i]
                            e2 = edges[j]
                            s1, p1 = pair_sums[e1], pair_prods[e1]
                            s2, p2 = pair_sums[e2], pair_prods[e2]
                            
                            sums_wo = set()
                            prods_wo = set()
                            for k, e in enumerate(edges):
                                if k != i and k != j:
                                    sums_wo.add(pair_sums[e])
                                    prods_wo.add(pair_prods[e])
                            
                            # Try swapping e1 and e2 with two new edges
                            # This is expensive, so limit it
                            if time.time() > deadline:
                                break
                            
                            # Just try replacing one at a time in 2-opt
                            for new_e1 in all_pairs:
                                if new_e1 in edge_set:
                                    continue
                                s_new1 = pair_sums[new_e1]
                                p_new1 = pair_prods[new_e1]
                                
                                sums_new = sums_wo | {s_new1, s2}
                                prods_new = prods_wo | {p_new1, p2}
                                new_score = max(len(sums_new), len(prods_new))
                                
                                if new_score < best_score:
                                    edges[i] = new_e1
                                    edge_set.remove(e1)
                                    edge_set.add(new_e1)
                                    best_score = new_score
                                    best_edges = list(edges)
                                    improved = True
                                    break
                            
                            if improved:
                                break
                        if improved:
                            break
                
                if not improved:
                    break
            
            return best_edges
        
        # Run greedy construction
        edges = greedy_construct(values, all_pairs, pair_sums, pair_prods, prod_groups, sum_groups, m)
        
        # Run local search to improve
        if max_time is not None:
            remaining = max_time - (time.time() - (start_time if 'start_time' in dir() else time.time()))
            ls_time = max(0.5, min(remaining, 3.0))
        else:
            ls_time = 2.0
        
        edges = local_search(edges, values, all_pairs, pair_sums, pair_prods, ls_time)
        
        return edges

    # Try a few value sets
    # 1. {1..14}
    # 2. {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} (same)
    # 3. Maybe a set with more product collisions?
    #    e.g. {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
    #    What if we replace 14 with something that creates more product collisions?
    #    14*1=14, 14*2=28, 14*3=42...
    #    12*2=24, 12*3=36...
    #    13*2=26, 13*3=39...
    #    11*2=22, 11*3=33...
    #    10*2=20, 10*3=30...
    #    9*2=18, 9*3=27...
    #    8*2=16, 8*3=24 (collision with 12*2)
    #    7*2=14 (collision with 14*1)
    #    6*2=12 (collision with 3*4)
    #    5*2=10 (collision with 2*5)
    #    4*2=8 (collision with 1*8)
    #    3*2=6 (collision with 1*6)
    #    2*2 not allowed.
    
    #    It seems {1..14} has good collision potential.
    
    # Let's also try a random search around {1..14} to see if we can improve.
    # But first, let's just run the greedy on {1..14} and a few variations.
    
    value_sets_to_try = [
        list(range(1, 15)),
        list(range(2, 16)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16],
        # Sets with more product collisions
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 36],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 72],
        # Non-AP sets that might have better product structure
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 30],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 36],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 32],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 36],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 48],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 72],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 30],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 40],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 60],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 120],
    ]
    
    # Add some random sets with bias toward small values and product-friendly numbers
    for _ in range(30):
        if time.time() > deadline:
            break
        # Random set of 14 distinct ints from 1..25
        vals = random.sample(range(1, 26), 14)
        vals.sort()
        value_sets_to_try.append(vals)
    
    # Add sets with many divisors (more product collisions)
    divisor_rich = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20]
    value_sets_to_try.append(divisor_rich)
    divisor_rich2 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 24]
    value_sets_to_try.append(divisor_rich2)
    divisor_rich3 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 30]
    value_sets_to_try.append(divisor_rich3)
    divisor_rich4 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 36]
    value_sets_to_try.append(divisor_rich4)
    divisor_rich5 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 48]
    value_sets_to_try.append(divisor_rich5)
    divisor_rich6 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 60]
    value_sets_to_try.append(divisor_rich6)
    divisor_rich7 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 72]
    value_sets_to_try.append(divisor_rich7)
    divisor_rich8 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 90]
    value_sets_to_try.append(divisor_rich8)
    divisor_rich9 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 120]
    value_sets_to_try.append(divisor_rich9)
    divisor_rich10 = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 180]
    value_sets_to_try.append(divisor_rich10)
        
    for vals in value_sets_to_try:
        if time.time() > deadline:
            break
        
        # Compute time budget for this value set
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 0.5:
            break
        
        # Allocate more time to promising value sets
        ls_time = min(remaining * 0.3, 3.0)
        
        edges = optimize_edges(vals, ls_time)
        score = _raw_value(vals, edges)
        
        if score < best_score:
            best_score = score
            best_result = {'values': vals, 'edges': [list(e) for e in edges]}
            
            # If we hit the theoretical minimum for sums (25) and products are also low,
            # we might be near optimal.
            if best_score <= 25:
                break

    # If no result found (shouldn't happen), fallback
    if best_result is None:
        vals = list(range(1, 15))
        edges = optimize_edges(vals, 1.0)
        best_result = {'values': vals, 'edges': [list(e) for e in edges]}
        
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
# EVOLVE-BLOCK-END
