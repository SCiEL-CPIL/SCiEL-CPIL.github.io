import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for safety
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)
    edge_index_map = {e: i for i, e in enumerate(all_edges)}
    
    best_score = float('inf')
    best_result = None
    
    # Candidate value sets
    candidates = [
        list(range(1, 15)),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 18],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 18, 20],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32],
        [1, 2, 4, 8, 16, 3, 6, 12, 24, 9, 18, 36, 72, 5],
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 36],
        [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 14, 15, 18, 21],
        [2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 18, 20, 24, 30],
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 20, 24],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15],
    ]
    
    # Add some random small sets
    random.seed(42)
    for _ in range(20):
        pool = list(range(1, 40))
        vals = random.sample(pool, n)
        candidates.append(sorted(vals))
        
    # Add some sets with high multiplicative structure
    # Divisors of 2520 (2^3 * 3^2 * 5 * 7) are 48 in number. Pick 14.
    divs_2520 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 36, 40, 42, 45, 56, 60, 63, 70, 72, 84, 90, 105, 120, 126, 140, 168, 180, 210, 252, 280, 315, 360, 420, 504, 630, 840, 1260, 2520]
    for i in range(0, 10):
        # Take a random subset of 14
        subset = random.sample(divs_2520, 14)
        candidates.append(sorted(subset))

    def optimize_edges_for_values(values):
        # Precompute sums and products for all edges
        edge_sums = {}
        edge_prods = {}
        for u, v in all_edges:
            edge_sums[(u, v)] = values[u] + values[v]
            edge_prods[(u, v)] = values[u] * values[v]
        
        best_local_score = float('inf')
        best_local_edges = None
        
        # Multiple restarts with different initializations
        for restart in range(20):
            if time.time() > deadline - 2:
                break
                
            # Greedy Initialization
            # Sort edges by how "common" their sum/prod is among all edges
            # Or just random? Greedy by collision potential is better.
            
            # Let's try a greedy approach:
            # Start with empty set.
            # Add edge that minimizes increase in max(distinct_sums, distinct_prods)
            # To do this efficiently, we track counts.
            
            current_edges = set()
            sum_counts = {}
            prod_counts = {}
            
            # Calculate initial "attractiveness" of each edge
            # An edge is attractive if its sum or product matches another edge's sum or product
            # We will just iterate and pick the best available edge greedily
            
            remaining_edges = list(all_edges)
            
            for _ in range(m):
                if time.time() > deadline - 2:
                    break
                best_edge = None
                best_score_increase = float('inf')
                
                # To save time, we don't check ALL remaining edges if we have many.
                # But 91 is small.
                # However, doing this for 35 steps * 20 restarts * 30 value sets might be slow.
                # Let's use a simpler init: Random, then Local Search.
                
                # Actually, let's just use Random Init + Strong Local Search.
                # The local search with O(1) swaps is very fast.
                break # Skip greedy init, go to random init for speed in this loop structure
                
            current_edges = set(random.sample(all_edges, m))
            
            # Calculate initial score
            sum_counts = {}
            prod_counts = {}
            for u, v in current_edges:
                s = edge_sums[(u, v)]
                p = edge_prods[(u, v)]
                sum_counts[s] = sum_counts.get(s, 0) + 1
                prod_counts[p] = prod_counts.get(p, 0) + 1
                
            distinct_sums = len(sum_counts)
            distinct_prods = len(prod_counts)
            current_score = max(distinct_sums, distinct_prods)
            
            best_edges_restart = list(current_edges)
            best_score_restart = current_score
            
            # Local Search
            no_improve = 0
            max_no_improve = 50
            temp = 1.0 # For simulated annealing if needed, but let's stick to hill climbing first
            
            while no_improve < max_no_improve and time.time() < deadline - 2:
                improved = False
                
                # Try all swaps? 35 * (91-35) = ~2000 swaps.
                # Each swap is O(1) update + O(1) check.
                # 2000 * 20 restarts * 30 sets = 1.2M swaps. Should be fast in Python?
                # Maybe too slow. Let's limit swaps per iteration or use random swaps.
                
                # Random Swap Approach
                for _ in range(100): # 100 random swaps per iteration
                    if time.time() > deadline - 2:
                        break
                    
                    # Pick edge to remove
                    e_remove = random.choice(list(current_edges))
                    u1, v1 = e_remove
                    s1 = edge_sums[e_remove]
                    p1 = edge_prods[e_remove]
                    
                    # Pick edge to add
                    e_add = random.choice(all_edges)
                    if e_add in current_edges:
                        continue
                    u2, v2 = e_add
                    s2 = edge_sums[e_add]
                    p2 = edge_prods[e_add]
                    
                    # Calculate delta
                    # Remove e_remove
                    sum_counts[s1] -= 1
                    if sum_counts[s1] == 0:
                        del sum_counts[s1]
                        ds_new = distinct_sums - 1
                    else:
                        ds_new = distinct_sums
                        
                    prod_counts[p1] -= 1
                    if prod_counts[p1] == 0:
                        del prod_counts[p1]
                        dp_new = distinct_prods - 1
                    else:
                        dp_new = distinct_prods
                        
                    # Add e_add
                    if s2 in sum_counts:
                        sum_counts[s2] += 1
                        ds_new2 = ds_new
                    else:
                        sum_counts[s2] = 1
                        ds_new2 = ds_new + 1
                        
                    if p2 in prod_counts:
                        prod_counts[p2] += 1
                        dp_new2 = dp_new
                    else:
                        prod_counts[p2] = 1
                        dp_new2 = dp_new + 1
                        
                    new_score = max(ds_new2, dp_new2)
                    
                    if new_score < current_score:
                        # Accept
                        current_edges.discard(e_remove)
                        current_edges.add(e_add)
                        distinct_sums = ds_new2
                        distinct_prods = dp_new2
                        current_score = new_score
                        no_improve = 0
                        improved = True
                        
                        if current_score < best_score_restart:
                            best_score_restart = current_score
                            best_edges_restart = list(current_edges)
                    else:
                        # Reject: revert counts
                        # Revert Add
                        if s2 in sum_counts:
                            sum_counts[s2] -= 1
                            if sum_counts[s2] == 0:
                                del sum_counts[s2]
                        # Revert Remove
                        if s1 in sum_counts:
                            sum_counts[s1] += 1
                        else:
                            sum_counts[s1] = 1
                            
                        if p2 in prod_counts:
                            prod_counts[p2] -= 1
                            if prod_counts[p2] == 0:
                                del prod_counts[p2]
                        # Revert Remove
                        if p1 in prod_counts:
                            prod_counts[p1] += 1
                        else:
                            prod_counts[p1] = 1
                            
                        no_improve += 1
                        
                if not improved:
                    no_improve += 10 # Penalize non-improving iterations more?
                    # Actually, the loop above increments no_improve on reject.
                    # Let's just break if we haven't improved in a while.
                    # The inner loop runs 100 times. If none improved, no_improve increased by up to 100.
                    # So it will break soon.
                    
            if best_score_restart < best_local_score:
                best_local_score = best_score_restart
                best_local_edges = best_edges_restart
                
        return best_local_edges, best_local_score

    # Iterate over candidates
    # Shuffle candidates to avoid bias
    random.shuffle(candidates)
    
    for vals in candidates:
        if time.time() > deadline - 5:
            break
            
        # Check validity
        if len(set(vals)) != n or any(v < 1 or v > 10000 for v in vals):
            continue
            
        edges, local_score = optimize_edges_for_values(vals)
        
        if edges and local_score < best_score:
            best_score = local_score
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
            
        # If we found a very good score, we can stop or do more
        if best_score <= 15:
            # Try to improve further with more restarts on this specific set
            # Or just break if we are close to theoretical lower bound?
            # Min distinct sums for 14 numbers:
            # If numbers are 1..14, sums are 3..27 (25 distinct).
            # If we pick edges carefully, can we reduce this?
            # Yes, if we only pick edges that sum to a few values.
            # e.g. 1+10=11, 2+9=11, 3+8=11, 4+7=11, 5+6=11.
            # We can have many edges with sum 11.
            # So 15 is a very good target.
            pass

    if best_result is None:
        # Fallback
        best_result = {'values': list(range(1, 15)), 'edges': [list(e) for e in all_edges[:35]]}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
