import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    all_edges_set = set(all_edges)

    def score_of(values, edge_list):
        sums = set()
        prods = set()
        for u, v in edge_list:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def greedy_edge_selection(values, target_edges=m):
        """
        Select edges one by one to minimize the growth of distinct sums/products.
        """
        edges = []
        edges_set = set()
        current_sums = set()
        current_prods = set()
        
        # Pre-compute all potential edge contributions
        # To speed up, we can just iterate
        for _ in range(target_edges):
            best_edge = None
            best_score = float('inf')
            best_sums = None
            best_prods = None
            
            # Shuffle candidates to break ties randomly and avoid bias
            candidates = list(all_edges_set - edges_set)
            random.shuffle(candidates)
            
            for u, v in candidates:
                s = values[u] + values[v]
                p = values[u] * values[v]
                
                new_sums_count = len(current_sums) + (1 if s not in current_sums else 0)
                new_prods_count = len(current_prods) + (1 if p not in current_prods else 0)
                new_score = max(new_sums_count, new_prods_count)
                
                if new_score < best_score:
                    best_score = new_score
                    best_edge = (u, v)
                    # We don't need to store the full sets for every candidate, 
                    # but we need the final ones for the winner.
                    # However, checking membership is O(1).
                    # We can just pick the edge with min max(counts).
                    # Tie-breaking: prefer edges where both s and p are already present?
                    # Actually, if score is same, prefer edge that adds 0 new sums AND 0 new prods.
                    # Let's refine the comparison key: (new_score, 1 if s not in current_sums else 0, 1 if p not in current_prods else 0)
                    pass
            
            # Re-evaluate the best edge properly to update sets
            if best_edge is not None:
                u, v = best_edge
                s = values[u] + values[v]
                p = values[u] * values[v]
                current_sums.add(s)
                current_prods.add(p)
                edges.append(best_edge)
                edges_set.add(best_edge)
                
        return edges

    def local_search_values(values, edges, budget_iters=50000):
        """
        Hill climbing on values using a curated set of collision-prone numbers.
        """
        vals = list(values)
        cur_score = score_of(vals, edges)
        best_score = cur_score
        best_vals = list(vals)
        
        # Curated list of numbers that are highly composite or small
        # Divisors of 2520 (LCM of 1..10) and 5040, plus small primes
        candidate_pool = set()
        for base in [2520, 5040, 720, 360]:
            for i in range(1, int(math.sqrt(base)) + 1):
                if base % i == 0:
                    candidate_pool.add(i)
                    candidate_pool.add(base // i)
        # Add small numbers
        for i in range(1, 30):
            candidate_pool.add(i)
        
        # Filter to <= 10000
        candidate_pool = [x for x in candidate_pool if x <= 10000]
        
        T = 5.0
        T_min = 0.1
        cooling_rate = 0.999
        
        it = 0
        while it < budget_iters and time.time() < deadline - 5.0:
            it += 1
            
            idx = random.randint(0, n - 1)
            old_val = vals[idx]
            
            # Pick a new value from the pool, preferring values close to current or just random from pool
            # To ensure distinctness, we must check against other values
            new_val = random.choice(candidate_pool)
            if new_val == old_val:
                continue
            if any(v == new_val for k, v in enumerate(vals) if k != idx):
                continue
                
            vals[idx] = new_val
            new_score = score_of(vals, edges)
            
            delta = new_score - cur_score
            if delta <= 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_vals = list(vals)
            else:
                vals[idx] = old_val
                
            T = max(T_min, T * cooling_rate)
            
        return best_score, best_vals

    best_score = float('inf')
    best_vals = None
    best_edges = None

    # Strategy 1: Greedy Edges with Small Values
    # Initialize with small distinct integers
    init_vals_1 = list(range(1, n + 1))
    edges_1 = greedy_edge_selection(init_vals_1)
    sc1, v1 = local_search_values(init_vals_1, edges_1, budget_iters=20000)
    if sc1 < best_score:
        best_score, best_vals, best_edges = sc1, v1, edges_1

    # Strategy 2: Greedy Edges with Divisors of 72
    # 72 has many divisors: 1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72
    # Need 14 distinct. Add 16, 54? Or just use 1..14 but permute?
    # Let's use a mix of small composites.
    divisors_72 = [1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72, 16, 27] # 27 is not divisor of 72 but small composite
    # Ensure distinct
    divisors_72 = list(set(divisors_72))
    if len(divisors_72) < n:
        # Fill with next small composites
        for i in range(30, 100):
            if i not in divisors_72:
                divisors_72.append(i)
            if len(divisors_72) == n:
                break
    random.shuffle(divisors_72)
    init_vals_2 = divisors_72[:n]
    edges_2 = greedy_edge_selection(init_vals_2)
    sc2, v2 = local_search_values(init_vals_2, edges_2, budget_iters=20000)
    if sc2 < best_score:
        best_score, best_vals, best_edges = sc2, v2, edges_2

    # Strategy 3: Random Start + Greedy Refinement
    # Pick random small values, then use greedy to pick edges? 
    # Greedy edge selection is deterministic given values. 
    # So we just need diverse value starts.
    
    # Strategy 4: Powers of 2 and 3
    # 1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36
    vals_3 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    random.shuffle(vals_3)
    edges_3 = greedy_edge_selection(vals_3)
    sc3, v3 = local_search_values(vals_3, edges_3, budget_iters=20000)
    if sc3 < best_score:
        best_score, best_vals, best_edges = sc3, v3, edges_3

    # Final Polish: Local search on the best result found, allowing edge swaps too?
    # The directive says structural change. Let's stick to value search on the greedy graph.
    # But we can try a few more value configurations if time permits.
    
    if time.time() < deadline - 15.0:
        # Try a wider pool for local search on the best found
        sc4, v4 = local_search_values(best_vals, best_edges, budget_iters=30000)
        if sc4 < best_score:
            best_score, best_vals, best_edges = sc4, v4, best_edges

    if best_vals is None:
        best_vals = list(range(1, n + 1))
        best_edges = list(all_edges[:m])

    final_values = [int(v) for v in best_vals]
    final_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': final_values, 'edges': final_edges}
