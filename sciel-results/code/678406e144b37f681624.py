"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 84.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    max_val = 10000
    
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)  # 91
    edge_indices = list(range(num_all_edges))
    
    def fast_evaluate(values, edges):
        """Optimized evaluation using local variable access."""
        vals = values
        s_set = set()
        p_set = set()
        for u, v in edges:
            s_set.add(vals[u] + vals[v])
            p_set.add(vals[u] * vals[v])
        return max(len(s_set), len(p_set))
    
    def greedy_edges(values, all_edges, rng, time_limit=None):
        """Pick edges greedily to maximize reuse of existing sums/products."""
        sums_seen = set()
        prods_seen = set()
        selected = []
        remaining = list(all_edges)
        rng.shuffle(remaining)
        
        for _ in range(m):
            if time_limit and (time.time() - start_time) > time_limit:
                break
            best_edge = None
            best_score = float('inf')
            for edge in remaining:
                u, v = edge
                s = values[u] + values[v]
                p = values[u] * values[v]
                # Score: prefer edges that reuse existing sums/products
                score = 0
                if s not in sums_seen:
                    score += 1
                if p not in prods_seen:
                    score += 1
                if score < best_score:
                    best_score = score
                    best_edge = edge
            if best_edge is None:
                break
            selected.append(best_edge)
            remaining.remove(best_edge)
            u, v = best_edge
            sums_seen.add(values[u] + values[v])
            prods_seen.add(values[u] * values[v])
        
        return selected
    
    def local_search_edges(values, edges, rng, time_limit):
        """Local search on edges: swap one edge for another."""
        edges_set = set(tuple(e) for e in edges)
        current_score = fast_evaluate(values, edges)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(300):
                if (time.time() - start_time) > time_limit:
                    break
                i = rng.randint(0, len(edges)-1)
                new_edge = rng.choice(all_edges)
                if tuple(new_edge) in edges_set:
                    continue
                
                old_edge = edges[i]
                edges[i] = new_edge
                edges_set.discard(tuple(old_edge))
                edges_set.add(tuple(new_edge))
                
                new_score = fast_evaluate(values, edges)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    edges[i] = old_edge
                    edges_set.discard(tuple(new_edge))
                    edges_set.add(tuple(old_edge))
        
        return edges, current_score
    
    def local_search_values(values, edges, rng, time_limit):
        """Local search on values: perturb one value at a time."""
        current_score = fast_evaluate(values, edges)
        val_set = set(values)
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for _ in range(100):
                if (time.time() - start_time) > time_limit:
                    break
                idx = rng.randint(0, n-1)
                old_val = values[idx]
                candidates = set()
                candidates.add(old_val + rng.choice([-3,-2,-1,1,2,3]))
                candidates.add(old_val + rng.choice([-10,-5,5,10]))
                for base in [2, 3, 4, 5, 6, 8, 10, 12]:
                    candidates.add(base * rng.randint(1, 150))
                candidates.add(rng.randint(1, max_val))
                
                for new_val in candidates:
                    if new_val < 1 or new_val > max_val:
                        continue
                    if new_val in val_set:
                        continue
                    values[idx] = new_val
                    val_set.discard(old_val)
                    val_set.add(new_val)
                    new_score = fast_evaluate(values, edges)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        values[idx] = old_val
                        val_set.discard(new_val)
                        val_set.add(old_val)
        
        return values, current_score
    
    def find_edges_for_values(values, target_k, time_limit):
        """
        Find 35 edges such that the number of distinct sums <= target_k 
        and number of distinct products <= target_k.
        Uses greedy with constraint checking.
        """
        edge_sums = []
        edge_prods = []
        for u, v in all_edges:
            edge_sums.append(values[u] + values[v])
            edge_prods.append(values[u] * values[v])
        
        unique_sums = set(edge_sums)
        unique_prods = set(edge_prods)
        
        if len(unique_sums) > target_k or len(unique_prods) > target_k:
            return None
        
        sum_to_edges = {}
        for i, s in enumerate(edge_sums):
            if s not in sum_to_edges:
                sum_to_edges[s] = []
            sum_to_edges[s].append(i)
        
        prod_to_edges = {}
        for i, p in enumerate(edge_prods):
            if p not in prod_to_edges:
                prod_to_edges[p] = []
            prod_to_edges[p].append(i)
        
        sums_used = set()
        prods_used = set()
        selected = []
        remaining = list(range(num_all_edges))
        
        def edge_priority(i):
            s_count = len(sum_to_edges[edge_sums[i]])
            p_count = len(prod_to_edges[edge_prods[i]])
            return (s_count, p_count)
        
        remaining.sort(key=edge_priority, reverse=True)
        
        for i in remaining:
            if len(selected) >= m:
                break
            s = edge_sums[i]
            p = edge_prods[i]
            if len(sums_used) < target_k or s in sums_used:
                if len(prods_used) < target_k or p in prods_used:
                    selected.append(i)
                    sums_used.add(s)
                    prods_used.add(p)
        
        if len(selected) >= m:
            return [all_edges[i] for i in selected[:m]]
        
        return None
    
    def find_edges_optimal(values, time_limit):
        """Find the best set of 35 edges for given values using iterative deepening on K."""
        best_edges = None
        best_k = float('inf')
        
        for k in range(1, 40):
            if (time.time() - start_time) > time_limit:
                break
            edges = find_edges_for_values(values, k, time_limit)
            if edges is not None:
                actual_score = fast_evaluate(values, edges)
                if actual_score <= k:
                    best_edges = edges
                    best_k = k
                    break
        
        if best_edges is None:
            edges = greedy_edges(values, all_edges, random.Random(42), time_limit)
            best_edges = edges
            best_k = fast_evaluate(values, edges)
        
        return best_edges, best_k
    
    def generate_structured_values(rng, time_limit):
        """Generate values with high factor overlap to encourage product collisions."""
        # Strategy: pick a base set of primes, then create values as products of subsets
        # This creates many shared factors
        small_primes = [2, 3, 5, 7, 11, 13]
        
        # Generate values by multiplying subsets of small primes
        candidates = set()
        for r in range(1, 5):
            for subset in combinations(small_primes, r):
                prod = 1
                for p in subset:
                    prod *= p
                if 1 <= prod <= max_val:
                    candidates.add(prod)
        
        # Also add some random multiples for diversity
        for _ in range(20):
            base = rng.choice([2, 3, 4, 6, 8, 10, 12, 15, 20, 24, 30])
            candidates.add(base * rng.randint(1, 300))
        
        # Ensure we have at least 14 distinct values
        val_list = list(candidates)
        rng.shuffle(val_list)
        
        if len(val_list) < n:
            # Fill with random values
            existing = set(val_list)
            while len(val_list) < n:
                v = rng.randint(1, max_val)
                if v not in existing:
                    val_list.append(v)
                    existing.add(v)
        
        return val_list[:n]
    
    def generate_arithmetic_progression_values(rng, time_limit):
        """Generate values in arithmetic progression for sum collisions."""
        a = rng.randint(1, 50)
        d = rng.randint(1, 20)
        values = [a + i * d for i in range(n)]
        
        # Ensure distinct and within bounds
        seen = set()
        final_values = []
        for v in values:
            if v not in seen and 1 <= v <= max_val:
                final_values.append(v)
                seen.add(v)
        
        while len(final_values) < n:
            v = rng.randint(1, max_val)
            if v not in seen:
                final_values.append(v)
                seen.add(v)
        
        return final_values[:n]
    
    def generate_mixed_values(rng, time_limit):
        """Mix different strategies for diversity."""
        strategy = rng.randint(0, 2)
        if strategy == 0:
            return generate_structured_values(rng, time_limit)
        elif strategy == 1:
            return generate_arithmetic_progression_values(rng, time_limit)
        else:
            # Random with bias towards small numbers
            values = []
            seen = set()
            while len(values) < n:
                if rng.random() < 0.7:
                    v = rng.randint(1, 100)
                else:
                    v = rng.randint(1, max_val)
                if v not in seen:
                    values.append(v)
                    seen.add(v)
            return values
    
    # Main search loop
    best_overall_score = float('inf')
    best_result = None
    rng = random.Random(42)
    
    # Initial value strategies - focus on values that create many sum/product collisions
    value_strategies = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58],
        [1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144],
        [2, 3, 4, 6, 12, 18, 24, 36, 48, 72, 96, 144, 192, 288],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
        [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
        [2, 4, 6, 12, 18, 24, 36, 48, 60, 72, 84, 96, 108, 120],
        [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192],
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
    ]
    
    # Process curated strategies first
    for init_values in value_strategies:
        if (time.time() - start_time) > deadline:
            break
        
        # 1. Find optimal edges for these values
        edges, score = find_edges_optimal(init_values, deadline - 5.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': edges}
        
        # 2. Local search on edges
        if (time.time() - start_time) < deadline - 5.0:
            edges2, score2 = local_search_edges(list(init_values), edges, rng, deadline - 5.0)
            if score2 < best_overall_score:
                best_overall_score = score2
                best_result = {'values': list(init_values), 'edges': edges2}
        
        # 3. Local search on values
        if (time.time() - start_time) < deadline - 5.0:
            values, score3 = local_search_values(list(init_values), edges2, rng, deadline - 5.0)
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values), 'edges': edges2}
            
            # 4. Refine edges with new values
            if (time.time() - start_time) < deadline - 5.0:
                edges4, score4 = local_search_edges(values, edges2, rng, deadline - 5.0)
                if score4 < best_overall_score:
                    best_overall_score = score4
                    best_result = {'values': list(values), 'edges': edges4}
    
    # Then try generated strategies
    num_generated = 15
    for gen_idx in range(num_generated):
        if (time.time() - start_time) > deadline - 10.0:
            break
        
        # Generate new values
        gen_rng = random.Random(42 + gen_idx * 1000)
        init_values = generate_mixed_values(gen_rng, deadline - 10.0)
        
        # 1. Find optimal edges
        edges, score = find_edges_optimal(init_values, deadline - 10.0)
        
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': list(init_values), 'edges': edges}
        
        # 2. Local search on edges
        if (time.time() - start_time) < deadline - 10.0:
            edges2, score2 = local_search_edges(list(init_values), edges, rng, deadline - 10.0)
            if score2 < best_overall_score:
                best_overall_score = score2
                best_result = {'values': list(init_values), 'edges': edges2}
        
        # 3. Local search on values
        if (time.time() - start_time) < deadline - 10.0:
            values, score3 = local_search_values(list(init_values), edges2, rng, deadline - 10.0)
            if score3 < best_overall_score:
                best_overall_score = score3
                best_result = {'values': list(values), 'edges': edges2}
            
            # 4. Refine edges with new values
            if (time.time() - start_time) < deadline - 10.0:
                edges4, score4 = local_search_edges(values, edges2, rng, deadline - 10.0)
                if score4 < best_overall_score:
                    best_overall_score = score4
                    best_result = {'values': list(values), 'edges': edges4}
    
    # Final polish
    if best_result is not None and (time.time() - start_time) < deadline - 2.0:
        values = list(best_result['values'])
        edges = list(best_result['edges'])
        
        for _ in range(15):
            if (time.time() - start_time) > deadline - 2.0:
                break
            values, s1 = local_search_values(values, edges, rng, deadline - 2.0)
            edges, s2 = local_search_edges(values, edges, rng, deadline - 2.0)
            if s2 < best_overall_score:
                best_overall_score = s2
                best_result = {'values': list(values), 'edges': edges}
    
    if best_result is None:
        best_result = {
            'values': list(range(1, n+1)),
            'edges': [list(e) for e in list(all_edges)[:m]]
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
