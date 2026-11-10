import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    all_edges = list(combinations(range(n), 2))
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))
    
    def greedy_select_edges(values, K, time_limit):
        """Select exactly m edges such that #distinct sums <= K and #distinct prods <= K."""
        # Precompute sum and product for each edge
        edge_sum = {}
        edge_prod = {}
        for i, j in all_edges:
            edge_sum[(i, j)] = values[i] + values[j]
            edge_prod[(i, j)] = values[i] * values[j]
        
        # Sort edges by how "popular" their sum and product are
        # Edges whose sum/product already appears frequently are more valuable
        sum_freq = defaultdict(int)
        prod_freq = defaultdict(int)
        for e in all_edges:
            sum_freq[edge_sum[e]] += 1
            prod_freq[edge_prod[e]] += 1
        
        def edge_priority(e):
            return (sum_freq[edge_sum[e]], prod_freq[edge_prod[e]])
        
        # Try multiple orderings
        best_edges = None
        best_score = float('inf')
        
        for attempt in range(20):
            if time.time() >= time_limit:
                break
            
            # Different sorting strategies
            if attempt == 0:
                sorted_edges = sorted(all_edges, key=edge_priority, reverse=True)
            elif attempt == 1:
                sorted_edges = sorted(all_edges, key=lambda e: edge_sum[e])
            elif attempt == 2:
                sorted_edges = sorted(all_edges, key=lambda e: edge_prod[e])
            else:
                # Random perturbation
                shuffled = all_edges[:]
                random.shuffle(shuffled)
                # Sort by priority but with some randomness
                sorted_edges = sorted(shuffled, key=edge_priority, reverse=True)
            
            # Greedy: add edges that don't exceed K distinct sums/prods
            selected = []
            sum_set = set()
            prod_set = set()
            
            for e in sorted_edges:
                if len(selected) >= m:
                    break
                s = edge_sum[e]
                p = edge_prod[e]
                if len(sum_set) < K and s not in sum_set:
                    sum_set.add(s)
                if len(prod_set) < K and p not in prod_set:
                    prod_set.add(p)
                if len(sum_set) <= K and len(prod_set) <= K:
                    selected.append(e)
            
            if len(selected) == m:
                score = max(len(sum_set), len(prod_set))
                if score < best_score:
                    best_score = score
                    best_edges = selected[:]
        
        return best_edges, best_score
    
    def local_search_edges(values, initial_edges, K, time_limit):
        """Improve edge selection via swaps while keeping #sums, #prods <= K."""
        if not initial_edges:
            return None, float('inf')
        
        edge_sum = {}
        edge_prod = {}
        for i, j in all_edges:
            edge_sum[(i, j)] = values[i] + values[j]
            edge_prod[(i, j)] = values[i] * values[j]
        
        current_edges = list(initial_edges)
        current_sum_set = set(edge_sum[e] for e in current_edges)
        current_prod_set = set(edge_prod[e] for e in current_edges)
        current_score = max(len(current_sum_set), len(current_prod_set))
        
        best_edges = list(current_edges)
        best_score = current_score
        
        edge_set = set(current_edges)
        
        while time.time() < time_limit:
            improved = False
            # Try swapping: remove one edge, add one not in set
            for rem_idx, rem_edge in enumerate(current_edges):
                if time.time() >= time_limit:
                    break
                rem_s = edge_sum[rem_edge]
                rem_p = edge_prod[rem_edge]
                
                for add_edge in all_edges:
                    if add_edge in edge_set:
                        continue
                    add_s = edge_sum[add_edge]
                    add_p = edge_prod[add_edge]
                    
                    # Compute new sets
                    new_sum_set = set(current_sum_set)
                    new_prod_set = set(current_prod_set)
                    
                    # Remove rem_edge's contributions
                    if new_sum_set and rem_s in new_sum_set:
                        # Check if any other edge has this sum
                        count = 0
                        for e in current_edges:
                            if e != rem_edge and edge_sum[e] == rem_s:
                                count += 1
                                break
                        if count == 0:
                            new_sum_set.discard(rem_s)
                    
                    if new_prod_set and rem_p in new_prod_set:
                        count = 0
                        for e in current_edges:
                            if e != rem_edge and edge_prod[e] == rem_p:
                                count += 1
                                break
                        if count == 0:
                            new_prod_set.discard(rem_p)
                    
                    # Add add_edge's contributions
                    new_sum_set.add(add_s)
                    new_prod_set.add(add_p)
                    
                    new_score = max(len(new_sum_set), len(new_prod_set))
                    
                    if new_score < best_score:
                        # Apply swap
                        current_edges[rem_idx] = add_edge
                        edge_set.discard(rem_edge)
                        edge_set.add(add_edge)
                        current_sum_set = new_sum_set
                        current_prod_set = new_prod_set
                        best_score = new_score
                        best_edges = list(current_edges)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_edges, best_score
    
    def find_best_K(values, time_limit):
        """Find minimum K such that we can select m edges with <= K distinct sums and prods."""
        # Lower bound: at least ceil(m/14) for sums (14 vertices, max 14 distinct sums per vertex pair)
        # Actually, min possible distinct sums is small. Let's binary search K.
        
        # First, find the minimum possible distinct sums/prods for any m edges
        edge_sum = {}
        edge_prod = {}
        for i, j in all_edges:
            edge_sum[(i, j)] = values[i] + values[j]
            edge_prod[(i, j)] = values[i] * values[j]
        
        all_sums = set(edge_sum.values())
        all_prods = set(edge_prod.values())
        
        # Try K from small to large
        for K in range(3, min(len(all_sums), len(all_prods)) + 1):
            if time.time() >= time_limit:
                break
            
            edges, score = greedy_select_edges(values, K, time_limit)
            if edges is not None and len(edges) == m:
                # Try to improve with local search
                improved_edges, improved_score = local_search_edges(values, edges, K, time_limit)
                if improved_edges is not None:
                    edges = improved_edges
                    score = improved_score
                return edges, score
        
        return None, float('inf')
    
    # Generate structured value sets
    value_candidates = []
    
    # 1. Small consecutive: 1..14
    value_candidates.append(list(range(1, n + 1)))
    
    # 2. Arithmetic progressions
    for start in range(1, 10):
        for step in range(1, 10):
            vals = [start + i * step for i in range(n)]
            if max(vals) <= maximum and min(vals) >= 1 and len(set(vals)) == n:
                value_candidates.append(vals)
    
    # 3. Geometric-like sequences
    for base in range(2, 8):
        vals = [base ** i for i in range(n)]
        if max(vals) <= maximum and min(vals) >= 1 and len(set(vals)) == n:
            value_candidates.append(vals)
    
    # 4. Powers of 2
    vals = [2 ** i for i in range(n)]
    if max(vals) <= maximum and len(set(vals)) == n:
        value_candidates.append(vals)
    
    # 5. Squares
    vals = [(i + 1) ** 2 for i in range(n)]
    if max(vals) <= maximum and len(set(vals)) == n:
        value_candidates.append(vals)
    
    # 6. Cubes
    vals = [(i + 1) ** 3 for i in range(n)]
    if max(vals) <= maximum and len(set(vals)) == n:
        value_candidates.append(vals)
    
    # 7. Primes (first 14 primes)
    def is_prime(x):
        if x < 2:
            return False
        for d in range(2, int(x ** 0.5) + 1):
            if x % d == 0:
                return False
        return True
    primes = []
    x = 2
    while len(primes) < n:
        if is_prime(x):
            primes.append(x)
        x += 1
    if max(primes) <= maximum:
        value_candidates.append(primes)
    
    # 8. Random sets with structure
    random.seed(42)
    for _ in range(50):
        # Generate random distinct values
        vals = random.sample(range(1, maximum + 1), n)
        value_candidates.append(sorted(vals))
    
    # 9. Pairs that create many collisions
    # Try values where a+b = c+d for many pairs
    # E.g., symmetric sets
    for center in range(5, 50):
        vals = []
        for i in range(n // 2):
            vals.append(center - i - 1)
            vals.append(center + i + 1)
        if len(vals) == n and min(vals) >= 1 and max(vals) <= maximum and len(set(vals)) == n:
            value_candidates.append(sorted(vals))
    
    # Deduplicate
    seen = set()
    unique_candidates = []
    for vc in value_candidates:
        key = tuple(sorted(vc))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(vc)
    
    best_result = None
    best_raw = float('inf')
    
    # Sort candidates by heuristic: prefer sets with more sum/product collisions
    def collision_score(vals):
        sums = [vals[i] + vals[j] for i in range(n) for j in range(i + 1, n)]
        prods = [vals[i] * vals[j] for i in range(n) for j in range(i + 1, n)]
        return len(set(sums)) + len(set(prods))
    
    unique_candidates.sort(key=collision_score)
    
    for values in unique_candidates:
        if time.time() >= deadline:
            break
        
        edges, score = find_best_K(values, deadline)
        if edges is not None and score < best_raw:
            best_raw = score
            best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
        
        # Early termination if we found a very good solution
        if best_raw <= 5:
            break
    
    # If we haven't found anything good, use greedy with 1..14
    if best_result is None:
        values = list(range(1, n + 1))
        edges, score = greedy_select_edges(values, 20, deadline)
        if edges is None:
            edges = list(all_edges[:m])
            score = _raw_value(values, edges)
        best_result = {'values': list(values), 'edges': [list(e) for e in edges]}
    
    # Ensure types are correct
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
