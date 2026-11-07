import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = 14
    m = 35

    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))

    def _optimize_edges_for_values(values, time_budget):
        """Find 35 edges minimizing max(distinct_sums, distinct_prods) for given values."""
        all_edges = list(combinations(range(n), 2))
        num_edges = len(all_edges)
        
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        # Precompute unique sums and products
        all_sums = sorted(set(edge_sums))
        all_prods = sorted(set(edge_prods))
        sum_to_idx = {s: i for i, s in enumerate(all_sums)}
        prod_to_idx = {p: i for i, p in enumerate(all_prods)}
        
        num_sums = len(all_sums)
        num_prods = len(all_prods)
        
        # Binary search on K: find minimum K such that we can select 35 edges
        # with at most K distinct sums AND at most K distinct products
        
        # For a given K, check if there exist 35 edges using <= K sums and <= K prods
        # This is a combinatorial problem. We'll use a greedy + local search approach
        # with bitmask tracking for speed.
        
        def can_achieve(K, values, edge_sums, edge_prods, sum_to_idx, prod_to_idx, num_sums, num_prods, time_budget):
            """Check if 35 edges can be selected with <= K distinct sums and <= K distinct prods."""
            # Use bitmask for sums and products if they fit
            # num_sums and num_prods could be up to ~91, so bitmask of 91 bits fits in Python int
            
            # Greedy initialization: pick edges that reuse existing sums/prods
            random.seed(42)
            best_edges = None
            best_score = (K + 1, K + 1)  # (distinct_sums, distinct_prods)
            
            for attempt in range(100):
                if time.time() > time_budget:
                    break
                
                # Random greedy initialization
                selected = set()
                sum_mask = 0
                prod_mask = 0
                sum_count = 0
                prod_count = 0
                
                # Shuffle edges
                edge_order = list(range(num_edges))
                random.shuffle(edge_order)
                
                # First pass: try to build a set of 35 edges
                # Strategy: prefer edges whose sum/product are already in the mask
                for idx in edge_order:
                    if len(selected) >= m:
                        break
                    s_idx = sum_to_idx[edge_sums[idx]]
                    p_idx = prod_to_idx[edge_prods[idx]]
                    
                    # Check if adding this edge keeps us within K
                    new_sum_count = sum_count + (1 if not (sum_mask >> s_idx) & 1 else 0)
                    new_prod_count = prod_count + (1 if not (prod_mask >> p_idx) & 1 else 0)
                    
                    if new_sum_count <= K and new_prod_count <= K:
                        selected.add(idx)
                        sum_mask |= (1 << s_idx)
                        prod_mask |= (1 << p_idx)
                        sum_count = new_sum_count
                        prod_count = new_prod_count
                
                if len(selected) < m:
                    continue
                
                # Local search: try swaps to reduce distinct counts
                current_score = (sum_count, prod_count)
                improved = True
                while improved and time.time() < time_budget:
                    improved = False
                    non_selected = [i for i in range(num_edges) if i not in selected]
                    if not non_selected:
                        break
                    
                    best_swap = None
                    best_swap_score = current_score
                    
                    for i in selected:
                        s_out_idx = sum_to_idx[edge_sums[i]]
                        p_out_idx = prod_to_idx[edge_prods[i]]
                        
                        # Tentatively remove i
                        # Count how many times each sum/prod is used
                        # For efficiency, track counts
                        sum_counts = {}
                        prod_counts = {}
                        for idx in selected:
                            s = edge_sums[idx]
                            p = edge_prods[idx]
                            sum_counts[s] = sum_counts.get(s, 0) + 1
                            prod_counts[p] = prod_counts.get(p, 0) + 1
                        
                        sum_counts[edge_sums[i]] -= 1
                        prod_counts[edge_prods[i]] -= 1
                        if sum_counts[edge_sums[i]] == 0:
                            del sum_counts[edge_sums[i]]
                        if prod_counts[edge_prods[i]] == 0:
                            del prod_counts[edge_prods[i]]
                        
                        active_sums = len(sum_counts)
                        active_prods = len(prod_counts)
                        
                        for j in non_selected:
                            s_in = edge_sums[j]
                            p_in = edge_prods[j]
                            
                            new_active_sums = active_sums + (1 if s_in not in sum_counts else 0)
                            new_active_prods = active_prods + (1 if p_in not in prod_counts else 0)
                            
                            if new_active_sums > K or new_active_prods > K:
                                continue
                            
                            score = (new_active_sums, new_active_prods)
                            if score < best_swap_score:
                                best_swap_score = score
                                best_swap = (i, j)
                    
                    if best_swap:
                        i, j = best_swap
                        selected.remove(i)
                        selected.add(j)
                        current_score = best_swap_score
                        improved = True
                        if current_score[0] == 1 and current_score[1] == 1:
                            break
                
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(selected)
                
                if best_score[0] <= K and best_score[1] <= K:
                    return True, best_edges
            
            return False, best_edges

        # Start with a reasonable K and decrease
        # Initial estimate: greedy might give K around 15-25
        best_K = 30
        best_edge_set = None
        
        # Try K from high to low
        for K in range(30, 5, -1):
            if time.time() > time_budget:
                break
            success, edges = can_achieve(K, values, edge_sums, edge_prods, sum_to_idx, prod_to_idx, num_sums, num_prods, time_budget)
            if success:
                best_K = K
                best_edge_set = edges
                # Try to go lower
                continue
            else:
                break
        
        if best_edge_set is None:
            # Fallback: greedy
            selected = []
            sum_counts = {}
            prod_counts = {}
            for _ in range(m):
                best_idx = -1
                best_score = (float('inf'), float('inf'))
                for idx in range(num_edges):
                    if idx in selected:
                        continue
                    s = edge_sums[idx]
                    p = edge_prods[idx]
                    new_active_sums = len(sum_counts) + (1 if sum_counts.get(s, 0) == 0 else 0)
                    new_active_prods = len(prod_counts) + (1 if prod_counts.get(p, 0) == 0 else 0)
                    score = (max(new_active_sums, new_active_prods), new_active_sums + new_active_prods)
                    if score < best_score:
                        best_score = score
                        best_idx = idx
                if best_idx != -1:
                    selected.append(best_idx)
                    s = edge_sums[best_idx]
                    p = edge_prods[best_idx]
                    sum_counts[s] = sum_counts.get(s, 0) + 1
                    prod_counts[p] = prod_counts.get(p, 0) + 1
            best_edge_set = set(selected)
        
        return [all_edges[i] for i in best_edge_set]

    def _joint_optimize(time_budget):
        """Jointly optimize values and edges."""
        best_result = None
        best_score = float('inf')
        
        # Initial value sets to try
        value_sets = [
            list(range(1, n + 1)),
            list(range(2, n + 2)),
            [i * 7 + 1 for i in range(n)],
            [i * 11 + 1 for i in range(n)],
            [i * 13 + 1 for i in range(n)],
            [2**i + 1 for i in range(n)],
            [i * i + 1 for i in range(n)],
        ]
        
        # Add some random value sets
        random.seed(123)
        for _ in range(20):
            vals = random.sample(range(1, 10001), n)
            vals.sort()
            value_sets.append(vals)
        
        for values in value_sets:
            if time.time() > time_budget:
                break
            
            # Check if values are distinct positive integers <= 10000
            if len(set(values)) != n or any(v < 1 or v > 10000 for v in values):
                continue
            
            edges = _optimize_edges_for_values(values, time_budget)
            if edges is None or len(edges) != m:
                continue
            
            score = _raw_value(values, edges)
            if score < best_score:
                best_score = score
                best_result = (list(values), list(edges))
            
            # Local search on values
            improved = True
            while improved and time.time() < time_budget:
                improved = False
                for i in range(n):
                    for delta in [-3, -2, -1, 1, 2, 3, -7, -5, 5, 7]:
                        new_values = list(values)
                        new_values[i] += delta
                        if new_values[i] < 1 or new_values[i] > 10000:
                            continue
                        if len(set(new_values)) != n:
                            continue
                        
                        new_edges = _optimize_edges_for_values(new_values, time_budget)
                        if new_edges is None or len(new_edges) != m:
                            continue
                        
                        new_score = _raw_value(new_values, new_edges)
                        if new_score < score:
                            values = new_values
                            edges = new_edges
                            score = new_score
                            best_score = score
                            best_result = (list(values), list(edges))
                            improved = True
                            break
                    if improved:
                        break
        
        if best_result is None:
            values = list(range(1, n + 1))
            edges = _optimize_edges_for_values(values, time_budget)
            best_result = (list(values), list(edges))
        
        return best_result

    # Run joint optimization
    values, edges = _joint_optimize(deadline)
    
    # Final validation
    if not values or len(values) != n:
        values = list(range(1, n + 1))
        edges = _optimize_edges_for_values(values, deadline)
    
    # Ensure types are correct
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {
        'values': values,
        'edges': edges
    }
