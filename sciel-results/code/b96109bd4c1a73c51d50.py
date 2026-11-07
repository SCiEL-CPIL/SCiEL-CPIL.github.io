"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 14
    m = 35
    
    def eval_edges(values, edges):
        """Compute max(distinct sums, distinct products) for given values and edges."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def greedy_edge_select(values, target_max=None):
        """Greedily select m edges to minimize max(distinct_sums, distinct_prods)."""
        all_edges = list(combinations(range(n), 2))
        
        # Precompute sums and products
        edge_data = []
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_data.append((u, v, s, p))
        
        # Greedy: at each step, pick the edge that minimizes the increase
        selected = []
        sel_set = set()
        sum_counts = {}
        prod_counts = {}
        
        for _ in range(m):
            best_edge = None
            best_score = float('inf')
            for u, v, s, p in edge_data:
                e = (u, v)
                if e in sel_set:
                    continue
                new_sum_count = len(sum_counts) + (1 if s not in sum_counts else 0)
                new_prod_count = len(prod_counts) + (1 if p not in prod_counts else 0)
                score = max(new_sum_count, new_prod_count)
                if score < best_score:
                    best_score = score
                    best_edge = (u, v, s, p)
            
            if best_edge is None:
                break
            
            u, v, s, p = best_edge
            selected.append((u, v))
            sel_set.add((u, v))
            sum_counts[s] = sum_counts.get(s, 0) + 1
            prod_counts[p] = prod_counts.get(p, 0) + 1
        
        return selected, eval_edges(values, selected)
    
    def local_search_edges(values, edges, max_steps=None):
        """Improve edge selection via 1-swap local search."""
        all_edges = list(combinations(range(n), 2))
        current_edges = list(edges)
        current_score = eval_edges(values, current_edges)
        in_set = set(current_edges)
        
        best_edges = list(current_edges)
        best_score = current_score
        
        no_improve = 0
        max_no_improve = 5000
        steps = 0
        if max_steps is None:
            max_steps = 100000
        
        while time.time() < deadline - 3 and no_improve < max_no_improve and steps < max_steps:
            steps += 1
            idx_remove = random.randint(0, m - 1)
            e_remove = current_edges[idx_remove]
            
            # Try a random non-edge
            while True:
                e_add = random.choice(all_edges)
                if e_add not in in_set:
                    break
            
            new_edges = list(current_edges)
            new_edges[idx_remove] = e_add
            new_score = eval_edges(values, new_edges)
            
            if new_score <= current_score:
                current_edges = new_edges
                in_set.discard(e_remove)
                in_set.add(e_add)
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(current_edges)
            else:
                no_improve += 1
        
        return best_edges, best_score
    
    def simulated_annealing_edges(values, edges, max_steps=None):
        """Simulated annealing on edge set with 1-swap moves."""
        all_edges = list(combinations(range(n), 2))
        current_edges = list(edges)
        current_score = eval_edges(values, current_edges)
        in_set = set(current_edges)
        
        best_edges = list(current_edges)
        best_score = current_score
        
        if max_steps is None:
            max_steps = 50000
        
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        steps = 0
        
        while time.time() < deadline - 3 and steps < max_steps and T > T_min:
            steps += 1
            idx_remove = random.randint(0, m - 1)
            e_remove = current_edges[idx_remove]
            
            while True:
                e_add = random.choice(all_edges)
                if e_add not in in_set:
                    break
            
            new_edges = list(current_edges)
            new_edges[idx_remove] = e_add
            new_score = eval_edges(values, new_edges)
            
            delta = new_score - current_score
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_edges = new_edges
                in_set.discard(e_remove)
                in_set.add(e_add)
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = list(current_edges)
            
            T *= cooling
        
        return best_edges, best_score
    
    def optimize_values_with_edges(values, edge_init=None):
        """Joint optimization: perturb values and re-optimize edges."""
        if edge_init is None:
            edge_init, _ = greedy_edge_select(values)
        
        best_edges, best_score = edge_init, eval_edges(values, edge_init)
        
        # Local search on edges first
        ls_edges, ls_score = local_search_edges(values, edge_init, max_steps=5000)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
        
        # Simulated annealing on edges
        sa_edges, sa_score = simulated_annealing_edges(values, best_edges, max_steps=20000)
        if sa_score < best_score:
            best_score = sa_score
            best_edges = sa_edges
        
        return best_edges, best_score
    
    def generate_value_candidates():
        """Generate diverse value candidate sets."""
        candidates = []
        
        # 1. Small consecutive ranges
        for start in range(1, 20):
            candidates.append(list(range(start, start + n)))
        
        # 2. Multiples of small integers
        for base in [2, 3, 4, 5, 6, 8, 10]:
            vals = list(range(base, base * n, base))
            if len(vals) == n:
                candidates.append(vals)
        
        # 3. Geometric-like progressions (integer)
        candidates.append([1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18])
        candidates.append([1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16])
        candidates.append([2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 25, 27])
        
        # 4. Values with many common factors
        candidates.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
        candidates.append([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
        candidates.append([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56])
        
        # 5. Random clustered sets
        random.seed(42)
        for _ in range(10):
            center = random.randint(1, 50)
            vals = set()
            while len(vals) < n:
                v = random.randint(max(1, center - 10), center + 10)
                if v <= 10000:
                    vals.add(v)
            candidates.append(sorted(vals))
        
        # 6. Random sets with structure
        for _ in range(10):
            base = random.randint(1, 20)
            vals = []
            for i in range(n):
                v = base * (i + 1) + random.randint(-2, 2)
                if v > 0 and v <= 10000 and v not in vals:
                    vals.append(v)
                else:
                    vals.append(base * (i + 1))
            # Ensure distinct
            seen = set()
            unique_vals = []
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    unique_vals.append(v)
            while len(unique_vals) < n:
                v = unique_vals[-1] + 1
                if v <= 10000:
                    unique_vals.append(v)
            candidates.append(unique_vals[:n])
        
        # Deduplicate
        unique_candidates = []
        seen = set()
        for vals in candidates:
            key = tuple(sorted(vals))
            if key not in seen and len(set(vals)) == n:
                if all(v > 0 and v <= 10000 for v in vals):
                    seen.add(key)
                    unique_candidates.append(vals)
        
        return unique_candidates
    
    def hill_climb_values(values, edges, max_steps=10000):
        """Hill climb on values while re-optimizing edges."""
        current_values = list(values)
        current_edges = list(edges)
        current_score = eval_edges(current_values, current_edges)
        
        best_values = list(current_values)
        best_edges = list(current_edges)
        best_score = current_score
        
        steps = 0
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline - 3 and steps < max_steps and no_improve < max_no_improve:
            steps += 1
            # Pick a random vertex to perturb
            idx = random.randint(0, n - 1)
            old_val = current_values[idx]
            
            # Generate a new value
            if random.random() < 0.5:
                # Small perturbation
                new_val = old_val + random.randint(-5, 5)
            else:
                # Larger perturbation
                new_val = random.randint(1, 10000)
            
            if new_val <= 0 or new_val > 10000:
                no_improve += 1
                continue
            
            # Check distinctness
            if any(current_values[j] == new_val for j in range(n) if j != idx):
                no_improve += 1
                continue
            
            current_values[idx] = new_val
            
            # Re-optimize edges quickly
            new_edges, new_score = greedy_edge_select(current_values)
            
            if new_score <= current_score:
                current_edges = new_edges
                current_score = new_score
                no_improve = 0
                if current_score < best_score:
                    best_score = current_score
                    best_values = list(current_values)
                    best_edges = list(current_edges)
            else:
                current_values[idx] = old_val
                no_improve += 1
        
        return best_values, best_edges, best_score
    
    # Main optimization loop
    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    candidates = generate_value_candidates()
    
    # Sort candidates by initial greedy score to prioritize promising ones
    scored_candidates = []
    for vals in candidates:
        if time.time() > deadline - 10:
            break
        _, score = greedy_edge_select(vals)
        scored_candidates.append((score, vals))
    
    scored_candidates.sort(key=lambda x: x[0])
    
    for score_init, vals in scored_candidates:
        if time.time() > deadline - 10:
            break
        
        # Full edge optimization
        edges, score = optimize_values_with_edges(vals)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = list(vals)
            best_overall_edges = list(edges)
        
        # Try hill climbing on values
        if time.time() < deadline - 5:
            hc_vals, hc_edges, hc_score = hill_climb_values(vals, edges, max_steps=5000)
            if hc_score < best_overall_score:
                best_overall_score = hc_score
                best_overall_values = list(hc_vals)
                best_overall_edges = list(hc_edges)
                
                # Further edge optimization on improved values
                if time.time() < deadline - 3:
                    final_edges, final_score = optimize_values_with_edges(hc_vals, hc_edges)
                    if final_score < best_overall_score:
                        best_overall_score = final_score
                        best_overall_values = list(hc_vals)
                        best_overall_edges = list(final_edges)
    
    # If we still have time, try random value sets
    if time.time() < deadline - 5:
        for _ in range(30):
            if time.time() > deadline - 3:
                break
            
            values = sorted(random.sample(range(1, 10001), n))
            edges, score = greedy_edge_select(values)
            
            if score < best_overall_score:
                best_overall_score = score
                best_overall_values = list(values)
                best_overall_edges = list(edges)
                
                if time.time() < deadline - 3:
                    ls_edges, ls_score = local_search_edges(values, edges, max_steps=3000)
                    if ls_score < best_overall_score:
                        best_overall_score = ls_score
                        best_overall_values = list(values)
                        best_overall_edges = ls_edges
    
    # Final validation and type conversion
    if best_overall_edges is None:
        values = list(range(1, 15))
        edges, _ = greedy_edge_select(values)
        best_overall_values = values
        best_overall_edges = edges
    
    return {
        'values': [int(v) for v in best_overall_values],
        'edges': [[int(u), int(v)] for u, v in best_overall_edges]
    }
# EVOLVE-BLOCK-END
