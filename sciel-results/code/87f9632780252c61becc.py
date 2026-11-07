import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    # 90s limit, use 85s for safety
    deadline = start_time + 85.0
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    
    def get_score(values, edges):
        """Calculate the raw score: max(distinct_sums, distinct_prods)."""
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
        
    def can_fit_edges(values, target_k):
        """
        Backtracking search to see if we can select exactly 35 edges 
        such that distinct sums <= target_k and distinct products <= target_k.
        Uses Union-Find to group edges by (sum, product) signature to prune.
        """
        # Precompute sum and product for all possible edges
        edge_props = {}
        for u, v in all_edges:
            s = values[u] + values[v]
            p = values[u] * values[v]
            edge_props[(u, v)] = (s, p)
            
        # Sort edges by frequency of their signature to prioritize "dense" edges?
        # Or just iterate in a fixed order. 
        # Let's try sorting by (sum, product) to group similar ones early.
        sorted_edges = sorted(all_edges, key=lambda e: edge_props[e])
        
        # State for backtracking
        # We need to track:
        # 1. Number of edges selected
        # 2. Set of sums used
        # 3. Set of products used
        # 4. Union-Find structure for "signature groups" to quickly check if adding an edge 
        #    would exceed K distinct sums or products?
        # Actually, Union-Find on edges sharing the SAME sum or SAME product helps us 
        # count distinct sums/products if we add an edge.
        
        # A simpler constraint check:
        # If we maintain `used_sums` and `used_prods`, adding edge (u,v) is valid 
        # if len(used_sums) < K or (values[u]+values[v]) in used_sums
        # AND len(used_prods) < K or (values[u]*values[v]) in used_prods
        
        # However, backtracking on 91 edges choosing 35 is huge. 
        # We need strong pruning.
        
        # Heuristic: If the number of unique sums among ALL 91 edges is < K, 
        # then we can't even have K distinct sums, but that's fine, we just need <= K.
        # The bottleneck is usually hitting exactly K or close to it.
        
        # Let's use a recursive DFS with memoization on (index, current_edge_count, frozenset(used_sums), frozenset(used_prods))?
        # Frozensets are too big.
        
        # Alternative: Since K is small (e.g., 10-20), the state space of used_sums/products is limited 
        # if we only track the COUNT? No, we need to know WHICH ones to avoid duplicates.
        
        # Let's try a randomized greedy backtracking (beam search or limited depth backtracking).
        
        def dfs(idx, count, used_sums, used_prods, current_edges):
            if time.time() > deadline:
                return None
            
            if count == m:
                return current_edges
                
            if idx == len(sorted_edges):
                return None
            
            # Pruning: if even taking all remaining edges doesn't reach m, fail
            if count + (len(sorted_edges) - idx) < m:
                return None
                
            # Option 1: Skip current edge
            res = dfs(idx + 1, count, used_sums, used_prods, current_edges)
            if res is not None:
                return res
                
            # Option 2: Take current edge
            u, v = sorted_edges[idx]
            s, p = edge_props[(u, v)]
            
            # Check if adding this edge violates K constraint
            # We can add it if:
            # 1. s is already in used_sums OR len(used_sums) < K
            # 2. p is already in used_prods OR len(used_prods) < K
            
            if (s in used_sums or len(used_sums) < K) and (p in used_prods or len(used_prods) < K):
                new_sums = used_sums | {s}
                new_prods = used_prods | {p}
                res = dfs(idx + 1, count + 1, new_sums, new_prods, current_edges + [(u, v)])
                if res is not None:
                    return res
                    
            return None

        # Initial call
        # To speed up, we can randomize the order of edges slightly or use a heuristic order.
        # Let's try running this a few times with different random shuffles if it fails.
        # But first, let's just run it once with the sorted order.
        
        # Optimization: If K is very small, this might be too slow. 
        # We will only call this if K is small enough.
        
        res = dfs(0, 0, set(), set(), [])
        return res

    def optimize_values_for_k(values, K, max_iter=2000):
        """
        Try to find a set of values that allows a valid 35-edge subgraph with score <= K.
        Uses SA on values, and for each candidate, runs the edge CSP.
        """
        current_values = list(values)
        best_edges = None
        
        # Initial check
        edges = can_fit_edges(current_values, K)
        if edges:
            return current_values, edges
            
        # SA Loop
        T_start = 5.0
        T_end = 0.1
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
                
            T = T_start * (T_end / T_start) ** (i / max_iter)
            
            # Perturb values
            new_values = list(current_values)
            idx = random.randint(0, n - 1)
            
            # Try to make values more "collidable"
            # Strategy: Move value towards another value or a random small number
            # If we pick a random other index, we can swap or move towards it.
            if random.random() < 0.5:
                # Move towards another value
                target_idx = random.randint(0, n - 1)
                if target_idx == idx:
                    continue
                target_val = current_values[target_idx]
                step = random.randint(-5, 5)
                new_val = target_val + step
            else:
                # Small random step
                delta = random.randint(-10, 10)
                new_val = current_values[idx] + delta
                
            if 1 <= new_val <= 10000 and new_val not in current_values:
                new_values[idx] = new_val
            else:
                continue
                
            # Check if this new configuration allows a valid edge set for K
            edges = can_fit_edges(new_values, K)
            if edges:
                # Found a valid configuration!
                # We can return immediately or continue to see if we can improve?
                # The task is to minimize max(distinct_sums, distinct_prods).
                # If we found one for K, the score is <= K.
                # We can try smaller K in the outer loop.
                return new_values, edges
                
            # If not found, should we accept the move?
            # This is tricky. We don't have a score for "invalid" configs other than "fail".
            # We can accept with probability exp(-1/T) to explore, or reject.
            # Let's accept with a small probability to escape local minima in value space.
            if random.random() < 0.1:
                current_values = new_values
                
        return None, None

    # Main Search Strategy
    # 1. Try small K values (10, 11, 12, ...)
    # 2. For each K, run a limited search for values that support a K-solution.
    # 3. If found, return it.
    # 4. If not found within time, fallback to standard SA minimizing score.
    
    random.seed(42)
    
    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None
    
    # Initial diverse configs for fallback SA
    initial_configs = []
    v1 = random.sample(range(1, 51), n)
    initial_configs.append((v1, set(random.sample(all_edges, m))))
    
    v2 = random.sample(range(1, 101), n)
    initial_configs.append((v2, set(random.sample(all_edges, m))))
    
    # Try CSP for small K
    # Start with K=10, go up to 20?
    # Each CSP attempt is fast if it fails quickly, but can be slow if it explores deep.
    # We limit the time for each K.
    
    time_per_k = 5.0 # seconds
    start_k = 10
    end_k = 25
    
    for K in range(start_k, end_k + 1):
        if time.time() > deadline - 10: # Leave 10s for fallback
            break
            
        # Try a few random initial value sets for this K
        found = False
        for _ in range(5): # 5 restarts per K
            if time.time() > deadline - 10:
                break
                
            # Generate random values biased towards small numbers (more collisions)
            # Mix of small and medium
            vals = []
            for i in range(n):
                if i < 7:
                    vals.append(random.randint(1, 20))
                else:
                    vals.append(random.randint(1, 100))
                    
            # Ensure distinct
            while len(set(vals)) < n:
                idx = random.randint(0, n-1)
                old = vals[idx]
                vals[idx] = random.randint(1, 10000)
                while vals[idx] == old:
                    vals[idx] = random.randint(1, 10000)
            
            # Run value optimization for this K
            opt_vals, opt_edges = optimize_values_for_k(vals, K, max_iter=500)
            
            if opt_edges:
                # Calculate actual score
                score = get_score(opt_vals, opt_edges)
                if score < best_global_score:
                    best_global_score = score
                    best_global_values = opt_vals
                    best_global_edges = opt_edges
                found = True
                # If we found a solution for K, the score is <= K.
                # We can break out of K loop if K is very small? 
                # Or just continue to try smaller K? No, we are iterating K upwards.
                # So if we found one for K=10, score <= 10.
                # We can stop trying larger K if we want, but maybe we can find a better one?
                # No, larger K means weaker constraint, so if K=10 works, K=11 is easier.
                # We want the MINIMUM score. So if we found one for K=10, we should try K=9?
                # My loop is 10 to 25. So I should start from small K.
                # If I find one for K=10, I can stop searching for larger K?
                # Actually, I should search K=1, 2, ... 
                # But K=1 is impossible (14 distinct values -> at least 13 sums).
                # Min distinct sums for 14 distinct values:
                # If values are 1..14, sums range 3..27. Many distinct.
                # Let's assume min K is around 15-20.
                
                # Let's break the inner loop and continue to next K?
                # No, if we found a solution for K, we can return it immediately if we are confident?
                # The task is to minimize. If we found one for K=12, and we haven't tried K=11, 
                # we should try K=11.
                # So we should iterate K from low to high.
                # If we find a solution for K, we can stop the current K search.
                # But we should keep trying lower K? No, we are going UP.
                # So we should start at K=15, 16...
                # If we find one for K=15, we stop.
                # If we don't find one for K=15, we try K=16.
                
                break # Break inner restart loop
                
        if found:
            # We found a solution for this K. 
            # Since we iterate K upwards, this is the smallest K found so far.
            # We can break the outer loop.
            break
            
    # If no CSP solution found, fallback to standard SA
    if best_global_values is None:
        # Standard SA
        def joint_sa(initial_values, initial_edges, max_iter=5000, T_start=1.0, T_end=0.05):
            current_values = list(initial_values)
            current_edges = set(initial_edges)
            current_score = get_score(current_values, current_edges)
            best_score = current_score
            best_values = list(current_values)
            best_edges = set(current_edges)
            
            for i in range(max_iter):
                if time.time() > deadline - 2:
                    break
                T = T_start * (T_end / T_start) ** (i / max_iter)
                
                move_type = random.random()
                new_values = current_values
                new_edges = current_edges
                
                if move_type < 0.5:
                    e_out = random.choice(list(current_edges))
                    while True:
                        e_in = random.choice(all_edges)
                        if e_in not in current_edges:
                            break
                    new_edges = (current_edges - {e_out}) | {e_in}
                else:
                    idx = random.randint(0, n - 1)
                    if random.random() < 0.7:
                        idx2 = random.randint(0, n - 1)
                        if idx2 == idx:
                            continue
                        new_values = list(current_values)
                        new_values[idx], new_values[idx2] = new_values[idx2], new_values[idx]
                    else:
                        delta = random.randint(-5, 5)
                        new_val = current_values[idx] + delta
                        if 1 <= new_val <= 10000 and new_val not in current_values:
                            new_values = list(current_values)
                            new_values[idx] = new_val
                        else:
                            continue
                            
                new_score = get_score(new_values, new_edges)
                delta_score = new_score - current_score
                if delta_score <= 0 or random.random() < math.exp(-delta_score / max(T, 1e-9)):
                    current_values = new_values
                    current_edges = new_edges
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_values = list(current_values)
                        best_edges = set(current_edges)
                        
            return best_score, best_values, best_edges

        for vals, edges in initial_configs:
            if time.time() > deadline - 2:
                break
            score, best_v, best_e = joint_sa(vals, edges, max_iter=10000)
            if score < best_global_score:
                best_global_score = score
                best_global_values = best_v
                best_global_edges = best_e
                
    # Fallback if still nothing
    if best_global_values is None:
        best_global_values = list(range(1, 15))
        best_global_edges = set(all_edges[:35])
        
    # Ensure types
    final_values = [int(v) for v in best_global_values]
    final_edges = [[int(u), int(v)] for u, v in best_global_edges]
    
    return {
        'values': final_values,
        'edges': final_edges
    }
