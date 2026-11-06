"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    
    deadline = time.time() + 112  # 8s safety margin before 120s kill
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        if len(S) < 3:
            return True
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True
    
    def can_add(S_set, x):
        """Check if x can be added to S_set while maintaining sum-free property."""
        if x in S_set:
            return False
        if 2 * x in S_set:
            return False
        for a in S_set:
            if a + x in S_set:
                return False
            if a - x in S_set:
                return False
        return True
    
    def fast_greedy_add(S_list, candidates, S_set=None):
        """Greedily add elements from candidates to S while maintaining sum-free property."""
        if S_set is None:
            S_set = set(S_list)
        result = list(S_list)
        
        for c in candidates:
            if c in S_set:
                continue
            if can_add(S_set, c):
                result.append(c)
                S_set.add(c)
        
        return result, S_set
    
    def build_schur_graph(values):
        """Build adjacency for Schur graph: edge if a+b=c or 2a=b for some c in values."""
        n = len(values)
        val_set = set(values)
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                a, b = values[i], values[j]
                s = a + b
                if s in val_set:
                    k = values.index(s)
                    adj[i].add(k)
                    adj[k].add(i)
                    if j != k:
                        adj[j].add(k)
                        adj[k].add(j)
                if 2 * a == b:
                    adj[i].add(j)
                    adj[j].add(i)
                elif 2 * b == a:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return adj
    
    def spectral_partition(values, adj, rng):
        """Use spectral bisection to get initial partition."""
        n = len(values)
        if n <= 2:
            return [values[:], values[1:]] if n > 1 else [[], []]
        
        try:
            import numpy as np
        except ImportError:
            # Fallback: greedy coloring
            color = [-1] * n
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                can_be_0 = all(color[j] != 0 for j in adj[i])
                color[i] = 0 if can_be_0 else 1
            set1 = [values[i] for i in range(n) if color[i] == 0]
            set2 = [values[i] for i in range(n) if color[i] == 1]
            return [set1, set2]
        
        D = np.zeros((n, n))
        A = np.zeros((n, n))
        for i in range(n):
            D[i][i] = len(adj[i])
            for j in adj[i]:
                A[i][j] = 1.0
        
        L = D - A
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
        except:
            color = [-1] * n
            order = list(range(n))
            rng.shuffle(order)
            for i in order:
                can_be_0 = all(color[j] != 0 for j in adj[i])
                color[i] = 0 if can_be_0 else 1
            set1 = [values[i] for i in range(n) if color[i] == 0]
            set2 = [values[i] for i in range(n) if color[i] == 1]
            return [set1, set2]
        
        idx = np.argsort(eigenvalues)
        fiedler = eigenvectors[:, idx[1]]
        
        set1 = [values[i] for i in range(n) if fiedler[i] >= 0]
        set2 = [values[i] for i in range(n) if fiedler[i] < 0]
        
        return [set1, set2]
    
    def local_search_2swap(S_list, values, time_limit=5):
        """Improve S using 1-swap and 2-swap moves."""
        S_set = set(S_list)
        start_time = time.time()
        
        # Phase 1: Greedy additions
        candidates = [x for x in values if x not in S_set]
        S_list, S_set = fast_greedy_add(S_list, candidates, S_set)
        
        # Phase 2: 1-swap moves (remove x, add y)
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            S_list_copy = list(S_list)
            S_set_copy = set(S_set)
            
            for x in S_list_copy:
                if time.time() - start_time > time_limit:
                    break
                S_test = S_set_copy - {x}
                S_list_test = [y for y in S_list_copy if y != x]
                
                # Try adding excluded elements
                for y in values:
                    if y in S_test or y == x:
                        continue
                    if can_add(S_test, y):
                        # Try greedy fill after swap
                        S_new_list = S_list_test + [y]
                        S_new_set = S_test | {y}
                        remaining = [z for z in values if z not in S_new_set and z != x]
                        S_new_list, S_new_set = fast_greedy_add(S_new_list, remaining, S_new_set)
                        
                        if len(S_new_set) > len(S_set):
                            S_list = S_new_list
                            S_set = S_new_set
                            improved = True
                            S_list_copy = list(S_list)
                            S_set_copy = set(S_set)
                            break
                if improved:
                    break
        
        # Phase 3: 2-swap moves (remove x,y, add z,w)
        if time.time() - start_time < time_limit:
            S_list_copy = list(S_list)
            S_set_copy = set(S_set)
            excluded = [y for y in values if y not in S_set_copy]
            
            for i in range(len(S_list_copy)):
                if time.time() - start_time > time_limit:
                    break
                x = S_list_copy[i]
                for j in range(i+1, len(S_list_copy)):
                    if time.time() - start_time > time_limit:
                        break
                    y = S_list_copy[j]
                    
                    S_test = S_set_copy - {x, y}
                    S_list_test = [z for z in S_list_copy if z not in (x, y)]
                    
                    # Try adding pairs from excluded
                    for a in excluded:
                        if time.time() - start_time > time_limit:
                            break
                        if a in S_test:
                            continue
                        if not can_add(S_test, a):
                            continue
                        S_temp_set = S_test | {a}
                        S_temp_list = S_list_test + [a]
                        
                        for b in excluded:
                            if b == a or b in S_temp_set:
                                continue
                            if not can_add(S_temp_set, b):
                                continue
                            
                            S_new_list = S_temp_list + [b]
                            S_new_set = S_temp_set | {b}
                            
                            # Greedy fill
                            remaining = [z for z in values if z not in S_new_set and z not in (x, y)]
                            S_new_list, S_new_set = fast_greedy_add(S_new_list, remaining, S_new_set)
                            
                            if len(S_new_set) > len(S_set):
                                S_list = S_new_list
                                S_set = S_new_set
                                improved = True
                                S_list_copy = list(S_list)
                                S_set_copy = set(S_set)
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return S_list
    
    def solve_one(values, time_budget=30):
        """Find a large sum-free subset of values."""
        if time.time() > deadline:
            return []
        
        n = len(values)
        if n == 0:
            return []
        
        rng = Random(42)
        
        # Strategy 1: Start with odd numbers (always sum-free)
        odd_subset = [x for x in values if x % 2 != 0]
        best = odd_subset
        best_set = set(best)
        
        # Strategy 2: Greedy from multiple orderings
        orderings = [
            sorted(values, key=abs),
            sorted(values, key=abs, reverse=True),
            sorted(values),
            sorted(values, reverse=True),
        ]
        
        for ordering in orderings:
            if time.time() > deadline:
                break
            candidates = [x for x in ordering if x not in best_set]
            if not candidates:
                continue
            trial_list, trial_set = fast_greedy_add(list(best), candidates, set(best))
            if len(trial_list) > len(best):
                best = trial_list
                best_set = trial_set
        
        # Strategy 3: Spectral bisection
        if time.time() < deadline - 5:
            try:
                adj = build_schur_graph(values)
                partitions = spectral_partition(values, adj, rng)
                for part in partitions:
                    if len(part) > len(best):
                        part_set = set(part)
                        candidates = [x for x in values if x not in part_set]
                        trial_list, trial_set = fast_greedy_add(part, candidates, part_set)
                        if len(trial_list) > len(best):
                            best = trial_list
                            best_set = trial_set
            except Exception:
                pass
        
        # Local search with 2-swap
        remaining_time = deadline - time.time() - 2
        if remaining_time > 3:
            ls_time = min(remaining_time - 2, time_budget * 0.4)
            improved = local_search_2swap(best, values, time_limit=ls_time)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)
        
        # Random restarts
        remaining_time = deadline - time.time() - 3
        num_restarts = max(1, int(remaining_time / 4))
        for i in range(num_restarts):
            if time.time() > deadline - 3:
                break
            shuffled = list(values)
            rng.shuffle(shuffled)
            trial_list, trial_set = fast_greedy_add([], shuffled)
            if len(trial_list) > len(best):
                ls_time = min(2, deadline - time.time() - 3)
                if ls_time > 1:
                    improved = local_search_2swap(trial_list, values, time_limit=ls_time)
                    if len(improved) > len(best):
                        best = improved
                        best_set = set(best)
                elif len(trial_list) > len(best):
                    best = trial_list
                    best_set = trial_set
        
        return best
    
    # Process instances with proportional time allocation
    instances = [(60, 801), (120, 802), (240, 803)]
    total_size = sum(s for s, _ in instances)
    
    results = []
    for size, seed in instances:
        if time.time() > deadline:
            results.append([])
            continue
        values = instance(size, seed)
        
        # Allocate time proportional to size
        time_budget = max(8, (size / total_size) * 95)
        subset = solve_one(values, time_budget=time_budget)
        
        # Final verification and fallback
        if not is_sum_free(subset):
            subset = [x for x in values if x % 2 != 0]
        
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
