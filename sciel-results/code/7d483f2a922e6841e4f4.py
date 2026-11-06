"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    from random import Random
    
    deadline = time.time() + 115  # 5s safety margin before 120s kill
    
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
    
    def fast_greedy_add(S_list, candidates, S_set=None):
        """
        Greedily add elements from candidates to S while maintaining sum-free property.
        Uses conflict-tracking for O(n) per candidate check.
        """
        if S_set is None:
            S_set = set(S_list)
        result = list(S_list)
        s_list = list(S_set)
        
        # Build set of all pairwise sums and differences
        pairwise_sums = set()
        pairwise_diffs = set()
        for i, a in enumerate(s_list):
            for b in s_list[i:]:
                pairwise_sums.add(a + b)
                if a != b:
                    pairwise_diffs.add(a - b)
                    pairwise_diffs.add(b - a)
        
        for c in candidates:
            if c in S_set:
                continue
            # Check conflicts
            if 2 * c in S_set:
                continue
            if c in pairwise_sums:
                continue
            if c in pairwise_diffs:
                continue
            # c is valid, add it
            result.append(c)
            S_set.add(c)
            # Update pairwise sums/diffs with new element
            for a in s_list:
                pairwise_sums.add(a + c)
                pairwise_diffs.add(a - c)
                pairwise_diffs.add(c - a)
            s_list.append(c)
        
        return result, S_set
    
    def build_schur_graph(values):
        """
        Build adjacency list for the Schur graph on values.
        Edge (i,j) exists if values[i] + values[j] = values[k] for some k,
        or 2*values[i] = values[j].
        We want independent sets (sum-free subsets).
        """
        n = len(values)
        val_set = set(values)
        # Map value to index for quick lookup
        val_to_idx = {v: i for i, v in enumerate(values)}
        
        adj = [set() for _ in range(n)]
        
        for i in range(n):
            for j in range(i, n):
                a, b = values[i], values[j]
                s = a + b
                if s in val_set:
                    k = val_to_idx[s]
                    adj[i].add(k)
                    adj[k].add(i)
                    if j != k:
                        adj[j].add(k)
                        adj[k].add(j)
                # Check 2a = b or 2b = a
                if 2 * a == b:
                    adj[i].add(j)
                    adj[j].add(i)
                elif 2 * b == a:
                    adj[i].add(j)
                    adj[j].add(i)
        
        return adj
    
    def spectral_bisection_init(values, adj, rng):
        """
        Use spectral bisection (Fiedler vector) to find a good initial partition.
        Returns two candidate subsets (the two color classes).
        """
        n = len(values)
        if n <= 2:
            return [values[:], values[1:]] if n > 1 else [[], []]
        
        # Build Laplacian matrix L = D - A
        # D[i][i] = degree(i), A[i][j] = 1 if edge
        # For small n, use numpy for eigendecomposition
        try:
            import numpy as np
        except ImportError:
            # Fallback: use simple greedy coloring
            return greedy_coloring(values, adj, rng)
        
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
            return greedy_coloring(values, adj, rng)
        
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Sort eigenvalues and take the second one
        idx = np.argsort(eigenvalues)
        fiedler = eigenvectors[:, idx[1]]
        
        # Partition: elements with positive fiedler values go to set 1, negative to set 2
        set1 = [values[i] for i in range(n) if fiedler[i] >= 0]
        set2 = [values[i] for i in range(n) if fiedler[i] < 0]
        
        return [set1, set2]
    
    def greedy_coloring(values, adj, rng):
        """Fallback: greedy 2-coloring of the Schur graph."""
        n = len(values)
        color = [-1] * n  # -1 uncolored, 0/1 colors
        order = list(range(n))
        rng.shuffle(order)
        
        for i in order:
            # Try color 0
            can_be_0 = True
            for j in adj[i]:
                if color[j] == 0:
                    can_be_0 = False
                    break
            if can_be_0:
                color[i] = 0
            else:
                color[i] = 1
        
        set1 = [values[i] for i in range(n) if color[i] == 0]
        set2 = [values[i] for i in range(n) if color[i] == 1]
        return [set1, set2]
    
    def local_search_improved(S_list, values, time_limit=8):
        """
        Improve S by:
        1. Greedy additions (fast)
        2. Remove-one-and-readd with swap
        """
        S_set = set(S_list)
        start_time_ls = time.time()
        
        # Phase 1: Fast greedy additions
        cand_set = set(values) - S_set
        S_list, S_set = fast_greedy_add(S_list, list(cand_set), S_set)
        
        # Phase 2: Remove-one-and-readd with swap
        improved = True
        while improved and time.time() - start_time_ls < time_limit:
            improved = False
            S_list_copy = list(S_list)
            S_set_copy = set(S_set)
            
            for x in S_list_copy:
                if time.time() - start_time_ls > time_limit:
                    break
                # Try removing x and re-adding others
                S_test = S_set_copy - {x}
                S_list_test = [y for y in S_list_copy if y != x]
                
                # Try to add back elements that were previously excluded
                candidates = [y for y in values if y not in S_test and y != x]
                S_list_new, S_set_new = fast_greedy_add(S_list_test, candidates, S_test)
                
                if len(S_set_new) > len(S_set):
                    S_list = S_list_new
                    S_set = S_set_new
                    improved = True
                    S_list_copy = list(S_list)
                    S_set_copy = set(S_set)
                    break
            
            # Phase 2b: Try swap moves
            if not improved and time.time() - start_time_ls < time_limit:
                S_list_copy = list(S_list)
                S_set_copy = set(S_set)
                excluded = [y for y in values if y not in S_set_copy]
                
                for x in S_list_copy:
                    if time.time() - start_time_ls > time_limit:
                        break
                    S_test = S_set_copy - {x}
                    S_list_test = [y for y in S_list_copy if y != x]
                    
                    # Try adding excluded elements one at a time
                    for y in excluded:
                        if y in S_test:
                            continue
                        # Check if y can be added to S_test
                        if 2 * y in S_test:
                            continue
                        # Check pairwise conflicts
                        valid = True
                        for a in S_list_test:
                            if a + y in S_test or y - a in S_test:
                                valid = False
                                break
                        if valid:
                            # Swap: remove x, add y
                            S_list_new = S_list_test + [y]
                            S_set_new = S_test | {y}
                            # Try to add more after the swap
                            remaining = [z for z in values if z not in S_set_new and z != x]
                            S_list_new, S_set_new = fast_greedy_add(S_list_new, remaining, S_set_new)
                            if len(S_set_new) > len(S_set):
                                S_list = S_list_new
                                S_set = S_set_new
                                improved = True
                                S_list_copy = list(S_list)
                                S_set_copy = set(S_set)
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
        
        # Strategy 2: Try greedy from several orderings, starting from odd_subset
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
        
        # Strategy 3: Spectral bisection on Schur graph
        if time.time() < deadline - 5:
            try:
                adj = build_schur_graph(values)
                partitions = spectral_bisection_init(values, adj, rng)
                for part in partitions:
                    if len(part) > len(best):
                        # Refine this partition with greedy
                        part_set = set(part)
                        candidates = [x for x in values if x not in part_set]
                        trial_list, trial_set = fast_greedy_add(part, candidates, part_set)
                        if len(trial_list) > len(best):
                            best = trial_list
                            best_set = trial_set
            except Exception:
                pass
        
        # Local search on best
        remaining_time = deadline - time.time() - 2
        if remaining_time > 3:
            ls_time = min(remaining_time - 2, time_budget * 0.5)
            improved = local_search_improved(best, values, time_limit=ls_time)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)
        
        # Try random restarts from scratch
        remaining_time = deadline - time.time() - 3
        num_restarts = max(1, int(remaining_time / 5))
        for i in range(num_restarts):
            if time.time() > deadline - 3:
                break
            shuffled = list(values)
            rng.shuffle(shuffled)
            trial_list, trial_set = fast_greedy_add([], shuffled)
            if len(trial_list) > len(best):
                # Local search on this new best
                ls_time = min(3, deadline - time.time() - 3)
                if ls_time > 1:
                    improved = local_search_improved(trial_list, values, time_limit=ls_time)
                    if len(improved) > len(best):
                        best = improved
                        best_set = set(best)
                elif len(trial_list) > len(best):
                    best = trial_list
                    best_set = trial_set
        
        return best
    
    # Process instances in order of increasing size to allocate time proportionally
    instances = [(60, 801), (120, 802), (240, 803)]
    total_size = sum(s for s, _ in instances)
    
    results = []
    for size, seed in instances:
        if time.time() > deadline:
            results.append([])
            continue
        values = instance(size, seed)
        
        # Allocate time proportional to size, with minimum guarantee
        time_budget = max(10, (size / total_size) * 100)
        subset = solve_one(values, time_budget=time_budget)
        
        # Final verification and fallback
        if not is_sum_free(subset):
            subset = [x for x in values if x % 2 != 0]
        
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
