"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
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
    
    def build_conflict_sets(S_list):
        """Build pairwise sums and differences for conflict checking."""
        pairwise_sums = set()
        pairwise_diffs = set()
        for i, a in enumerate(S_list):
            for b in S_list[i:]:
                pairwise_sums.add(a + b)
                if a != b:
                    pairwise_diffs.add(a - b)
                    pairwise_diffs.add(b - a)
        return pairwise_sums, pairwise_diffs
    
    def can_add(c, S_set, pairwise_sums, pairwise_diffs):
        """Check if c can be added to S while maintaining sum-free property."""
        if c in S_set:
            return False
        if 2 * c in S_set:
            return False
        if c in pairwise_sums:
            return False
        if c in pairwise_diffs:
            return False
        return True
    
    def fast_greedy_add(S_list, candidates, S_set=None):
        """
        Greedily add elements from candidates to S while maintaining sum-free property.
        Returns (new_list, new_set).
        """
        if S_set is None:
            S_set = set(S_list)
        result = list(S_list)
        s_list = list(S_set)
        
        pairwise_sums, pairwise_diffs = build_conflict_sets(s_list)
        
        for c in candidates:
            if can_add(c, S_set, pairwise_sums, pairwise_diffs):
                result.append(c)
                S_set.add(c)
                for a in s_list:
                    pairwise_sums.add(a + c)
                    pairwise_diffs.add(a - c)
                    pairwise_diffs.add(c - a)
                s_list.append(c)
        
        return result, S_set
    
    def local_search_bitmask(S_list, values, time_limit=8):
        """
        Improve S using bitmask-based conflict tracking for speed.
        Uses 1-flip (remove/add) and 2-flip (swap) moves.
        """
        S_set = set(S_list)
        values_set = set(values)
        start_time_ls = time.time()
        
        # Phase 1: Fast greedy additions
        cand_set = values_set - S_set
        S_list, S_set = fast_greedy_add(S_list, list(cand_set), S_set)
        
        # Build index mapping
        val_to_idx = {v: i for i, v in enumerate(values)}
        n_vals = len(values)
        
        # Represent S as a bitmask
        S_mask = 0
        for v in S_set:
            if v in val_to_idx:
                S_mask |= (1 << val_to_idx[v])
        
        best_size = len(S_set)
        no_improve_count = 0
        
        while time.time() - start_time_ls < time_limit:
            # Check if any element can be added
            added_any = False
            for v in values:
                if v not in S_set and can_add(v, S_set, *build_conflict_sets(list(S_set))):
                    S_list.append(v)
                    S_set.add(v)
                    S_mask |= (1 << val_to_idx[v])
                    added_any = True
                    best_size = len(S_set)
                    no_improve_count = 0
                    break
            
            if added_any:
                continue
            
            # Try 1-flip moves: remove x, try to add others
            S_list_copy = list(S_list)
            S_set_copy = set(S_set)
            improved_1flip = False
            
            for x in S_list_copy:
                if time.time() - start_time_ls > time_limit:
                    break
                
                S_test = S_set_copy - {x}
                S_list_test = [y for y in S_list_copy if y != x]
                ps, pd = build_conflict_sets(S_list_test)
                
                # Try to add as many as possible
                new_list = list(S_list_test)
                new_set = set(S_test)
                for y in values:
                    if y not in new_set and y != x and can_add(y, new_set, ps, pd):
                        new_list.append(y)
                        new_set.add(y)
                        for a in S_list_test:
                            ps.add(a + y)
                            pd.add(a - y)
                            pd.add(y - a)
                        S_list_test = new_list
                
                if len(new_set) > best_size:
                    S_list = new_list
                    S_set = new_set
                    S_mask = 0
                    for v in S_set:
                        if v in val_to_idx:
                            S_mask |= (1 << val_to_idx[v])
                    best_size = len(S_set)
                    no_improve_count = 0
                    improved_1flip = True
                    break
            
            if improved_1flip:
                continue
            
            # Try 2-flip (swap) moves: remove x, add y
            improved_2flip = False
            S_list_copy = list(S_list)
            S_set_copy = set(S_set)
            excluded = [y for y in values if y not in S_set_copy]
            
            for x in S_list_copy:
                if time.time() - start_time_ls > time_limit:
                    break
                
                S_test = S_set_copy - {x}
                S_list_test = [y for y in S_list_copy if y != x]
                ps, pd = build_conflict_sets(S_list_test)
                
                for y in excluded:
                    if y in S_test:
                        continue
                    if can_add(y, S_test, ps, pd):
                        # Swap: remove x, add y
                        new_list = S_list_test + [y]
                        new_set = S_test | {y}
                        # Try to add more after the swap
                        ps2, pd2 = build_conflict_sets(new_list)
                        for z in values:
                            if z not in new_set and z != x and can_add(z, new_set, ps2, pd2):
                                new_list.append(z)
                                new_set.add(z)
                                for a in S_list_test + [y]:
                                    ps2.add(a + z)
                                    pd2.add(a - z)
                                    pd2.add(z - a)
                        
                        if len(new_set) > best_size:
                            S_list = new_list
                            S_set = new_set
                            S_mask = 0
                            for v in S_set:
                                if v in val_to_idx:
                                    S_mask |= (1 << val_to_idx[v])
                            best_size = len(S_set)
                            no_improve_count = 0
                            improved_2flip = True
                            break
                if improved_2flip:
                    break
            
            if not improved_1flip and not improved_2flip:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break
        
        return S_list
    
    def polish_pass(S_list, values, time_limit=5):
        """Final high-precision polish: try all single-element swaps and additions."""
        S_set = set(S_list)
        values_set = set(values)
        start_time_polish = time.time()
        best = list(S_list)
        best_set = set(S_set)
        best_size = len(S_set)
        
        # Try adding any single element
        for v in values:
            if time.time() - start_time_polish > time_limit:
                break
            if v not in S_set:
                ps, pd = build_conflict_sets(list(S_set))
                if can_add(v, S_set, ps, pd):
                    trial = S_list + [v]
                    trial_set = S_set | {v}
                    if len(trial_set) > best_size:
                        best = trial
                        best_set = trial_set
                        best_size = len(trial_set)
                        S_set = trial_set
                        S_list = trial
        
        # Try single-element swaps
        excluded = [y for y in values if y not in best_set]
        for x in best:
            if time.time() - start_time_polish > time_limit:
                break
            S_test = best_set - {x}
            S_list_test = [y for y in best if y != x]
            ps, pd = build_conflict_sets(S_list_test)
            
            for y in excluded:
                if time.time() - start_time_polish > time_limit:
                    break
                if y in S_test:
                    continue
                if can_add(y, S_test, ps, pd):
                    trial = S_list_test + [y]
                    trial_set = S_test | {y}
                    if len(trial_set) > best_size:
                        best = trial
                        best_set = trial_set
                        best_size = len(trial_set)
        
        return best
    
    def solve_one(values, time_budget=30):
        """Find a large sum-free subset of values."""
        if time.time() > deadline:
            return []
        
        n = len(values)
        if n == 0:
            return []
        
        # Start with odd numbers (always sum-free)
        odd_subset = [x for x in values if x % 2 != 0]
        best = odd_subset
        best_set = set(best)
        
        # Try greedy from several orderings, starting from odd_subset
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
        
        # Local search on best
        remaining_time = deadline - time.time() - 3
        if remaining_time > 3:
            ls_time = min(remaining_time - 3, time_budget * 0.4)
            improved = local_search_bitmask(best, values, time_limit=ls_time)
            if len(improved) > len(best):
                best = improved
                best_set = set(best)
        
        # Polish pass
        remaining_time = deadline - time.time() - 2
        if remaining_time > 2:
            polish_time = min(remaining_time - 2, time_budget * 0.2)
            if polish_time > 1:
                polished = polish_pass(best, values, time_limit=polish_time)
                if len(polished) > len(best):
                    best = polished
                    best_set = set(best)
        
        # Try random restarts from scratch
        rng = random.Random(42)
        remaining_time = deadline - time.time() - 4
        if remaining_time > 5:
            num_restarts = max(1, int(remaining_time / 8))
            for i in range(num_restarts):
                if time.time() > deadline - 4:
                    break
                shuffled = list(values)
                rng.shuffle(shuffled)
                trial_list, trial_set = fast_greedy_add([], shuffled)
                if len(trial_list) > len(best):
                    # Local search on this new best
                    ls_time = min(4, deadline - time.time() - 4)
                    if ls_time > 1:
                        improved = local_search_bitmask(trial_list, values, time_limit=ls_time)
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
        time_budget = max(10, (size / total_size) * 130)
        subset = solve_one(values, time_budget=time_budget)
        
        # Final verification and fallback
        if not is_sum_free(subset):
            subset = [x for x in values if x % 2 != 0]
        
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
