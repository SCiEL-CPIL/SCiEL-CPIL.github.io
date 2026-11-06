import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_set = []
    best_len = 0
    
    def is_valid_addition(x, selected, sums_set):
        """
        Check if x can be added to selected without creating duplicate triple sums.
        Returns the set of new sums if valid, else None.
        """
        n = len(selected)
        
        # We will check all new sums involving x.
        # 1. (x, x, x)
        # 2. (x, x, s) for s in selected
        # 3. (x, s1, s2) for s1 <= s2 in selected
        
        # Optimization: 
        # The number of new sums is 1 + n + n(n+1)/2.
        # We can check against sums_set (O(1) per check) and a temporary set for internal collisions.
        
        temp_new_sums = set()
        
        # 1. 3x
        v = 3 * x
        if v in sums_set or v in temp_new_sums:
            return None
        temp_new_sums.add(v)
        
        # 2. 2x + s
        for s in selected:
            v = 2 * x + s
            if v in sums_set or v in temp_new_sums:
                return None
            temp_new_sums.add(v)
            
        # 3. x + s1 + s2
        for i in range(n):
            s1 = selected[i]
            for j in range(i, n):
                s2 = selected[j]
                v = x + s1 + s2
                if v in sums_set or v in temp_new_sums:
                    return None
                temp_new_sums.add(v)
                
        return temp_new_sums

    def run_greedy(candidates):
        """
        Greedily add elements from the provided candidate list.
        """
        selected = []
        sums_set = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
            new_sums = is_valid_addition(x, selected, sums_set)
            if new_sums is not None:
                selected.append(x)
                sums_set.update(new_sums)
                
        return selected

    # 1. Run standard linear greedy
    # This is a good baseline.
    res = run_greedy(range(1, 4097))
    if len(res) > best_len:
        best_set = res
        best_len = len(res)
        
    # 2. Run randomized greedy multiple times
    # Randomizing the order can help escape local optima where early small numbers block later larger ones.
    
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 5.0: # Leave 5s buffer
            break
            
        random.seed(i + 100)
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        
        res = run_greedy(candidates)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            
    # 3. Local Search / Swap Improvement
    # The greedy algorithm produces a maximal set for the given order, but not necessarily maximum.
    # We can try to improve the best set found by:
    # a) Removing an element and trying to add two or more new ones.
    # b) Simply trying to add any remaining number that fits (though greedy should have done this for its specific order, 
    #    a different order might have allowed a different set of numbers to be present, potentially leaving "holes" 
    #    that other numbers can fill if we restart the filling process on the complement? No, that's complex).
    
    # A simpler local search: 
    # Try to swap an element $s_i$ in the best set with a candidate $x$ not in the set.
    # If we remove $s_i$, we free up some sum constraints. Can we add $x$? 
    # Actually, a better swap is: remove $s_i$, then greedily try to add new elements from the remaining candidates 
    # until we can't add any more. If the final size is larger, we keep it.
    
    # This is expensive. Let's see if we have time.
    # If best_len is ~30, checking a swap involves:
    # 1. Removing s_i: O(1) logic, but we need to recompute sums_set? 
    #    Removing from a set of sums is hard because we don't know which sums involved s_i.
    #    So we must recompute sums_set from scratch for the new set.
    #    Recomputing sums for a set of size K is O(K^3) if done naively, or O(K^2 * K) if optimized?
    #    Actually, generating all triple sums for K elements is O(K^3).
    #    If K=30, K^3 = 27000. This is fast.
    #    So for each swap candidate:
    #    - Create new_set = best_set without s_i
    #    - Compute all sums for new_set (O(K^3))
    #    - Greedily add from remaining candidates (O(N * K^2))
    #    - If len(new_final_set) > best_len, update.
    
    # Let's try a limited number of swaps on the best set found.
    
    if best_len > 0:
        best_set_sorted = sorted(best_set)
        # Try removing each element in the best set
        for idx, s_i in enumerate(best_set_sorted):
            if time.time() > deadline - 2.0:
                break
            
            # Create a copy of the set without s_i
            trial_set = [s for j, s in enumerate(best_set_sorted) if j != idx]
            
            # Recompute sums for trial_set
            trial_sums = set()
            n_trial = len(trial_set)
            for a in range(n_trial):
                for b in range(a, n_trial):
                    for c in range(b, n_trial):
                        trial_sums.add(trial_set[a] + trial_set[b] + trial_set[c])
            
            # Now try to add more elements from the full range [1, 4096]
            # We should prioritize elements that were NOT in the original best set?
            # Or just any element?
            # Let's create a candidate list of all numbers not in trial_set.
            # To make it efficient, we can shuffle or sort them.
            # Since we are trying to improve, maybe sort by value? Or random?
            # Let's use a random order to explore different possibilities.
            random.seed(idx + 200)
            candidates = [x for x in range(1, 4097) if x not in trial_set]
            random.shuffle(candidates)
            
            # Greedy add
            for x in candidates:
                if time.time() > deadline - 2.0:
                    break
                new_sums = is_valid_addition(x, trial_set, trial_sums)
                if new_sums is not None:
                    trial_set.append(x)
                    trial_sums.update(new_sums)
                    
            if len(trial_set) > best_len:
                best_set = trial_set
                best_len = len(trial_set)
                
    return best_set
