import time
import random

def construct():
    n = 256
    deadline = time.time() + 85.0  # Safety margin
    
    # Base tail: [129, 256]
    # For any a < b <= c in [129, 256], b+c >= 129+130 = 259.
    # Max a = 128. 259 is not divisible by any a in [1, 128] that is > 259/2?
    # Actually, the proof is: if a, b, c > n/2, then b+c > n.
    # If a | b+c, then b+c = k*a.
    # Since b,c <= n, b+c <= 2n. So k can be 1 or 2.
    # If k=1, b+c=a. But b>a, c>a, so b+c > 2a > a. Contradiction.
    # If k=2, b+c=2a. Since b>a and c>=b>a, b+c > 2a. Contradiction.
    # So [129, 256] is valid.
    
    k = 129
    base_tail = set(range(k, n + 1))
    
    def can_add(x, S):
        """Check if x can be added to set S while maintaining the property."""
        # S is a set of ints. x is an int not in S.
        # We need to check triples (a,b,c) from S U {x} with a < b <= c.
        # Since x is new, x must be involved in the triple.
        
        S_list = sorted(S)
        
        # Case 1: x is the smallest element (a=x)
        # Need x < b <= c in S.
        # Check if x divides b+c.
        for i, b in enumerate(S_list):
            if x >= b:
                continue
            for c in S_list[i:]:
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle element (b=x)
        # Need a < x <= c in S.
        # Check if a divides x+c.
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest element (c=x)
        # Need a < b <= x in S U {x}.
        # Subcase 3a: b = x. Then a < x. Check if a divides x+x = 2x.
        for a in S_list:
            if a >= x:
                break
            if (2 * x) % a == 0:
                return False
        
        # Subcase 3b: b < x. Then a < b < x. Check if a divides b+x.
        for i, b in enumerate(S_list):
            if b >= x:
                break
            for a in S_list[:i]:
                if (b + x) % a == 0:
                    return False
        
        return True

    def is_valid_set(S):
        """Verify that set S satisfies the property for all triples."""
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            for j in range(i + 1, len(S_sorted)):
                b = S_sorted[j]
                for k_idx in range(j, len(S_sorted)):
                    c = S_sorted[k_idx]
                    if (b + c) % a == 0:
                        return False
        return True

    best_set = set(base_tail)
    best_size = len(best_set)
    
    candidates = list(range(1, k))
    
    # Strategy 1: Greedy with different orders
    orders = [
        'ascending',
        'descending',
        'random1',
        'random2',
        'random3'
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
            
        current_set = set(base_tail)
        
        if order == 'ascending':
            cands = list(range(1, k))
        elif order == 'descending':
            cands = list(range(k - 1, 0, -1))
        elif order == 'random1':
            cands = list(range(1, k))
            random.seed(1)
            random.shuffle(cands)
        elif order == 'random2':
            cands = list(range(1, k))
            random.seed(42)
            random.shuffle(cands)
        elif order == 'random3':
            cands = list(range(1, k))
            random.seed(123)
            random.shuffle(cands)
        
        # Greedy addition
        for x in cands:
            if time.time() > deadline:
                break
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    # Strategy 2: Local Search (Swaps)
    # Try to improve the best set found so far
    if time.time() < deadline - 10.0:
        current_set = set(best_set)
        improved = True
        iterations = 0
        max_iterations = 200
        
        while improved and iterations < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            current_list = sorted(current_set)
            
            # Try removing an element from the "small" part (1..128)
            # and adding a different one, then greedily filling
            small_elements = [x for x in current_list if x < k]
            
            for remove_x in small_elements:
                if time.time() > deadline - 5.0:
                    break
                    
                trial_set = set(current_set)
                trial_set.discard(remove_x)
                
                # Try adding any candidate not in trial_set
                # We prioritize adding candidates that might allow more additions
                # Just try all candidates in random order? Or just check if any single swap improves size immediately?
                # A swap that keeps size same might enable further additions.
                
                # Let's try to find a replacement that allows at least one more addition
                # Or simply: try all possible replacements, and for each, do a quick greedy fill
                # to see if we can beat the current size.
                
                # To be efficient, we can just check if the trial_set + {add_x} is valid,
                # and then try to add more.
                
                # We'll iterate through candidates not in trial_set
                # We can limit this to candidates close to remove_x or random
                # Let's just try all candidates 1..128 not in trial_set
                # This is 128 checks * can_add (O(N^2)) = 128 * 20000 = 2.5M ops.
                # This is fast enough for a few iterations.
                
                found_better = False
                for add_x in range(1, k):
                    if add_x in trial_set:
                        continue
                    
                    if can_add(add_x, trial_set):
                        new_set = trial_set | {add_x}
                        
                        # Try to greedily add more from the remaining candidates
                        # We need to define the order of remaining candidates
                        # Let's use descending order for speed (large numbers are easier to add)
                        remaining = [x for x in range(k-1, 0, -1) if x not in new_set]
                        
                        for x in remaining:
                            if can_add(x, new_set):
                                new_set.add(x)
                                
                        if len(new_set) > len(current_set):
                            current_set = new_set
                            improved = True
                            found_better = True
                            break
                        elif len(new_set) == len(current_set):
                            # If size is same, we might still prefer a set that is "more open"
                            # but checking "openness" is expensive.
                            # Let's just accept it if it's the same size and we haven't found a bigger one yet?
                            # No, that might lead to a local minimum.
                            # Let's only accept if it's strictly better or if we are stuck.
                            # Actually, if we don't accept same-size swaps, we might miss a path.
                            # But if we do, we might loop.
                            # Let's try a random swap if stuck.
                            pass
                
                if found_better:
                    break
                    
    # Final validation
    if best_set is None or not is_valid_set(best_set):
        return sorted(list(base_tail))
    
    return sorted(list(best_set))
