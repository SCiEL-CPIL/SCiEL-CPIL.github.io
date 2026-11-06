import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit

    best_set = []
    best_len = 0

    def is_valid_and_violators(S):
        """
        Returns (is_valid, list_of_violators).
        """
        n = len(S)
        if n < 3:
            return True, []
        
        S_sorted = sorted(S)
        violators = []
        
        for i, x in enumerate(S_sorted):
            others = S_sorted[:i] + S_sorted[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for val in others:
                for k in range(m, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << val)
                        
            found = False
            for k in range(2, m + 1):
                target_sum = k * x
                if dp[k] & (1 << target_sum):
                    found = True
                    break
                    
            if found:
                violators.append(x)
                
        return len(violators) == 0, violators

    def prune_to_valid(S):
        """
        Removes elements until S is valid.
        """
        S = list(S)
        # Limit iterations to avoid infinite loops in case of bugs
        max_iter = len(S) + 10
        while max_iter > 0:
            max_iter -= 1
            valid, violators = is_valid_and_violators(S)
            if valid:
                return S
            if not violators:
                return S
            
            val_to_remove = random.choice(violators)
            try:
                S.remove(val_to_remove)
            except ValueError:
                # Should not happen
                break
                
        return S

    def expand_greedy(S):
        """
        Tries to add elements to S to maximize size.
        """
        S = list(S)
        S_set = set(S)
        candidates = [x for x in range(1, 513) if x not in S_set]
        random.shuffle(candidates)
        
        changed = True
        while changed and len(S) < 512:
            changed = False
            for x in candidates:
                if x in S_set:
                    continue
                test_set = S + [x]
                valid, _ = is_valid_and_violators(test_set)
                if valid:
                    S.append(x)
                    S_set.add(x)
                    candidates.remove(x)
                    changed = True
        return S

    # 1. Try Spread initialization
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(lo)
        if lo != hi:
            spread.append(hi)
        lo += 1
        hi -= 1
        
    pruned_spread = prune_to_valid(spread)
    expanded_spread = expand_greedy(pruned_spread)
    if len(expanded_spread) > best_len:
        best_set = expanded_spread
        best_len = len(best_set)

    # 2. Try Random initializations
    # Use a variety of initial sizes
    sizes = [30, 40, 50, 60, 70, 80, 90]
    
    while time.time() - start_time < deadline:
        init_size = random.choice(sizes)
        init_set = random.sample(range(1, 513), init_size)
        
        pruned_set = prune_to_valid(init_set)
        expanded_set = expand_greedy(pruned_set)
        
        if len(expanded_set) > best_len:
            best_set = expanded_set
            best_len = len(best_set)
            
        # If we reached 48 or more, we can try to be more aggressive
        # or just keep searching for 49+
        
    return best_set
