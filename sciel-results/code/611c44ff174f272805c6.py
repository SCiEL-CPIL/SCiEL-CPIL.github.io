import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin

    def is_valid(S):
        if not S:
            return True
        n = len(S)
        # Generate all non-empty subset sums
        current_sums = {0}
        for x in S:
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + x)
            current_sums |= new_sums
        
        sums = current_sums - {0}
        
        # Check for duplicate sums
        # If the number of unique sums is less than the number of non-empty subsets, 
        # there are duplicates, which is invalid.
        if (1 << n) - 1 != len(sums):
            return False
            
        # Check divisibility
        sorted_sums = sorted(sums)
        sum_set = set(sorted_sums)
        max_sum = sorted_sums[-1]
        
        for s in sorted_sums:
            multiple = 2 * s
            while multiple <= max_sum:
                if multiple in sum_set:
                    return False
                multiple += s
        return True

    def greedy_construct(order_type='random'):
        S = set()
        candidates = list(range(1, 257))
        
        if order_type == 'random':
            random.shuffle(candidates)
        elif order_type == 'ascending':
            pass # Already sorted
        elif order_type == 'descending':
            candidates.sort(reverse=True)
            
        for x in candidates:
            test_S = list(S) + [x]
            if is_valid(test_S):
                S.add(x)
        return list(S)

    def local_search(initial_S, time_limit):
        S = set(initial_S)
        best_S = list(S)
        best_size = len(S)
        start = time.time()
        
        while time.time() < start + time_limit:
            current_size = len(S)
            
            # Try to add a new element
            added = False
            candidates = [x for x in range(1, 257) if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                test_S = list(S) + [x]
                if is_valid(test_S):
                    S.add(x)
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break
            
            if added:
                continue
                
            # Try a swap if no addition was possible
            if current_size > 0 and candidates:
                # Pick a random element to remove
                to_remove = random.choice(list(S))
                S.remove(to_remove)
                
                # Pick a random element to add
                to_add = random.choice(candidates)
                test_S = list(S) + [to_add]
                if is_valid(test_S):
                    S.add(to_add)
                    # Size is unchanged, so we don't update best_S
                    # But the new configuration might allow future additions
                else:
                    # Revert the removal
                    S.add(to_remove)
                    
        return best_S

    best_set = []
    
    # Initial quick greedy to establish a baseline
    baseline = greedy_construct('random')
    if len(baseline) > len(best_set):
        best_set = baseline

    # Main search loop
    while time.time() < deadline - 5.0:
        # Generate multiple greedy candidates with different strategies
        candidates = []
        candidates.append(greedy_construct('random'))
        candidates.append(greedy_construct('ascending'))
        candidates.append(greedy_construct('descending'))
        
        # Select the best candidate to refine
        best_candidate = max(candidates, key=len)
        
        # Determine search time
        remaining = deadline - time.time()
        search_time = min(4.0, remaining - 1.0)
        
        if search_time <= 0:
            break
            
        # Refine the best candidate
        improved = local_search(best_candidate, search_time)
        
        # Update global best
        if len(improved) > len(best_set):
            best_set = improved

    return list(best_set)
