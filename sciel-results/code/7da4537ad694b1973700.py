import sys
import time

sys.setrecursionlimit(10000)

def construct():
    # 1. Generate a baseline using the greedy algorithm
    values, sums = [], set()
    for x in range(1, 257):
        # Calculate suffix sums of current 'values'
        suffix_sums = [0]
        s = 0
        for a in reversed(values):
            s += a
            suffix_sums.append(s)
        
        # Check if x can be added
        valid = True
        new_sums = []
        for s in suffix_sums:
            ns = x + s
            if ns in sums:
                valid = False
                break
            new_sums.append(ns)
        
        if valid:
            values.append(x)
            sums.update(new_sums)
            
    best_len = len(values)
    best_seq = values[:]
    
    # 2. Try to find a longer sequence using optimized DFS with pruning
    start_time = time.time()
    time_limit = 55.0 # Leave 5s for overhead
    
    # Target length to find
    # We know greedy gives 9. Let's try to find 10.
    # If we find 10, we can try 11, etc.
    
    def dfs(current_seq, current_sums, start_val, target_len):
        if time.time() - start_time > time_limit:
            return None
        
        if len(current_seq) == target_len:
            return current_seq[:]
        
        # Pruning 1: Not enough numbers left
        if len(current_seq) + (256 - start_val + 1) < target_len:
            return None
            
        # Pruning 2: Range of sums
        # We need to fit total_intervals = target_len * (target_len + 1) // 2 distinct sums.
        # Current min sum is min(current_seq) (or 0 if empty, but sums are >= 1)
        # Actually, the smallest sum is the smallest element.
        # The largest possible sum is sum(current_seq) + sum of the largest (target_len - len(current_seq)) available numbers.
        
        k_needed = target_len - len(current_seq)
        
        if not current_seq:
            min_sum = 1 # Smallest possible element
        else:
            min_sum = min(current_seq)
            
        # Max possible sum: sum of current + sum of largest k_needed numbers from [start_val, 256]
        # To maximize the sum, we pick the largest available numbers.
        # Available numbers are start_val ... 256.
        # We need to pick k_needed numbers. The largest are 256, 255, ...
        # But they must be > last_val? No, start_val is last_val + 1.
        # So the largest k_needed numbers in [start_val, 256] are:
        # If 256 - start_val + 1 >= k_needed:
        #   sum_largest = sum(range(256 - k_needed + 1, 257))
        # Else:
        #   sum_largest = sum(range(start_val, 257)) # And we already failed Pruning 1
        
        if 256 - start_val + 1 >= k_needed:
            sum_largest = sum(range(256 - k_needed + 1, 257))
        else:
            sum_largest = sum(range(start_val, 257))
            
        current_sum = sum(current_seq)
        max_possible_sum = current_sum + sum_largest
        
        total_intervals = target_len * (target_len + 1) // 2
        
        # The sums must be distinct integers in [min_sum, max_possible_sum]
        # So we need max_possible_sum - min_sum + 1 >= total_intervals
        if max_possible_sum - min_sum + 1 < total_intervals:
            return None
            
        last_val = current_seq[-1] if current_seq else 0
        
        # Calculate suffix sums for current_seq
        suffix_sums = [0]
        s = 0
        for a in reversed(current_seq):
            s += a
            suffix_sums.append(s)
            
        # Heuristic: Try smaller values first? Or larger?
        # Smaller values allow for more flexibility in sum range?
        # Let's stick to ascending order, it's standard.
        
        for x in range(start_val, 257):
            # Check validity
            valid = True
            new_sums_list = []
            for s in suffix_sums:
                ns = x + s
                if ns in current_sums:
                    valid = False
                    break
                new_sums_list.append(ns)
                
            if not valid:
                continue
                
            # Additional pruning before recursing?
            # The new min_sum might increase if x > min(current_seq)? No, min is min.
            # The new max_possible_sum will be tighter in the next call.
            
            next_sums = set(current_sums)
            next_sums.update(new_sums_list)
            
            res = dfs(current_seq + [x], next_sums, x + 1, target_len)
            if res is not None:
                return res
                
        return None

    # Try to find longer sequences
    target = best_len + 1
    # Upper bound: The sum of the first L integers is L(L+1)/2.
    # If we use the smallest integers, the sums are small.
    # Max length is bounded by the range [1, 256].
    # A rough upper bound: If we use 1..L, sum is ~L^2/2.
    # But we are not restricted to 1..L.
    # Let's just try up to 12.
    
    while target <= 12:
        if time.time() - start_time > time_limit:
            break
            
        res = dfs([], set(), 1, target)
        if res is not None:
            best_seq = res
            best_len = len(res)
            target += 1
        else:
            break

    return best_seq
