import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin

    best_set = [31, 32, 34]
    best_len = 3

    def get_sums(S):
        sums = set()
        n = len(S)
        for i in range(1, 1 << n):
            s = 0
            for j in range(n):
                if i & (1 << j):
                    s += S[j]
            sums.add(s)
        return sums

    def is_valid_sums(sums):
        # sums is a set of integers
        # Check if any u in sums divides any other v in sums (v != u)
        # Equivalent to: for all u in sums, no multiple of u (except u) is in sums
        # Max sum is bounded.
        max_sum = max(sums) if sums else 0
        for u in sums:
            # Check multiples 2u, 3u, ... up to max_sum
            # If k*u is in sums, return False
            for k in range(2, max_sum // u + 1):
                if k * u in sums:
                    return False
        return True

    def try_extend(S, candidates, time_limit):
        S = list(S)
        sums = get_sums(S)
        
        # Sort candidates randomly
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in S:
                continue
            
            # New sums
            new_sums = set(sums)
            new_sums.add(x)
            for s in sums:
                new_sums.add(s + x)
            
            if is_valid_sums(new_sums):
                S.append(x)
                sums = new_sums
        
        return S

    all_numbers = list(range(1, 257))
    
    # We will run several attempts with different random seeds/orders
    # Each attempt starts from the seed [31, 32, 34]
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        random.seed(attempt)
        current_set = [31, 32, 34]
        candidates = [x for x in all_numbers if x not in current_set]
        
        extended_set = try_extend(current_set, candidates, deadline)
        
        if len(extended_set) > best_len:
            best_len = len(extended_set)
            best_set = extended_set
            # If we find a very good set, we can try to extend it further in subsequent iterations
            # But the current loop structure restarts from seed each time.
            # To leverage progress, we could use the best_set as the new seed for the next iteration?
            # No, that might get stuck in a local optimum.
            # However, if we find size 5, starting from size 5 is better than starting from size 3.
            # Let's update the seed to the best set found so far?
            # Risk: if the best set is a local optimum, we might not escape.
            # But random shuffling of candidates helps.
            
            # Let's try a hybrid: keep the original seed for some attempts, and the best found for others.
            if attempt % 10 == 0:
                # Force restart from original seed occasionally to avoid getting stuck
                pass 
            else:
                # Use the current best as the base for extension
                # But try_extend modifies S. We passed current_set which was a copy.
                # Let's just update the base for the next iteration if we improved.
                # Actually, the loop structure is:
                # 1. Pick a base set (seed or best)
                # 2. Try to extend it
                # 3. If improved, update best.
                
                # Let's change the logic:
                # Instead of always starting from [31,32,34], 
                # let's start from `best_set` in subsequent iterations if we want to grow it.
                # But `best_set` is a list.
                
                # Let's just stick to starting from the original seed for simplicity and robustness,
                # unless the improvement is significant.
                # Actually, extending a larger set is harder (more sums to check).
                # But finding a larger set from scratch is also hard.
                
                # Let's try extending the best_set found in the previous iteration.
                # We need to be careful not to use the same candidate order.
                pass

    return best_set
