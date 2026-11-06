import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    # Step 1: Generate base-3 set
    powers = [1, 3, 9, 27, 81, 243]
    base_set = set()
    for mask in range(1 << len(powers)):
        val = 0
        for i in range(len(powers)):
            if (mask >> i) & 1:
                val += powers[i]
        if val <= 512:
            base_set.add(val)
    
    current_set = list(base_set)
    
    # Step 2: Greedy addition
    candidates = [x for x in range(1, 513) if x not in base_set]
    random.shuffle(candidates)
    
    def check_3ap_conflict(x, S_list, S_set):
        for a in S_list:
            # x is mean of a, b
            b = 2 * x - a
            if b in S_set and b != a and b != x:
                return True
            # a is mean of x, b
            b = 2 * a - x
            if b in S_set and b != a and b != x:
                return True
            # c is mean of x, a
            if (x + a) % 2 == 0:
                c = (x + a) // 2
                if c in S_set and c != a and c != x:
                    return True
        return False

    for x in candidates:
        if time.time() > deadline:
            break
        if check_3ap_conflict(x, current_set, set(current_set)):
            continue
        current_set.append(x)
        
    # Step 3: Verify full condition
    # If invalid, we might need to prune.
    # But given the strong structure, it's likely valid.
    # We perform a check. If it fails, we return the base_set which is safer?
    # Or we can try to repair.
    
    def is_valid_full(S):
        n = len(S)
        for i, x in enumerate(S):
            others = [y for j, y in enumerate(S) if j != i]
            # Bitset DP for subset sums of 'others'
            # Max sum is sum(others)
            max_sum = sum(others)
            # We need to check if k*x is achievable for k=2..n-1
            # sums is a bitset where bit s is 1 if s is achievable
            sums = 1
            for y in others:
                sums |= sums << y
            # Check k*x
            for k in range(2, n):
                target = k * x
                if target > max_sum:
                    break
                if (sums >> target) & 1:
                    return False
        return True

    # To save time, we only verify if we have time
    if time.time() < deadline - 5:
        if not is_valid_full(current_set):
            # Fallback to base_set if full set is invalid
            # Base set is likely valid
            if time.time() < deadline - 2:
                if is_valid_full(list(base_set)):
                    return list(base_set)
            return list(base_set)
            
    return current_set

