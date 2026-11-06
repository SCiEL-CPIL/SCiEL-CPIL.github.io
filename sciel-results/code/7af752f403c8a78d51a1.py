import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    max_n = 512
    max_size = 48

    def get_violations(S):
        S_list = list(S)
        n = len(S_list)
        if n < 3:
            return []

        violations = []
        
        for i, y in enumerate(S_list):
            elements = [x for j, x in enumerate(S_list) if j != i]
            m = len(elements)

            dp = [0] * (m + 1)
            dp[0] = 1
            current_max_k = 0

            for x in elements:
                for k in range(current_max_k, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= (dp[k] << x)
                current_max_k += 1

            for k in range(2, m + 1):
                target_sum = k * y
                if target_sum > 512 * k:
                    continue
                if (dp[k] >> target_sum) & 1:
                    violations.append((y, k))
                    
        return violations

    def is_valid(S):
        return len(get_violations(S)) == 0

    def greedy_build(order):
        S = []
        for x in order:
            if len(S) >= max_size:
                break
            new_S = S + [x]
            if is_valid(new_S):
                S = new_S
        return S

    # 1. Greedy initializations to ensure a valid baseline
    random.seed(42)
    
    best_valid_set = []
    
    # Try a few greedy orders
    orders = [
        range(1, max_n + 1),
        range(max_n, 0, -1),
    ]
    
    for _ in range(5):
        o = list(range(1, max_n + 1))
        random.shuffle(o)
        orders.append(o)
        
    for o in orders:
        if time.time() > deadline - 20.0:
            break
        s = greedy_build(o)
        if len(s) > len(best_valid_set):
            best_valid_set = s
            
    if len(best_valid_set) >= max_size:
        return best_valid_set

    # 2. Try to expand best_valid_set using SA or random swaps
    current_set = list(best_valid_set)
    all_numbers = set(range(1, max_n + 1))
    
    # If we are far from 48, maybe try random 48 sets directly?
    # Let's try 50 random sets of size 48.
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        S = set(random.sample(range(1, max_n + 1), max_size))
        if is_valid(S):
            return list(S)

    # 3. SA to improve current_set towards 48 valid
    # Start from a random 48 set if current is small?
    # No, start from best_valid_set and try to add/swap.
    
    # If best_valid_set is small, we might need to rebuild.
    # Let's start SA from a random 48 set, but track best valid found.
    
    random_set = list(random.sample(range(1, max_n + 1), max_size))
    current_set = random_set
    current_violations = len(get_violations(current_set))
    
    temp = 50.0
    cooling = 0.995
    
    while time.time() < deadline - 2.0:
        if current_violations == 0:
            if len(current_set) > len(best_valid_set):
                best_valid_set = current_set
            return current_set
        
        viols = get_violations(current_set)
        
        if viols:
            y_remove = random.choice(viols)[0]
        else:
            y_remove = random.choice(current_set)
            
        candidates = [x for x in all_numbers if x not in current_set]
        if not candidates:
            break
            
        x_add = random.choice(candidates)
        
        new_set = [x for x in current_set if x != y_remove]
        new_set.append(x_add)
        
        new_viols = get_violations(new_set)
        new_violations = len(new_viols)
        
        delta = new_violations - current_violations
        
        if delta < 0 or (temp > 0.01 and random.random() < pow(sys.float_info.max, -delta / temp)):
            current_set = new_set
            current_violations = new_violations
            
            if current_violations == 0:
                if len(current_set) > len(best_valid_set):
                    best_valid_set = current_set
                return current_set

        temp *= cooling
        
    return best_valid_set

