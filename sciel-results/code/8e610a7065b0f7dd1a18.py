import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 512
    target = 48

    def build_dp(arr, max_k):
        m = len(arr)
        if m == 0:
            return [1] + [0]*max_k
        dp = [0] * (max_k + 1)
        dp[0] = 1
        for val in arr:
            for k in range(min(m, max_k), 0, -1):
                dp[k] |= (dp[k - 1] << val)
        return dp

    def is_valid_set(S):
        m = len(S)
        if m <= 2:
            return True
        
        for i, x in enumerate(S):
            others = S[:i] + S[i+1:]
            m_others = len(others)
            sum_others = sum(others)
            
            # Check k=2 first (AP)
            # Is there a, b in others such that a+b = 2x?
            # a+b = 2x => b = 2x - a
            # Use set for O(1) lookup
            others_set = set(others)
            for a in others:
                b = 2 * x - a
                if b != a and b in others_set:
                    return False
            
            # Check k >= 3
            max_k_possible = min(m_others, sum_others // x)
            if max_k_possible < 3:
                continue
                
            dp = build_dp(others, max_k_possible)
            for k in range(3, max_k_possible + 1):
                target_sum = k * x
                if (dp[k] >> target_sum) & 1:
                    return False
        return True

    def greedy_build(order):
        C = []
        for x in order:
            if len(C) >= target:
                break
            C.append(x)
            if is_valid_set(C):
                pass
            else:
                C.pop()
        return C

    best_set = []
    rng = random.Random(42)
    
    # Try descending first (fast to check if it works)
    order = list(range(n, 0, -1))
    C = greedy_build(order)
    if len(C) > len(best_set):
        best_set = C
        
    # Try ascending
    order = list(range(1, n + 1))
    C = greedy_build(order)
    if len(C) > len(best_set):
        best_set = C
        
    # Random trials
    trials = 0
    while time.time() < deadline - 2.0:
        trials += 1
        if len(best_set) >= target:
            break
            
        order = list(range(1, n + 1))
        rng.shuffle(order)
        
        C = greedy_build(order)
        if len(C) > len(best_set):
            best_set = C
            
    return best_set
