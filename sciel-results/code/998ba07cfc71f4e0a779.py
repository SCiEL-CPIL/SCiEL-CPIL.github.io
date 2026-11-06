import time
import random
import math

def construct():
    start = time.time()
    deadline = start + 80.0
    
    max_n = 512
    max_size = 48
    
    def violations(S):
        if len(S) < 3:
            return 0
        S_list = list(S)
        n = len(S_list)
        v_count = 0
        max_sum = sum(S_list)
        
        for i, x in enumerate(S_list):
            dp = 1
            for j, y in enumerate(S_list):
                if i != j:
                    dp |= (dp << y)
            
            for k in range(2, n):
                target = x * k
                if target > max_sum:
                    break
                if (dp >> target) & 1:
                    v_count += 1
        return v_count

    def repair(S):
        """Remove elements to make S valid."""
        S = list(S)
        v = violations(S)
        while v > 0 and len(S) > 0:
            # Find element with most violations
            # This requires re-calculating or tracking.
            # Simple: remove random element? Or element with highest value?
            # Let's remove the element that appears in the most "conflicting" subsets.
            # Approximation: remove random element.
            # Better: remove element x that minimizes violations(S \ {x})?
            # That is O(N) violations calls.
            # N=48. 48 calls. Fast enough.
            
            best_v = float('inf')
            best_idx = -1
            for i in range(len(S)):
                S_temp = S[:i] + S[i+1:]
                v_temp = violations(S_temp)
                if v_temp < best_v:
                    best_v = v_temp
                    best_idx = i
            
            S = S[:best_idx] + S[best_idx+1:]
            v = best_v
            
        return S

    def simulated_annealing(initial_S, time_limit):
        S = list(initial_S)
        current_v = violations(S)
        
        if current_v == 0:
            return S
        
        T = 5.0
        cool_rate = 0.999
        min_T = 0.01
        
        S_set = set(S)
        all_numbers = set(range(1, max_n + 1))
        
        while time.time() < time_limit and current_v > 0:
            a = random.choice(S)
            # Pick b not in S
            # To avoid list creation, we can pick random int and check
            b = random.randint(1, max_n)
            while b in S_set:
                b = random.randint(1, max_n)
            
            S_idx = S.index(a)
            S[S_idx] = b
            S_set.remove(a)
            S_set.add(b)
            
            new_v = violations(S)
            delta = new_v - current_v
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_v = new_v
            else:
                S[S_idx] = a
                S_set.remove(b)
                S_set.add(a)
            
            T *= cool_rate
            if T < min_T:
                T = min_T
        
        return S

    def greedy_build(order, time_limit):
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        
        def build_dp(S):
            dp_local = [0] * (max_size + 1)
            dp_local[0] = 1
            for x in S:
                for k in range(len(S) - 1, -1, -1):
                    dp_local[k + 1] |= (dp_local[k] << x)
            return dp_local

        def check_valid(x, S, dp):
            n_s = len(S)
            for k in range(2, n_s + 1):
                target = x * k
                if target > max_n * max_size:
                    continue
                if (dp[k] >> target) & 1:
                    return False
            for y in S:
                for j in range(1, n_s + 1):
                    target = y * (j + 1) - x
                    if target < 0:
                        continue
                    if target > max_n * max_size:
                        continue
                    if (dp[j] >> target) & 1:
                        return False
            return True

        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > time_limit:
                break
            if x in S:
                continue
            if check_valid(x, S, dp):
                S.append(x)
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
        return S

    best_set = []
    best_size = 0
    
    # 1. Greedy baseline
    orders = [
        range(1, max_n + 1),
        range(max_n, 0, -1),
    ]
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        S_g = greedy_build(order, deadline - 5.0)
        if len(S_g) > best_size:
            best_set = S_g
            best_size = len(S_g)
            if best_size >= max_size:
                return best_set

    random.seed(42)
    # Random greedy
    while time.time() < deadline - 20.0:
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        S_g = greedy_build(order, deadline - 20.0)
        if len(S_g) > best_size:
            best_set = S_g
            best_size = len(S_g)
            if best_size >= max_size:
                return best_set

    # 2. SA + Repair
    for _ in range(5):
        if time.time() > deadline - 10.0:
            break
        
        S_init = random.sample(range(1, max_n + 1), max_size)
        S_sa = simulated_annealing(S_init, deadline - 10.0)
        
        v = violations(S_sa)
        if v == 0:
            return S_sa
        
        S_rep = repair(S_sa)
        if len(S_rep) > best_size:
            best_set = S_rep
            best_size = len(S_rep)
            if best_size >= max_size:
                return best_set

    return best_set
