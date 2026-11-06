import random

def construct():
    best_set = []
    
    def is_valid(S):
        n = len(S)
        if n < 2:
            return True
        for a in S:
            others = [y for y in S if y != a]
            m = len(others)
            if m < 2:
                continue
            dp = [0] * (m + 1)
            dp[0] = 1
            for y in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << y)
            max_sum = sum(others)
            for k in range(2, m + 1):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def run_greedy(order):
        selected = []
        for x in order:
            candidate = selected + [x]
            if is_valid(candidate):
                selected = candidate
        return selected

    # Deterministic orders
    orders = [
        list(range(512, 0, -1)),
        list(range(1, 513)),
    ]
    
    # Randomized orders
    # Bias towards large numbers
    base_list = list(range(1, 513))
    for _ in range(100):
        random.shuffle(base_list)
        # Sort by random key with bias?
        # Or just shuffle and try.
        # To bias, we can sample without replacement, picking larger numbers with higher prob.
        # Simpler: just shuffle.
        orders.append(base_list[:])
        
    # Even better: Heuristic order
    # Try adding numbers in descending order but skip some if they cause issues?
    # No, greedy is greedy.
    
    for order in orders:
        res = run_greedy(order)
        if len(res) > len(best_set):
            best_set = res
            if len(best_set) == 48:
                break
                
    return best_set
