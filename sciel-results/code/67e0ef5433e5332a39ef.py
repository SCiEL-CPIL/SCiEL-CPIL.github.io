import time
import random

def construct():
    start_time = time.time()
    max_val = 256
    max_size = 60
    deadline = start_time + 55.0  # 5s margin
    
    best_set = []
    best_size = 0
    
    def check_set_validity(S):
        n = len(S)
        if n == 0:
            return True
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in S:
            for k in range(n, 0, -1):
                dp[k] |= (dp[k-1] << x)
        for i in range(n + 1):
            di = dp[i]
            if di == 0:
                continue
            for j in range(i + 1, n + 1):
                if di & dp[j]:
                    return False
        return True

    def try_add(dp, x, current_size):
        new_dp = dp[:]
        max_k_new = current_size + 1
        for k in range(1, max_k_new + 1):
            new_dp[k] |= (dp[k-1] << x)
        for i in range(max_k_new + 1):
            di = new_dp[i]
            if di == 0:
                continue
            for j in range(i + 1, max_k_new + 1):
                if di & new_dp[j]:
                    return None
        return new_dp

    def run_order(order):
        nonlocal best_set, best_size
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        current_size = 0
        for x in order:
            if current_size >= max_size:
                break
            new_dp = try_add(dp, x, current_size)
            if new_dp is not None:
                S.append(x)
                dp = new_dp
                current_size += 1
        if current_size > best_size:
            best_size = current_size
            best_set = S[:]
            if best_size == max_size:
                return True
        return False

    # Strategy 1: Top-k direct verification
    for k in range(1, max_size + 1):
        if time.time() > deadline:
            break
        top_k = list(range(256 - k + 1, 257))
        if check_set_validity(top_k):
            best_set = top_k
            best_size = k
        else:
            break

    # Strategy 2: Extend the top-k set with smaller elements
    if best_size > 0 and time.time() < deadline:
        base_set = best_set[:]
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in base_set:
            for k in range(len(base_set), 0, -1):
                dp[k] |= (dp[k-1] << x)
        current_size = len(base_set)
        base_set_set = set(base_set)
        remaining = [x for x in range(1, 257) if x not in base_set_set]
        for x in sorted(remaining, reverse=True):
            if time.time() > deadline:
                break
            if current_size >= max_size:
                break
            new_dp = try_add(dp, x, current_size)
            if new_dp is not None:
                base_set.append(x)
                dp = new_dp
                current_size += 1
        if current_size > best_size:
            best_size = current_size
            best_set = base_set[:]

    # Strategy 3: Greedy with various orders
    orders = [
        list(range(1, max_val + 1)),
        list(range(max_val, 0, -1)),
    ]
    for order in orders:
        if time.time() > deadline:
            break
        if run_order(order):
            break
        if best_size == max_size:
            break

    # Strategy 4: Random restarts
    while time.time() < deadline:
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        if run_order(candidates):
            break
        if best_size == max_size:
            break

    return best_set
