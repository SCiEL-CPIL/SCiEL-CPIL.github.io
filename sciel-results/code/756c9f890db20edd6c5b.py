import time
import random

def _is_valid_set(S):
    """
    Check if S is a valid set.
    """
    if len(S) < 3:
        return True
        
    S = list(S)
    N = len(S)
    
    for i in range(N):
        a = S[i]
        others = S[:i] + S[i+1:]
        n_others = len(others)
        
        if n_others < 2:
            continue
            
        max_sum = sum(others)
        
        dp = [0] * (n_others + 1)
        dp[0] = 1
        
        for val in others:
            for k in range(n_others - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << val)
                    
        for k in range(2, n_others + 1):
            target_sum = k * a
            if target_sum <= max_sum:
                if (dp[k] >> target_sum) & 1:
                    return False
            else:
                break
                
    return True

def _fast_pair_check(S, x):
    S_set = set(S)
    for a in S:
        b = 2 * x - a
        if b in S_set and b != a:
            return True
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a and b != x:
            return True
    return False

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_set = []
    best_len = 0
    
    candidates = list(range(1, 513))
    rng = random.Random(42)
    
    # Helper to run a greedy pass
    def run_greedy(order):
        current_set = []
        for x in order:
            if time.time() - start_time > deadline - 2:
                break
            if len(current_set) >= 48:
                break
            if _fast_pair_check(current_set, x):
                continue
            test_set = current_set + [x]
            if _is_valid_set(test_set):
                current_set.append(x)
        return current_set

    # 1. Deterministic orders
    orders = [
        candidates[::-1],
        candidates[:],
        sorted(candidates, key=lambda x: x % 7),
        sorted(candidates, key=lambda x: x % 11)
    ]
    
    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
        s = run_greedy(order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s
        if best_len >= 48:
            break

    # 2. Randomized Greedy
    while time.time() - start_time < deadline - 10:
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        s = run_greedy(rand_order)
        if len(s) > best_len:
            best_len = len(s)
            best_set = s
            if best_len >= 48:
                break

    # 3. Local Search (Swaps) if we are close to 48
    if best_len >= 40 and best_len < 48:
        # Try to improve best_set
        improved = True
        while improved and time.time() - start_time < deadline - 5:
            improved = False
            # Try swapping each element in best_set with a candidate
            # To save time, only try a few random swaps per iteration
            for _ in range(20):
                if time.time() - start_time > deadline - 5:
                    break
                i = rng.randrange(len(best_set))
                a = best_set[i]
                # Pick a random candidate not in best_set
                b = rng.choice([c for c in candidates if c not in set(best_set)])
                
                new_set = best_set[:]
                new_set[i] = b
                if _is_valid_set(new_set):
                    # Check if we can add more
                    # Try to add any candidate
                    for c in candidates:
                        if c in set(new_set):
                            continue
                        if _fast_pair_check(new_set, c):
                            continue
                        test_set = new_set + [c]
                        if _is_valid_set(test_set):
                            new_set.append(c)
                            best_set = new_set
                            best_len = len(best_set)
                            improved = True
                            break
                    if best_len >= 48:
                        break

    if _is_valid_set(best_set):
        return best_set
    else:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
