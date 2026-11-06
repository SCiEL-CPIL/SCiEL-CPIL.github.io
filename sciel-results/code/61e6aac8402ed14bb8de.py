import time
import random

def construct():
    start = time.time()
    deadline = start + 55  # safety margin before 90s kill

    # Precompute base-3 candidates (numbers with no digit '2' in base-3)
    # These are 3-AP-free, a strong starting point
    base3 = []
    for i in range(1, 513):
        n = i
        ok = True
        while n > 0:
            if n % 3 == 2:
                ok = False
                break
            n //= 3
        if ok:
            base3.append(i)
    base3.sort()

    def is_valid_set(S):
        """Check if no element is the mean of any subset of >=2 other elements."""
        if len(S) < 2:
            return True
        n = len(S)
        S_set = set(S)
        
        for x in S:
            others = [v for v in S if v != x]
            if len(others) < 2:
                continue
            m = len(others)
            max_sum = sum(others)
            
            # DP: dp[k] is a bitset of achievable sums using exactly k elements from others
            dp = [0] * (m + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            for val in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k - 1] << val)
            
            # Check if x is mean of any subset of size k>=2
            # Mean = x means sum = k*x
            for k in range(2, m + 1):
                target = k * x
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def fast_check_new(S, new_elem):
        """Fast check: can we add new_elem to S? Only need to check new_elem against S,
        and check if any existing element becomes mean of a subset involving new_elem."""
        if len(S) < 1:
            return True
        S_set = set(S)
        
        # Check 1: new_elem is not mean of any subset of >=2 elements from S
        others = S
        if len(others) < 2:
            check1_pass = True
        else:
            m = len(others)
            max_sum = sum(others)
            dp = [0] * (m + 1)
            dp[0] = 1
            for val in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k - 1] << val)
            check1_pass = True
            for k in range(2, m + 1):
                target = k * new_elem
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    check1_pass = False
                    break
        
        if not check1_pass:
            return False
        
        # Check 2: For each x in S, x is not mean of any subset of >=2 from (S\{x}) ∪ {new_elem}
        # This is expensive. We'll do the full check if needed.
        # For speed, let's just do the full is_valid_set on S + [new_elem]
        return is_valid_set(S + [new_elem])

    # Strategy: Greedy with multiple orderings, then local search
    best_set = []
    
    # Try ascending greedy on base3
    S = []
    for x in base3:
        if len(S) >= 48:
            break
        if time.time() > deadline:
            break
        if is_valid_set(S + [x]):
            S.append(x)
    if len(S) > len(best_set):
        best_set = S
    if len(best_set) >= 48:
        return best_set

    # Try descending greedy
    if time.time() < deadline:
        S = []
        for x in reversed(base3):
            if len(S) >= 48:
                break
            if time.time() > deadline:
                break
            if is_valid_set(S + [x]):
                S.append(x)
        if len(S) > len(best_set):
            best_set = S
        if len(best_set) >= 48:
            return best_set

    # Try several random shuffles
    random.seed(42)
    for trial in range(20):
        if time.time() >= deadline:
            break
        cands = base3[:]
        random.shuffle(cands)
        S = []
        for x in cands:
            if len(S) >= 48:
                break
            if time.time() > deadline:
                break
            if is_valid_set(S + [x]):
                S.append(x)
        if len(S) > len(best_set):
            best_set = S
        if len(best_set) >= 48:
            return best_set

    # Local search: try to improve by swapping
    if len(best_set) < 48 and time.time() < deadline:
        used = set(best_set)
        pool = [x for x in base3 if x not in used]
        random.shuffle(pool)
        
        improved = True
        while improved and len(best_set) < 48 and time.time() < deadline:
            improved = False
            # Try adding from pool
            for x in pool[:]:
                if len(best_set) >= 48 or time.time() > deadline:
                    break
                if is_valid_set(best_set + [x]):
                    best_set.append(x)
                    pool.remove(x)
                    improved = True
                    break
            
            # Try swaps
            if not improved and len(best_set) < 48:
                for i in range(min(30, len(pool))):
                    if time.time() > deadline:
                        break
                    x_new = pool[i]
                    for j, x_old in enumerate(best_set):
                        new_S = best_set[:j] + best_set[j+1:] + [x_new]
                        if is_valid_set(new_S):
                            best_set = new_S
                            pool.remove(x_new)
                            improved = True
                            break
                    if improved:
                        break

    # Expand to all numbers if still stuck
    if len(best_set) < 48 and time.time() < deadline:
        used = set(best_set)
        all_nums = [x for x in range(1, 513) if x not in used]
        random.shuffle(all_nums)
        for x in all_nums:
            if len(best_set) >= 48 or time.time() > deadline:
                break
            if is_valid_set(best_set + [x]):
                best_set.append(x)

    if is_valid_set(best_set):
        return best_set
    return best_set

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s, Size: {len(res)}")
