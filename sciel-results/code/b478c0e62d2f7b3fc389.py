import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    
    def elapsed():
        return time.time() - start

    def check_valid(S):
        """Check if set S is valid: no element is the mean of any subset of >=2 OTHER elements."""
        n = len(S)
        if n < 3:
            return True
        
        # For each element x in S, check x is not mean of any subset of S\{x} with size >= 2
        # x = sum(subset)/k  =>  sum(subset) = k*x
        # So we need: for no k>=2, k*x is achievable as sum of exactly k distinct elements from S\{x}
        
        S_set = set(S)
        
        for idx_x, x in enumerate(S):
            others = [S[j] for j in range(n) if j != idx_x]
            m = len(others)
            if m < 2:
                continue
            
            max_sum = sum(others)
            # DP: dp[k] is a set (or bitset) of achievable sums using exactly k elements from others
            # Use Python integers as bitsets for efficiency
            # dp[k] has bit s set if sum s is achievable with exactly k elements
            dp = [0] * (m + 1)
            dp[0] = 1  # empty subset has sum 0
            
            for e in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << e)
            
            # Check if k*x is achievable for any k >= 2
            for k in range(2, m + 1):
                target = k * x
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        
        return True

    def incremental_check_add(S, x):
        """Check if adding x to S keeps it valid.
        Two conditions:
        1. x is not mean of any subset of S with size >= 2
        2. No element a in S becomes mean of a subset of (S\{a}) ∪ {x} with size >= 2
        
        For condition 1: x not mean of subset of S => k*x not achievable as sum of k elements from S
        For condition 2: For each a in S, a not mean of subset of (S\{a}) ∪ {x} with size >= 2.
            This means: for no k>=2, k*a is achievable as sum of k elements from (S\{a}) ∪ {x}.
            The subsets that include x: sum = (k-1)*a - x... wait.
            Actually: subset T of (S\{a}) ∪ {x} with |T| = k, sum(T) = k*a.
            If x ∉ T: already checked when S was valid.
            If x ∈ T: sum(T\{x}) = k*a - x, where T\{x} ⊆ S\{a}, |T\{x}| = k-1 >= 1.
            So we need: for no k>=2, (k*a - x) is achievable as sum of (k-1) elements from S\{a}.
        """
        n = len(S)
        
        # Condition 1: x not mean of subset of S
        if n >= 2:
            max_sum = sum(S)
            dp = [0] * (n + 1)
            dp[0] = 1
            for e in S:
                for k in range(n, 0, -1):
                    dp[k] |= (dp[k-1] << e)
            for k in range(2, n + 1):
                target = k * x
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        
        # Condition 2: For each a in S, check a not mean of subset of (S\{a}) ∪ {x}
        for a in S:
            others = [v for v in S if v != a]
            m = len(others)
            if m < 1:
                continue
            
            max_sum = sum(others)
            dp = [0] * (m + 1)
            dp[0] = 1
            for e in others:
                for k in range(m, 0, -1):
                    dp[k] |= (dp[k-1] << e)
            
            # For k >= 2: need (k*a - x) achievable as sum of (k-1) elements from others
            # j = k-1, j >= 1: target = (j+1)*a - x, achievable as sum of j elements from others
            for j in range(1, m + 1):
                target = (j + 1) * a - x
                if target < 0 or target > max_sum:
                    continue
                if (dp[j] >> target) & 1:
                    return False
        
        return True

    def quick_pair_check(S, x):
        """Fast necessary condition: 2x != a+b for distinct a,b in S."""
        S_set = set(S)
        for a in S:
            b = 2 * x - a
            if b != a and b in S_set:
                return False
        return True

    # Main construction using multiple strategies
    best_set = []
    best_len = 0
    
    # Strategy 1: Greedy with various orderings
    strategies = []
    
    # Descending
    strategies.append(list(range(512, 0, -1)))
    
    # Ascending
    strategies.append(list(range(1, 513)))
    
    # Randomized with multiple seeds
    for seed in range(20):
        cands = list(range(1, 513))
        random.seed(seed * 7919 + 13)
        random.shuffle(cands)
        strategies.append(cands)
    
    # Modular strategies
    for mod, rem in [(2,0), (2,1), (3,0), (3,1), (3,2), (4,1), (4,3), (5,1), (5,2), (5,3), (5,4)]:
        primary = [x for x in range(1, 513) if x % mod == rem]
        secondary = [x for x in range(1, 513) if x % mod != rem]
        strategies.append(primary + secondary)
    
    # Powers of 2 based
    p2 = [2**i for i in range(1, 10) if 2**i <= 512]
    rest = [x for x in range(1, 513) if x not in p2]
    strategies.append(p2 + rest)
    
    # Fibonacci-like spacing
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    fib_scaled = [x for x in range(1, 513) if any(abs(x - f) <= 1 for f in fib)]
    fib_rest = [x for x in range(1, 513) if x not in fib_scaled]
    strategies.append(fib_scaled + fib_rest)
    
    for candidates in strategies:
        if elapsed() > deadline - 5.0:
            break
        
        current = []
        used = set()
        
        for x in candidates:
            if elapsed() > deadline - 5.0:
                break
            if x in used:
                continue
            if len(current) >= 48:
                break
            
            # Quick precheck
            if current and not quick_pair_check(current, x):
                continue
            
            # Full incremental check
            if incremental_check_add(current, x):
                current.append(x)
                used.add(x)
        
        if len(current) > best_len:
            best_len = len(current)
            best_set = current[:]
        
        if best_len >= 48:
            break
    
    # Local search: try to improve by swapping elements
    if best_len < 48 and elapsed() < deadline - 10.0:
        improved = True
        while improved and elapsed() < deadline - 10.0:
            improved = False
            current = best_set[:]
            current_set = set(current)
            
            for i in range(len(current)):
                if elapsed() > deadline - 10.0:
                    break
                
                removed = current[i]
                temp = [x for j, x in enumerate(current) if j != i]
                temp_set = set(temp)
                
                # Try adding multiple new elements
                added = []
                for x in range(1, 513):
                    if elapsed() > deadline - 10.0:
                        break
                    if x in temp_set or x == removed:
                        continue
                    if len(added) >= 2:
                        break
                    if temp + added and not quick_pair_check(temp + added, x):
                        continue
                    if incremental_check_add(temp + added, x):
                        added.append(x)
                
                if len(added) >= 2:
                    new_set = temp + added
                    if len(new_set) > len(best_set):
                        best_set = new_set
                        best_len = len(new_set)
                        improved = True
                        break
    
    return best_set

if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(result)
