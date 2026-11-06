import math
import time
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = math.factorial(12)
    
    # Generate divisors
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.sort()
    
    # Filter > 12
    valid_divs = [d for d in divisors if d > 12]
    
    if not valid_divs:
        return [N]
        
    # We know k <= 8
    MAX_K = 8
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    def find_decomp_for_M(M):
        idx_end = bisect.bisect_right(valid_divs, M)
        if idx_end == 0:
            return None
            
        allowed = valid_divs[:idx_end]
        if not allowed:
            return None
            
        # Memoization: (remaining, idx, count) -> bool
        # But remaining can be large. 
        # Actually, we can just use a set of visited states to avoid re-computation in this DFS?
        # The DFS branches only on divisors of remaining.
        # Let's use a simple recursive function with pruning.
        
        # To speed up, we can precompute for each possible 'remaining' value? No, too many.
        # But the number of divisors of N is small (792).
        # 'remaining' will always be a divisor of N.
        # So we can map divisors to indices.
        
        # Let's just use recursion with lru_cache on (remaining, idx, count).
        # idx is index in 'allowed'.
        # Since 'allowed' changes with M, we can't cache across M calls easily unless we include M or the tuple of allowed.
        # But we are doing binary search on M.
        # The number of calls to find_decomp_for_M is ~log(792) ~ 10.
        # So we can define the cache inside find_decomp_for_M.
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(remaining, idx, count):
            if time.time() > deadline:
                return None
                
            if remaining == 1:
                return []
                
            if count >= MAX_K:
                return None
                
            if idx >= len(allowed):
                return None
                
            # Pruning: 
            # The smallest factor we can pick is allowed[idx].
            # If allowed[idx] > remaining, we can't pick anything.
            if allowed[idx] > remaining:
                return None
                
            # If remaining is prime:
            # It must be equal to some allowed[j] for j >= idx, and it must be the last factor.
            # Also, we must be able to pick it.
            if is_prime(remaining):
                # Check if remaining is in allowed[idx:]
                # Since allowed is sorted, we can bisect.
                pos = bisect.bisect_left(allowed, remaining, idx)
                if pos < len(allowed) and allowed[pos] == remaining:
                    return [remaining]
                return None
                
            # Try picking allowed[j] for j from idx to len(allowed)-1
            # such that allowed[j] divides remaining.
            # Optimization: We can break if allowed[j] > remaining.
            for j in range(idx, len(allowed)):
                d = allowed[j]
                if d > remaining:
                    break
                if remaining % d == 0:
                    res = dfs(remaining // d, j + 1, count + 1)
                    if res is not None:
                        return [d] + res
                        
            return None
            
        res = dfs(N, 0, 0)
        if res:
            return res
        return None

    # Binary search for M
    lo = 0
    hi = len(valid_divs) - 1
    best_res = None
    best_M = None
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        M_candidate = valid_divs[mid]
        
        res = find_decomp_for_M(M_candidate)
        
        if res is not None:
            best_res = res
            best_M = M_candidate
            hi = mid - 1
        else:
            lo = mid + 1
            
    if best_res is not None:
        return best_res
    else:
        return [N]

if __name__ == "__main__":
    res = construct()
    print(res)
    prod = 1
    for x in res:
        prod *= x
    print("Product:", prod)
    print("12!:", math.factorial(12))
    print("Max:", max(res))
    print("Distinct:", len(res) == len(set(res)))
    print("All > 12:", all(x > 12 for x in res))
    print("Count:", len(res))
