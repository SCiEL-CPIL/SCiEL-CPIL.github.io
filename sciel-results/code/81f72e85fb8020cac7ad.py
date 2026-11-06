import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin out of 60s
    
    N = math.factorial(12)
    
    # Generate all divisors of N
    divisors = []
    for i in range(1, int(math.isqrt(N)) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:
                divisors.append(N // i)
    divisors.sort()
    
    # Filter divisors > 12
    valid_divs = [d for d in divisors if d > 12]
    
    if not valid_divs:
        return [N]
        
    # We want to find the smallest M such that N can be decomposed 
    # into distinct factors from valid_divs that are <= M.
    
    def find_decomp_for_M(M):
        # Returns a list of factors if found, else None.
        # Factors must be distinct, > 12, <= M, product N.
        
        # Find index of last divisor <= M
        # Since valid_divs is sorted, we can use bisect
        import bisect
        idx_end = bisect.bisect_right(valid_divs, M)
        if idx_end == 0:
            return None
            
        allowed = valid_divs[:idx_end]
        
        if not allowed:
            return None
        
        # Precompute suffix products
        # suffix_prods[i] = product of allowed[i:]
        suffix_prods = [1] * (len(allowed) + 1)
        for i in range(len(allowed) - 1, -1, -1):
            suffix_prods[i] = suffix_prods[i+1] * allowed[i]
            
        # If total product of all allowed < N, impossible
        if suffix_prods[0] < N:
            return None

        def backtrack(remaining, idx):
            if time.time() > deadline:
                return None
            
            if remaining == 1:
                return []
            
            if idx >= len(allowed):
                return None
            
            # Prune: if the smallest available factor is larger than remaining,
            # we can't pick any more factors.
            if allowed[idx] > remaining:
                return None
                
            # Prune: if product of all remaining allowed factors < remaining
            if suffix_prods[idx] < remaining:
                return None
                
            # Option 1: Take allowed[idx]
            if remaining % allowed[idx] == 0:
                res = backtrack(remaining // allowed[idx], idx + 1)
                if res is not None:
                    return [allowed[idx]] + res
            
            # Option 2: Skip allowed[idx]
            res = backtrack(remaining, idx + 1)
            if res is not None:
                return res
            
            return None
        
        res = backtrack(N, 0)
        if res:
            return res
        return None

    # Binary search for the minimum M
    # The answer M must be one of the valid divisors.
    # We search for the smallest index in valid_divs such that find_decomp_for_M(valid_divs[idx]) is not None.
    
    lo = 0
    hi = len(valid_divs) - 1
    best_res = None
    best_M = None
    
    # Check if the largest M works (sanity check / fallback)
    # If even the largest M doesn't work, we return [N]
    # But [N] is always a valid decomposition (1 factor, N > 12, distinct).
    # Wait, the problem says "1..12 DISTINCT integers".
    # [N] is 1 integer. It is distinct. It is > 12. Product is N.
    # So [N] is always a valid solution.
    # We want to minimize the LARGEST factor.
    # So we are looking for the minimum possible max factor.
    
    while lo <= hi:
        if time.time() > deadline:
            break
            
        mid = (lo + hi) // 2
        M_candidate = valid_divs[mid]
        
        res = find_decomp_for_M(M_candidate)
        
        if res is not None:
            best_res = res
            best_M = M_candidate
            hi = mid - 1  # Try to find a smaller M
        else:
            lo = mid + 1   # M_candidate is too small, try larger
            
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
