import math

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order for efficient pruning in IDDFS
    divisors.sort(reverse=True)
    N = len(divisors)
    
    # Precompute suffix sums? Or just use sum() on slices?
    # Given N is small (60), sum() on slices is fine, but precomputation is safer for speed.
    # Let's precompute prefix sums for "largest k" and just use a constant for "smallest k" from the end.
    # Actually, for the "min sum" check, we need the sum of the smallest k available.
    # Since we iterate j from i, the available set shrinks.
    # Let's stick to simple checks inside the recursion.
    
    def search(i, rem, k):
        if k == 0:
            return rem == 0, []
        if i >= N:
            return False, []
        if rem < 0:
            return False, []
        if k > N - i:
            return False, []
        
        # Pruning 1: Max possible sum with k items from i..N-1
        # These are divisors[i], divisors[i+1], ..., divisors[i+k-1]
        if sum(divisors[i:i+k]) < rem:
            return False, []
            
        # Pruning 2: Min possible sum with k items from i..N-1
        # These are the smallest k items in the range, which are divisors[N-k]..divisors[N-1]
        # This is valid because N-i >= k is already checked.
        min_sum = sum(divisors[N-k:N])
        if min_sum > rem:
            return False, []
            
        for j in range(i, N - k + 1):
            if divisors[j] > rem:
                continue
            # Optional: If divisors[j] + sum of largest k-1 from j+1 < rem, break?
            # sum(divisors[j+1 : j+1+k-1]) is max sum of remaining k-1.
            # if divisors[j] + sum(divisors[j+1:j+k]) < rem: break
            if divisors[j] + sum(divisors[j+1:j+k]) < rem:
                break
                
            success, rest = search(j + 1, rem - divisors[j], k - 1)
            if success:
                return True, [divisors[j]] + rest
                
        return False, []

    # Iterative Deepening DFS
    for k in range(1, N + 1):
        success, result = search(0, target, k)
        if success:
            return result
            
    # Fallback (should not happen)
    return []
