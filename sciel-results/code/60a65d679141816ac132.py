import math
import time

def construct():
    N = 123457
    start_time = time.time()
    
    # is_smooth[i] is True if i is smooth with respect to primes processed so far
    is_smooth = [False] * (N + 1)
    is_smooth[1] = True
    
    # List of smooth numbers to iterate over for generation
    # Actually, we can just generate new numbers from the existing smooth numbers
    # But we need to know which numbers are in S.
    # Let's keep a list of current smooth numbers.
    smooth_nums = [1]
    
    # Simple prime generator
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        for i in range(2, limit + 1):
            if sieve[i]:
                yield i

    # We don't know the limit, but N is 123457.
    # We can generate primes up to N.
    primes = list(gen_primes(N))
    
    for p in primes:
        # Add numbers formed by multiplying existing smooth numbers by powers of p
        new_smooths = []
        # We need to iterate over a copy or track indices carefully
        # Because smooth_nums is growing
        # However, new numbers are formed from OLD smooth numbers * p^k
        # So we can iterate over the current length of smooth_nums
        
        current_len = len(smooth_nums)
        for i in range(current_len):
            x = smooth_nums[i]
            val = x * p
            while val <= N:
                if not is_smooth[val]:
                    is_smooth[val] = True
                    smooth_nums.append(val)
                    new_smooths.append(val)
                val *= p
        
        # Check for solution
        # Iterate a from 1 to N//2
        # If is_smooth[a] and is_smooth[N-a], we found it
        # Optimization: We only need to check if ANY a works.
        # We can just scan.
        found = False
        for a in range(1, N // 2 + 1):
            if is_smooth[a] and is_smooth[N - a]:
                return [a, N - a]
        
        # Time check
        if time.time() - start_time > 80:
             # Fallback? 
             # If we haven't found it, we might have a bug or it's very hard.
             # But smooth numbers should cover it quickly.
             # Return best effort?
             # For now, just break or continue.
             pass

    # Fallback
    return [1, N - 1]
