"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 1000000
    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # Cache for smooth numbers generation? 
    # Since we iterate B upwards, we can't easily reuse sets directly 
    # because the set of primes changes.
    # However, we can just regenerate.
    
    # To speed up, we can check if B is too small to have any candidates?
    # No, just run it.
    
    # Helper to generate B-smooth numbers <= N
    def generate_smooth(B):
        p_list = [p for p in primes if p <= B]
        if not p_list:
            return set()
        
        smooth = set([1])
        # Use a queue or list to expand
        # Start with 1
        # For each prime, multiply existing numbers
        # This is like a knapsack or subset product but with repetition allowed (powers)
        
        # Efficient way:
        # Iterate through primes.
        # Maintain a list of current smooth numbers.
        # For each prime p, for each number x in current list,
        # add x*p, x*p^2, ... as long as <= N.
        
        current_smooth = [1]
        for p in p_list:
            new_smooth = []
            # We need to add multiples of p to the existing smooth numbers
            # But we must avoid duplicates and maintain order?
            # Actually, we just want the set.
            # Let's iterate through the current list and generate powers of p multiplied by them.
            
            # To avoid modifying list while iterating, we collect new ones.
            # But we also need to combine with previously added ones?
            # Standard algorithm:
            # S = {1}
            # for p in primes:
            #   for x in S (snapshot):
            #     y = x
            #     while y * p <= N:
            #       y *= p
            #       S.add(y)
            
            # This works because any B-smooth number can be written as 
            # p1^a1 * p2^a2 ...
            # When we process p1, we generate p1^k.
            # When we process p2, we take existing (which includes 1 and p1^k) and multiply by p2^j.
            
            # We need to be careful with the "snapshot".
            # If we iterate over the set while adding, it's tricky.
            # We can use a list for the current state, then extend it.
            
            # Let's use a list `nums` that grows.
            # But `nums` contains numbers from previous primes.
            # We iterate over the *current size* of the list?
            # No, we should iterate over the numbers that were present *before* adding p's multiples,
            # AND the numbers added by p?
            # Actually, the standard way is:
            # new_nums = []
            # for x in current_smooth:
            #   y = x
            #   while y * p <= N:
            #     y *= p
            #     new_nums.append(y)
            # current_smooth.extend(new_nums)
            # 
            # Wait, this misses combinations like p1 * p2?
            # If p1 was processed first, current_smooth has 1, p1, p1^2...
            # When processing p2:
            # x=1 -> p2, p2^2...
            # x=p1 -> p1*p2, p1*p2^2...
            # x=p1^2 -> p1^2*p2...
            # This covers all combinations!
            
            # So:
            base_size = len(current_smooth)
            for i in range(base_size):
                x = current_smooth[i]
                y = x
                while y * p <= N:
                    y *= p
                    current_smooth.append(y)
                    
        return set(current_smooth)

    # Iterate through primes as potential max prime factor
    for B in primes:
        # Optimization: If B is large, the set becomes huge.
        # But we expect B to be small.
        
        smooth_set = generate_smooth(B)
        
        # Check for consecutive pairs in [10000, 999999]
        # Convert to sorted list to iterate in order? 
        # Or just iterate through the set and check n+1?
        # Iterating through set is O(K).
        # Checking n+1 in set is O(1).
        
        # We need n >= 10000.
        # We need n <= 999999 (so n+1 <= 1000000).
        
        found = False
        best_n = None
        
        # To find the *first* one (smallest n? No, we just need *a* n that minimizes LPF).
        # Since we iterate B upwards, the first B that yields *any* valid n is the minimal LPF.
        # Any n found with this B is optimal.
        
        for n in smooth_set:
            if n < 10000:
                continue
            if n > 999999:
                continue
            if (n + 1) in smooth_set:
                return n
                
    # Fallback
    return 10000

# EVOLVE-BLOCK-END
