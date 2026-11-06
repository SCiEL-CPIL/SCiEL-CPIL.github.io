import time
import numpy as np

def construct():
    t0 = time.time()
    DEADLINE = 80  # Safety margin for 90s limit
    
    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000
    N = MAXN + 1 # We need pp[n+1] where n can be MAXN, so index MAXN+1

    # 1. Sieve of Eratosthenes to get primes up to sqrt(MAXN)
    # Primes p > sqrt(MAXN) cannot have p^2 <= MAXN, so they contribute 1 to powerful part.
    limit = int(MAXN**0.5) + 1
    is_prime = np.ones(limit, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i in range(2, limit) if is_prime[i]]

    # 2. Compute powerful part P(m) for all m in [0, MAXN+1]
    # P(m) = product of p^k where k = v_p(m) and k >= 2
    pp = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        if elapsed() > DEADLINE:
            break
            
        p_power = p * p
        k = 2
        
        while p_power <= MAXN:
            # Indices divisible by p_power
            # We want to multiply pp[m] by p_power for m where v_p(m) == k
            # These are multiples of p_power but NOT multiples of p_power * p
            
            # Generate indices: 0, p_power, 2*p_power, ...
            # We can use slicing on the pp array directly?
            # pp[0::p_power] gives the slice.
            # But we need to exclude every p-th one in this slice.
            
            # Let's get the slice
            # The slice length is roughly N // p_power
            # We want to update elements at indices j where j % p != 0
            # in the sequence of multiples.
            
            # Efficient way:
            # mask = np.arange(0, N+1, p_power)
            # valid_mask = (np.arange(len(mask)) % p) != 0
            # pp[mask[valid_mask]] *= p_power
            
            # Even more efficient:
            # We can iterate over the slice in steps?
            # No, slicing is C-speed.
            
            # Let's try:
            # The indices are i * p_power for i = 0, 1, ...
            # We want i % p != 0.
            
            # We can construct the indices directly.
            # count = (N // p_power) + 1
            # indices = np.arange(0, N+1, p_power)
            # valid_indices = indices[(np.arange(len(indices)) % p) != 0]
            # pp[valid_indices] *= p_power
            
            # Optimization:
            # Instead of creating arange for mask every time, we can just do:
            # pp[p_power::p_power] is the slice.
            # We want to exclude every p-th element in this slice.
            # slice_len = len(pp[p_power::p_power])
            # We can use boolean indexing on the slice?
            # No, boolean indexing on a view returns a copy if we assign?
            # pp[p_power::p_power][mask] *= p_power might not update the original array if it's a view?
            # Actually, advanced indexing (which boolean indexing is) returns a copy.
            # So we MUST use advanced indexing with the original array indices.
            
            # So:
            indices = np.arange(p_power, N + 1, p_power) # Start from p_power, not 0, as 0 doesn't matter and 0%anything is 0 (excluded)
            # Wait, 0 is a multiple of p_power. v_p(0) is undefined/infinite, but we don't care about m=0.
            # The task asks for n in [10000, 1000000].
            
            # Let's stick to the full range to be safe, or just 10000 to N?
            # Precomputation for full range is fine.
            
            # indices = np.arange(0, N + 1, p_power)
            # mask = (np.arange(len(indices)) % p) != 0
            # valid_indices = indices[mask]
            # pp[valid_indices] *= p_power
            
            # This creates a few temporary arrays. For N=1e6, this is fast.
            
            # Alternative without arange for mask:
            # We can step through the slice with a step of p, and exclude those?
            # No, we want to INCLUDE those not divisible by p.
            
            # Let's use the arange method, it's clean.
            # Note: p_power can be large, so the array is small.
            
            # Optimization: If p_power > N, loop breaks.
            
            # One detail: The problem asks for n in [10000, 1000000].
            # We only need pp for 10000..1000001.
            # But computing for 0..1000001 is negligible overhead.
            
            # Let's refine the indices generation to be faster if possible.
            # For small p_power, the array is large.
            
            # pp[p_power::p_power] is a view.
            # We want to update elements at positions 1, 2, ..., p-1, p+1, ... in this view?
            # No, position 0 in the view is index p_power.
            # Position j in the view is index p_power * (1+j) if we start from p_power?
            # No, np.arange(0, N+1, p_power) starts with 0.
            # View pp[0::p_power] starts with pp[0].
            
            # Let's just use the explicit indices array.
            
            idx = np.arange(0, N + 1, p_power)
            # We want idx[i] where i % p != 0
            # i ranges from 0 to len(idx)-1
            # mask = (np.arange(len(idx)) % p) != 0
            # pp[idx[mask]] *= p_power
            
            # This is O(N/p_power).
            
            # Can we do it without creating idx and mask?
            # We can use slicing on pp?
            # pp[0::p_power] is the full set.
            # We want to exclude every p-th one.
            # This is hard to do with simple slicing.
            
            # Let's stick to idx and mask. It's safe and fast enough.
            
            # Small optimization:
            # If p_power is very small (e.g. 4), idx is large (~250k).
            # arange and mask are fast.
            
            # Let's try to optimize the inner loop by avoiding Python overhead in the while loop?
            # The while loop runs ~19 times for p=2, ~12 for p=3, etc.
            # Total iterations is small.
            
            # Execute update
            idx = np.arange(0, N + 1, p_power)
            mask = (np.arange(len(idx)) % p) != 0
            pp[idx[mask]] *= p_power

            p_power *= p
            k += 1

    if elapsed() > DEADLINE:
        # Fallback: return a reasonable guess if we ran out of time
        # But precomputation should take < 1s.
        return 10000

    # 3. Find max pp[n] * pp[n+1] for n in [10000, 1000000]
    # n goes up to 1000000, so n+1 goes up to 1000001.
    # We need pp[10000] to pp[1000001].
    
    # Vectorized product
    # pp_n = pp[10000:1000001]
    # pp_n1 = pp[10001:1000002]
    # prod = pp_n * pp_n1
    # max_idx = np.argmax(prod)
    # best_n = 10000 + max_idx
    
    # Note: pp is int64. Product might overflow?
    # Max powerful part is <= 1000000.
    # 1000000 * 1000000 = 10^12.
    # int64 max is ~9 * 10^18. So no overflow.
    
    pp_n = pp[10000:1000001]
    pp_n1 = pp[10001:1000002]
    prod = pp_n * pp_n1
    
    max_idx = np.argmax(prod)
    best_n = 10000 + max_idx
    
    return int(best_n)
