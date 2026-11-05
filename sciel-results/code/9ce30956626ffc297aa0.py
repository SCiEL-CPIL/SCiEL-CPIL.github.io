import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 1000000
    START = 10000
    
    # Sieve for primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Powerful part array
    # Max value is N, so int32 is enough for P, but product might be 10^12, so int64 for product.
    # P[k] <= k, so int32 is safe.
    P = np.ones(N + 1, dtype=np.int64)
    
    for p in primes:
        p = int(p)
        power = p * p
        while power <= N:
            next_power = power * p
            # Get indices divisible by power
            indices = np.arange(power, N + 1, power)
            
            if next_power > N:
                # All these indices have exactly this exponent (or higher, but no higher exists in range)
                # Wait, if next_power > N, then no multiple of next_power exists in [1, N].
                # So all indices in 'indices' have v_p(k) >= e.
                # Since we are processing powers in increasing order?
                # No, we process p^2, then p^3, etc.
                # If we do p^2 first, we multiply by p^2 for all multiples of p^2.
                # Then for p^3, we multiply by p^3 for multiples of p^3?
                # NO. The definition is the product of the FULL prime power.
                # If v_p(k) = 3, the contribution is p^3, not p^2 * p^3.
                # So we must be careful.
                
                # My previous logic:
                # For e=2: multiply by p^2 if v_p(k)=2.
                # For e=3: multiply by p^3 if v_p(k)=3.
                # This works.
                
                # So if next_power > N, then for all k in indices, v_p(k) cannot be > e?
                # Actually, if next_power > N, then there are no multiples of p^(e+1) in [1, N].
                # So all k in indices have v_p(k) == e?
                # Yes, because if v_p(k) >= e+1, then k would be a multiple of p^(e+1), which is > N.
                # So we can just multiply all of them.
                P[indices] *= power
            else:
                # Filter out multiples of next_power
                mask = indices % next_power != 0
                P[indices[mask]] *= power
            
            power = next_power
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # Calculate products
    # n in [START, N-1]
    # P[n] * P[n+1]
    products = P[START:N] * P[START+1:N+1]
    
    if time.time() > deadline:
        # Fallback
        return START
        
    max_idx = np.argmax(products)
    best_n = START + int(max_idx)
    
    return best_n
