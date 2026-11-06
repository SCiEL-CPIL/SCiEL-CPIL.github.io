import math
from math import gcd

def construct():
    N = 1000000
    START = 10000
    
    # 1. Generate all powerful numbers <= N
    # A powerful number is a product of prime powers p^e with e >= 2.
    # We can generate them by iterating through primes and combinations.
    
    primes = []
    is_prime = [True] * (int(math.isqrt(N)) + 2)
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, int(math.isqrt(N)) + 1, i):
                is_prime[j] = False
    # Add primes > sqrt(N) if they form a powerful number themselves?
    # p^2 <= N => p <= sqrt(N). So all prime bases are <= sqrt(N).
    
    powerful_numbers = []
    
    def gen_powerful(idx, current_val):
        if current_val > N:
            return
        if idx == len(primes):
            if current_val > 1:
                powerful_numbers.append(current_val)
            return
        
        p = primes[idx]
        # Option 1: skip this prime
        gen_powerful(idx + 1, current_val)
        
        # Option 2: include p^e for e >= 2
        power = p * p
        while power <= N:
            gen_powerful(idx + 1, current_val * power)
            power *= p
            
    gen_powerful(0, 1)
    
    # Remove 1 if present (it's not useful for maximization usually, but P(n) can be 1)
    # Actually, if n is squarefree, P(n)=1.
    # We should include 1 in the list?
    # If we use W1=1, then n is not constrained to be divisible by any powerful number > 1.
    # But we want to maximize P(n)*P(n+1).
    # If P(n)=1, it's small.
    # We can just add 1 to the list.
    if 1 not in powerful_numbers:
        powerful_numbers.append(1)
        
    # Sort descending
    powerful_numbers.sort(reverse=True)
    
    # 2. Helper to compute Powerful Part
    def get_powerful_part(x):
        if x < 2:
            return 1
        temp = x
        p_part = 1
        # Simple factorization
        # We can use the primes list generated above, but it only goes up to sqrt(N).
        # If x has a prime factor > sqrt(N), it can only appear with exponent 1.
        # So it won't contribute to the powerful part.
        # So we only need to check primes up to sqrt(x).
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                if exp >= 2:
                    p_part *= (p ** exp)
        # If temp > 1, it is a prime with exponent 1, so it doesn't contribute.
        return p_part

    # 3. Search for best n
    max_val = 0
    best_n = START
    
    # We iterate through pairs of powerful numbers (W1, W2)
    # W1 corresponds to P(n) core, W2 to P(n+1) core.
    # We want W1 * W2 to be large.
    # n is multiple of W1, n+1 is multiple of W2.
    # gcd(W1, W2) must be 1.
    
    L = len(powerful_numbers)
    
    for i in range(L):
        W1 = powerful_numbers[i]
        # Pruning: even if W2 is max possible, can we beat max_val?
        # Max possible W2 is powerful_numbers[0]
        if W1 * powerful_numbers[0] <= max_val:
            break
            
        for j in range(L):
            W2 = powerful_numbers[j]
            
            # Pruning
            if W1 * W2 <= max_val:
                break
                
            if gcd(W1, W2) != 1:
                continue
                
            # Solve:
            # n = k * W1
            # k * W1 + 1 = m * W2
            # k * W1 = m * W2 - 1
            # k * W1 = -1 (mod W2)
            # k = -1 * W1^(-1) (mod W2)
            
            try:
                # Python 3.8+ has pow(a, -1, m)
                inv = pow(W1, -1, W2)
            except ValueError:
                # Should not happen since gcd is 1
                continue
                
            k0 = (-inv) % W2
            # k = k0 + t * W2
            # n = (k0 + t * W2) * W1
            # n = k0 * W1 + t * W1 * W2
            
            base_n = k0 * W1
            step = W1 * W2
            
            # Find t such that n in [START, N]
            # base_n + t * step >= START
            # base_n + t * step <= N
            
            if base_n > N:
                # Check if we can go backwards? No, k0 is smallest non-negative.
                # But k can be negative?
                # n must be positive.
                # If base_n > N, no solution for this pair in range?
                # Wait, k0 is in [0, W2).
                # n = base_n + t * step.
                # If base_n > N, maybe t can be negative?
                # n must be >= START > 0.
                # If base_n is large, subtracting step might bring it down.
                # Let's find the smallest t such that n >= START.
                if step > 0:
                    t_min = math.ceil((START - base_n) / step)
                    if t_min < 0:
                        t_min = 0 # We want the smallest n >= START, but also n > 0.
                        # Actually if t_min < 0, it means base_n > START.
                        # We can try t = 0, -1, ... until n < START?
                        # No, we want n in [START, N].
                        # If base_n > N, we need to subtract steps.
                        # t = floor((N - base_n) / step)
                        t_max = math.floor((N - base_n) / step)
                        if t_max >= 0:
                            # This implies base_n <= N
                            pass
                        else:
                            # base_n > N
                            # We need to find t such that base_n + t*step <= N
                            # t <= (N - base_n) / step
                            # Since base_n > N, this is negative.
                            # Let's just find the largest n <= N.
                            t = math.floor((N - base_n) / step)
                            n_candidate = base_n + t * step
                            if n_candidate >= START:
                                p1 = get_powerful_part(n_candidate)
                                p2 = get_powerful_part(n_candidate + 1)
                                val = p1 * p2
                                if val > max_val:
                                    max_val = val
                                    best_n = n_candidate
                    else:
                        # base_n <= START
                        # Find smallest t such that n >= START
                        t = t_min
                        n_candidate = base_n + t * step
                        if n_candidate <= N:
                            p1 = get_powerful_part(n_candidate)
                            p2 = get_powerful_part(n_candidate + 1)
                            val = p1 * p2
                            if val > max_val:
                                max_val = val
                                best_n = n_candidate
                        # Check next t?
                        # Since step is large (product of powerful numbers),
                        # usually only one or few candidates per pair.
                        # We should check all t in range.
                        while n_candidate <= N:
                            p1 = get_powerful_part(n_candidate)
                            p2 = get_powerful_part(n_candidate + 1)
                            val = p1 * p2
                            if val > max_val:
                                max_val = val
                                best_n = n_candidate
                            n_candidate += step
                else:
                    continue
                    
    # Fallback: if no candidate found or max_val is 0, do a small scan?
    # The current program scans everything. This one is targeted.
    # If this fails to find a good one, it might be worse.
    # But powerful numbers are frequent.
    
    # To be safe, if max_val is very low, we might want to ensure we didn't miss something.
    # But the problem asks to maximize. The seed program scans all.
    # This program is an improvement in *approach*.
    
    return best_n
