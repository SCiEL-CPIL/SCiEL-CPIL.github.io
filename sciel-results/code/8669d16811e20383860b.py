import math
import time

def construct():
    deadline = time.time() + 145.0
    
    # Sieve limit for candidate primes p
    # 10^8 is about 100MB memory.
    # Let's try 50,000,000 to be safe on memory limits (usually 256MB-1GB).
    # 50MB is safe.
    limit = 50_000_000
    
    # Generate primes up to limit
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    best_k = 0
    best_p = 2
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Check p-smoothness for p^2+1 ... p^2+200
        start_val = p * p + 1
        rem = [start_val + i for i in range(200)]
        
        # Sieve with primes q <= p
        # We iterate through the global primes list
        for q in primes:
            if q > p:
                break
            
            # Find first multiple of q in [start_val, start_val + 199]
            # m = ceil(start_val / q) * q
            m = ((start_val + q - 1) // q) * q
            i = m - start_val
            
            if i >= 200:
                continue
                
            while i < 200:
                # Divide out all factors of q
                # rem[i] is guaranteed to be divisible by q at the start?
                # No, rem[i] has been modified by previous primes.
                # But rem[i] is a factor of the original number.
                # If original number was divisible by q, and we haven't removed q yet,
                # rem[i] should still be divisible by q?
                # Yes, because we only divide by primes < q (since we iterate in order).
                # So if q divides the original number, q must divide the current rem[i].
                # Wait.
                # Original N = 2 * 3 * 5.
                # q=2. rem becomes 3*5=15.
                # q=3. 15 is divisible by 3. rem becomes 5.
                # q=5. 5 is divisible by 5. rem becomes 1.
                # So yes, if q divides N, and we process primes in increasing order,
                # q will divide rem[i] when we reach q, UNLESS q was already divided out?
                # No, we only divide by q itself.
                # So we can skip the `if rem[i] % q == 0` check?
                # Let's verify.
                # If N is not divisible by q, we skip.
                # If N is divisible by q, then N = q^k * M.
                # We divide by all smaller primes. M is not divisible by any prime < q.
                # So rem[i] = q^k * (M / (primes < q part of M? No, M is coprime to primes < q)).
                # Actually, rem[i] is N divided by all prime factors < q.
                # So rem[i] is divisible by q if and only if N is divisible by q.
                # So we can just do the division loop without the modulo check?
                # while rem[i] % q == 0: rem[i] //= q
                # The first check `rem[i] % q == 0` is necessary to enter the loop.
                # But we can combine it.
                
                # Optimization:
                # if rem[i] % q == 0:
                #     while rem[i] % q == 0:
                #         rem[i] //= q
                
                # This is what I had. It's fine.
                
                if rem[i] % q == 0:
                    while rem[i] % q == 0:
                        rem[i] //= q
                i += q
        
        # Count leading 1s
        k = 0
        for i in range(200):
            if rem[i] == 1:
                k = i + 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
