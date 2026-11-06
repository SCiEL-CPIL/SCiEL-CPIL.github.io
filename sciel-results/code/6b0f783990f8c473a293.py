import time
import math
import random

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    best_result = {'start': 1, 'length': 1}
    
    # Generate primes up to 500
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(500)
    
    def compute_phi(n):
        """Compute Euler's totient function for n."""
        if n < 2:
            return 1
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    def verify_run(s, k):
        """Verify that phi(s), phi(s+1), ..., phi(s+k-1) are all distinct."""
        seen = set()
        for i in range(k):
            v = compute_phi(s + i)
            if v in seen:
                return False
            seen.add(v)
        return True

    def crt_solve(residues, moduli):
        """Solve system of congruences x ≡ residues[i] (mod moduli[i]) using CRT."""
        M = 1
        for m in moduli:
            M *= m
        x = 0
        for r, m in zip(residues, moduli):
            Mi = M // m
            # Find inverse of Mi mod m
            inv = pow(Mi, -1, m)
            x += r * Mi * inv
        return x % M

    # Strategy: Try to construct runs of length k using CRT
    # We assign distinct primes to each position
    # We want the product of primes to be not too large, so s is likely small
    
    # Try k from 20 down to 5
    for k in range(20, 4, -1):
        if time.time() > deadline - 10:
            break
        
        # Use the first k primes
        # This minimizes the product M
        selected_primes = primes[:k]
        M = 1
        for p in selected_primes:
            M *= p
        
        # If M is too large, the probability of s < 10^12 is low
        # But we can try a few permutations
        # If M > 10^14, skip or try fewer permutations
        max_perms = 1000 if M < 10**13 else 100 if M < 10**15 else 10
        
        # Try random permutations of prime assignments
        for _ in range(max_perms):
            if time.time() > deadline - 5:
                break
                
            permuted_primes = selected_primes[:]
            random.shuffle(permuted_primes)
            
            residues = []
            moduli = []
            for i in range(k):
                residues.append(-i)  # s + i ≡ 0 (mod permuted_primes[i])
                moduli.append(permuted_primes[i])
            
            try:
                s = crt_solve(residues, moduli)
            except:
                continue
            
            # Ensure s >= 1
            if s < 1:
                s += M
            
            # Check bounds
            if s + k - 1 > 10**12:
                continue
            
            # Verify the run
            if verify_run(s, k):
                best_result = {'start': s, 'length': k}
                break
        
        if best_result['length'] == k:
            break
    
    # Fallback: Search in small ranges
    if best_result['length'] < 5:
        # Search up to 10^6
        limit = 10**6
        for s in range(1, limit - 200):
            if time.time() > deadline - 5:
                break
            seen = set()
            length = 0
            for i in range(200):
                v = compute_phi(s + i)
                if v in seen:
                    break
                seen.add(v)
                length += 1
            if length > best_result['length']:
                best_result = {'start': s, 'length': length}
                if length >= 20:
                    break

    return best_result
