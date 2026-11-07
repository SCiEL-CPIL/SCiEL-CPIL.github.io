"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Sieve of Eratosthenes to find primes up to 256
    n = 256
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # The set of all primes <= 256 is a valid solution.
    # Condition: for distinct a, b, c in S, a does not divide b*c.
    # If a, b, c are distinct primes, a|bc implies a|b or a|c, which is false.
    # Thus, this set is valid.
    # Size is pi(256) = 54.
    
    # Verify validity just in case (though logically sound)
    # This check is O(N^3) which is 54^3 ~ 150k ops, very fast.
    S = primes
    valid = True
    for i in range(len(S)):
        if time.time() > deadline:
            break
        for j in range(len(S)):
            if i == j: continue
            for k in range(len(S)):
                if i == k or j == k: continue
                if S[j] * S[k] % S[i] == 0:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if valid:
        return S
    else:
        # Fallback: return empty list or a known small valid set if logic fails (shouldn't happen)
        return [2, 3, 5]

# EVOLVE-BLOCK-END
