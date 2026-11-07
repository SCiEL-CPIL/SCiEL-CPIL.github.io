import sys
import time
from math import isqrt

# Set of primes to use for generating k
# We can try to include more primes if time permits, but start with small ones
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
MAX_K = 10**9
MAX_M = 10**18

def sigma(p, a):
    # Sum of divisors of p^a
    # (p^(a+1) - 1) / (p - 1)
    if a == 0:
        return 1
    return (p**(a+1) - 1) // (p - 1)

# We will use a recursive search
# state: (current_k, current_sigma, next_prime_index)
# We need to be careful with recursion depth or use iterative approach
# Given the branching factor, recursion is fine.

solutions = {}
best_M = 0
best_k_count = 0

def search(k, sig_k, idx):
    global best_M, best_k_count
    
    # If k is a valid candidate, update solutions
    # We consider k as a candidate if k > 1? No, k=1 is valid.
    # But k=1 -> M=1. sigma(1)=1. 1*1=1.
    # Let's include k=1.
    
    M = k * sig_k
    if M <= MAX_M:
        if M not in solutions:
            solutions[M] = set()
        solutions[M].add(k)
        if len(solutions[M]) > best_k_count:
            best_k_count = len(solutions[M])
            best_M = M
            
    # Try multiplying by next primes
    for i in range(idx, len(PRIMES)):
        p = PRIMES[i]
        # Check if k * p <= MAX_K
        if k > MAX_K // p:
            break
        
        new_k = k * p
        # Calculate new sigma
        # sigma(new_k) = sigma(k) * sigma(p) if gcd(k,p)=1
        # Since we are generating by increasing primes, and k is composed of primes < p (if we enforce order) 
        # Wait, if we allow reusing p, we need to handle powers.
        # Standard smooth number generation:
        # We decide the exponent of p_i.
        
        # Let's restructure:
        # Instead of just multiplying by p once, we can multiply by p^a.
        # But we need to know the current exponent of p in k.
        # Easier: pass a dictionary of exponents? Or just recurse with "next prime" logic
        # where we can choose exponent 0, 1, 2... for the current prime.
        
        # Let's use a different recursion:
        # gen(k, sig_k, prime_idx)
        # 1. Record k.
        # 2. For p = PRIMES[prime_idx]:
        #    Let p_power = 1
        #    while k * p_power <= MAX_K:
        #       new_k = k * p_power
        #       # We need sigma(new_k).
        #       # If k has no factor p, sigma(new_k) = sigma(k) * sigma(p^a)
        #       # This requires knowing if p divides k.
        #       # Since we process primes in order, and we only multiply by p here,
        #       # k is composed of primes < p. So gcd(k, p) = 1.
        #       # So sigma(new_k) = sigma(k) * sigma(p^a).
        #       
        #       gen(new_k, new_sigma, prime_idx + 1)
        #       p_power *= p
        #    break if p_power > MAX_K // k

def gen(k, sig_k, prime_idx):
    global best_M, best_k_count
    
    # Record current k
    M = k * sig_k
    if M <= MAX_M:
        if M not in solutions:
            solutions[M] = set()
        solutions[M].add(k)
        if len(solutions[M]) > best_k_count:
            best_k_count = len(solutions[M])
            best_M = M
            
    if prime_idx >= len(PRIMES):
        return
        
    p = PRIMES[prime_idx]
    p_power = 1
    # Max power of p such that k * p_power <= MAX_K
    while k * p_power <= MAX_K:
        # Calculate sigma(p^a) where p_power = p^a
        # We can compute sigma incrementally
        # Let's pass sig_p_power as well? Or compute it.
        # sigma(p^a) = 1 + p + ... + p^a
        
        # We need sigma(new_k) = sig_k * sigma(p^a)
        # Let's compute sigma(p^a)
        # We can do this in the loop
        
        # Optimization: precompute powers and sigmas for each prime up to MAX_K
        # But MAX_K is 10^9, so for p=2, max power is 29.
        # We can just calculate in the loop.
        
        # Let's restructure the loop to calculate sigma(p^a) efficiently
        # Actually, we can just call a helper or compute it.
        
        # To avoid recalculating sigma(p^a) from scratch, we can maintain it.
        # But since we break out and recurse, we can just compute it.
        
        # Let's modify the loop to track p_power and sig_p_power
        pass

# Let's rewrite gen to be cleaner
def gen_smooth(k, sig_k, prime_idx):
    global best_M, best_k_count
    
    # Record
    M = k * sig_k
    if M <= MAX_M:
        if M not in solutions:
            solutions[M] = set()
        solutions[M].add(k)
        if len(solutions[M]) > best_k_count:
            best_k_count = len(solutions[M])
            best_M = M
            
    if prime_idx >= len(PRIMES):
        return
        
    p = PRIMES[prime_idx]
    p_power = 1
    sig_p_power = 1
    
    # We can multiply by p^a for a=0,1,2...
    # a=0 is handled by the next prime (recursing with same k)
    # But we are already here.
    # Let's iterate a from 1 to max_a
    while k * p_power <= MAX_K:
        # new_k = k * p_power
        # new_sig = sig_k * sig_p_power
        
        # Recurse with next prime
        gen_smooth(k * p_power, sig_k * sig_p_power, prime_idx + 1)
        
        # Update for next power
        p_power *= p
        sig_p_power = sig_p_power * p + 1 # Wait, sigma(p^(a+1)) = sigma(p^a) * p + 1?
        # sigma(p^a) = (p^(a+1)-1)/(p-1).
        # sigma(p^(a+1)) = (p^(a+2)-1)/(p-1) = p * (p^(a+1)-1)/(p-1) + 1 = p * sigma(p^a) + 1.
        # Yes.
        
        # Check overflow for p_power
        if p_power > MAX_K // p:
            break

# Initial call: k=1, sig_k=1, prime_idx=0
# But we must ensure we don't run out of time.
# The number of smooth numbers can be large.
# Let's add a time check.

start_time = time.time()
TIME_LIMIT = 140 # Leave 10s margin

def construct():
    global best_M, best_k_count, solutions
    solutions = {}
    best_M = 1
    best_k_count = 1 # k=1, M=1
    
    # Start search
    # We might want to limit the number of primes or the depth
    # Let's try with PRIMES up to 31 first to see if it's fast enough
    # If it finishes quickly, we can expand.
    
    # For safety, let's use a smaller set of primes initially
    # Or just run with the full set and check time.
    
    # To avoid deep recursion issues, Python's default limit is 1000.
    # Depth is len(PRIMES) ~ 25. So it's fine.
    
    gen_smooth(1, 1, 0)
    
    # Find the best M
    # best_M and best_k_count are updated in gen_smooth
    
    # Return the result
    # k must be a list of ints
    # M must be an int
    # Ensure types are correct
    
    if best_M == 0:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
        
    k_list = list(solutions[best_M])
    # Sort for consistency? Not required, but nice.
    k_list.sort()
    
    return {'M': best_M, 'k': k_list}

# We need to handle the case where the search takes too long.
# Since gen_smooth is recursive and doesn't check time, it might run for a long time.
# We can add a time check in gen_smooth.

def gen_smooth_timed(k, sig_k, prime_idx):
    global best_M, best_k_count, solutions, start_time, TIME_LIMIT
    
    if time.time() - start_time > TIME_LIMIT:
        return
        
    # Record
    M = k * sig_k
    if M <= MAX_M:
        if M not in solutions:
            solutions[M] = set()
        solutions[M].add(k)
        if len(solutions[M]) > best_k_count:
            best_k_count = len(solutions[M])
            best_M = M
            
    if prime_idx >= len(PRIMES):
        return
        
    p = PRIMES[prime_idx]
    p_power = 1
    sig_p_power = 1
    
    while k * p_power <= MAX_K:
        gen_smooth_timed(k * p_power, sig_k * sig_p_power, prime_idx + 1)
        
        p_power *= p
        sig_p_power = sig_p_power * p + 1
        
        if p_power > MAX_K // p:
            break

def construct():
    global best_M, best_k_count, solutions, start_time, TIME_LIMIT
    solutions = {}
    best_M = 1
    best_k_count = 1
    start_time = time.time()
    TIME_LIMIT = 145 # 5s margin
    
    # Use a subset of primes to ensure we finish
    # Let's try primes up to 100
    gen_smooth_timed(1, 1, 0)
    
    # If we didn't find anything better than seed, return seed?
    # But we found M=1 with k=1.
    # The seed has 3 solutions.
    # We should return the best found.
    
    if best_k_count < 3:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    k_list = list(solutions[best_M])
    k_list.sort()
    
    return {'M': best_M, 'k': k_list}
