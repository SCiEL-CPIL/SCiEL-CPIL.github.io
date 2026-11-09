import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 123457
    
    # List of small primes to test as the maximum prime factor P
    # The optimal P is likely small. We'll start from 2 and go up.
    # If we don't find a solution with small primes, we can increase the limit.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Function to generate all p-smooth numbers <= limit
    # A number is p-smooth if its largest prime factor is <= p
    def generate_smooth_numbers(primes_list, limit):
        smooth = {1}
        # We use a set to avoid duplicates
        # We can generate by multiplying existing smooth numbers by primes
        # However, a more controlled DFS is safer to avoid infinite loops or excessive memory
        # Let's use a recursive generator or a priority queue (Dijkstra-like)
        
        # Simpler approach for small N:
        # Start with 1.
        # For each prime in primes_list, multiply existing numbers by powers of prime.
        # This is tricky to do iteratively without duplicates.
        
        # Better approach: DFS
        def dfs(index, current_val):
            if time.time() - start_time > DEADLINE:
                return
            if current_val > limit:
                return
            smooth.add(current_val)
            if index == len(primes_list):
                return
            p = primes_list[index]
            # Multiply by p as many times as possible
            val = current_val
            while val * p <= limit:
                val *= p
                dfs(index, val) # Continue with same prime
                # Note: This allows powers of p
            # Move to next prime
            dfs(index + 1, current_val)
            
        dfs(0, 1)
        return smooth

    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate all numbers whose prime factors are a subset of primes <= p
        # Actually, we just need to generate numbers composed of primes in the list [2, ..., p]
        current_primes = primes[:primes.index(p) + 1]
        
        smooth_set = generate_smooth_numbers(current_primes, N)
        
        # Check if there is a pair a, b in smooth_set such that a + b = N
        found = False
        for a in smooth_set:
            b = N - a
            if b in smooth_set:
                # Ensure a, b are positive
                if a > 0 and b > 0:
                    return [a, b]
        
        # If no pair found with max prime factor p, try next prime
    
    # Fallback: If we didn't find a small smooth pair, we might need a larger prime.
    # Given N is small, we can fall back to a full sieve if needed, 
    # but the smooth number approach should find the optimal quickly.
    # Let's do a quick scan for the absolute best if the loop above didn't return.
    # This part is unlikely to be reached for N=123457 with p up to 97.
    
    # Fallback implementation: Sieve LPF and scan
    # This is O(N log log N) + O(N)
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            if best_max_pf == 2: # Theoretical minimum for N>2, but N is odd so one is 1? P(1)=1.
                break
                
    return [best_a, best_b]
