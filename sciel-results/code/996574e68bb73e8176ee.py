import math

def construct():
    limit = 1000000
    max_val = limit + 1
    
    # We need to find n in [10000, 1000000] that minimizes max(LPF(n), LPF(n+1)).
    # LPF(n) is the largest prime factor of n.
    
    # We use a sieve to compute the largest prime factor for all numbers up to max_val.
    # lpf[i] will hold the largest prime factor of i.
    # Initialization: lpf[i] = i for all i (assuming i is prime initially? No, we want the largest prime factor).
    # Better: initialize lpf[i] = 0.
    # Iterate p from 2 to max_val.
    # If lpf[p] == 0, then p is prime.
    # For all multiples j of p, set lpf[j] = p.
    # Since we iterate p in increasing order, the last prime p that divides j will be the largest prime factor.
    # Wait, if p is a factor of j, we update lpf[j] = p.
    # If we iterate p from 2 upwards, and p is prime, we update all multiples.
    # The largest prime factor of j is the largest prime p that divides j.
    # Since we go in increasing order, the last update to lpf[j] will come from the largest prime factor.
    # Example: j=12. Primes dividing 12: 2, 3.
    # p=2: lpf[12]=2.
    # p=3: lpf[12]=3.
    # p=4: not prime.
    # p=5: not factor.
    # Final lpf[12]=3. Correct.
    
    lpf = [0] * (max_val + 1)
    
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            # p is prime
            for j in range(p, max_val + 1, p):
                lpf[j] = p
                
    best_n = 10000
    min_score = float('inf')
    
    # Scan for the minimum max(LPF(n), LPF(n+1))
    for n in range(10000, limit + 1):
        score = lpf[n]
        next_score = lpf[n+1]
        if next_score > score:
            score = next_score
        if score < min_score:
            min_score = score
            best_n = n
            
    return best_n
