import sys

def construct():
    min_n = 10000
    max_n_plus_1 = 1000001
    
    # Generate primes up to a limit sufficient to cover the optimal P.
    # The optimal P is the minimum largest prime factor.
    # For n in [10^4, 10^6], P is likely small.
    # We'll generate primes up to 10000.
    
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(10000)
    
    for P in primes:
        # Generate all P-smooth numbers in [min_n, max_n_plus_1]
        # A number is P-smooth if all its prime factors are <= P.
        
        # Start with 1
        smooth_list = [1]
        
        # Get primes <= P
        allowed_primes = [p for p in primes if p <= P]
        
        for p in allowed_primes:
            new_items = []
            for s in smooth_list:
                val = s
                while val * p <= max_n_plus_1:
                    val *= p
                    if val >= min_n:
                        new_items.append(val)
            smooth_list.extend(new_items)
            
        # Remove duplicates
        smooth_set = set(smooth_list)
        
        # Check for consecutive integers
        sorted_smooths = sorted(smooth_set)
        for i in range(len(sorted_smooths) - 1):
            if sorted_smooths[i+1] - sorted_smooths[i] == 1:
                return sorted_smooths[i]
                
    # Fallback if not found (should not happen)
    return 10000

if __name__ == "__main__":
    print(construct())
