import math

def construct():
    # Primes congruent to 1 mod 4
    primes = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257]
    
    # We want to maximize product of (e_i + 1) subject to sum(e_i * log(p_i)) <= log(10^18)
    # Greedy approach: Try to assign exponents to smallest primes first.
    # Since (e+1) grows linearly with e but cost is e*log(p), small p is better.
    # However, diminishing returns? No, (e+1)/e increases for small e.
    # Let's just try a few configurations or a simple greedy.
    
    max_log = math.log(10**18)
    
    # Start with e_i = 1 for all, then upgrade small primes if possible.
    # Or simply iterate.
    
    # Let's use a recursive or iterative search for exponents.
    # Given the small number of primes (max ~20), we can be smart.
    
    # Strategy: 
    # 1. Start with all e_i = 0.
    # 2. Repeatedly increment e_i for the prime i that gives the best "gain per cost" 
    #    or simply increment the smallest prime available if it fits.
    #    Actually, (e+1) / log(p) is the metric?
    #    Let's just try to fit as many 1s as possible, then upgrade.
    
    exponents = [0] * len(primes)
    current_log = 0.0
    
    # Phase 1: Assign e_i = 1 to smallest primes until we can't.
    for i in range(len(primes)):
        if current_log + math.log(primes[i]) <= max_log:
            exponents[i] = 1
            current_log += math.log(primes[i])
        else:
            break
    
    # Phase 2: Try to upgrade exponents of smaller primes if we have room.
    # Check if we can increase e_i for any i.
    # Priority: Smallest primes first.
    for i in range(len(primes)):
        while current_log + math.log(primes[i]) <= max_log:
            exponents[i] += 1
            current_log += math.log(primes[i])
            
    # Construct N
    N = 1
    for i, e in enumerate(exponents):
        if e > 0:
            N *= primes[i] ** e
            
    # Generate pairs (x, y) such that x^2 + y^2 = N
    # Use the factorization in Gaussian integers.
    # For each p = a^2 + b^2, p = (a+bi)(a-bi).
    # N = prod p_i^{e_i}.
    # We can generate all Gaussian divisors.
    
    # Find a, b for each prime p = a^2 + b^2
    def sum_of_two_squares(p):
        # p is prime 1 mod 4
        # Use Tonelli-Shanks or just brute force for small p
        for x in range(1, int(math.isqrt(p)) + 1):
            y2 = p - x*x
            y = math.isqrt(y2)
            if y*y == y2:
                return (x, y)
        return None

    # Precompute a, b for each prime used
    ab_list = []
    for i, e in enumerate(exponents):
        if e > 0:
            ab = sum_of_two_squares(primes[i])
            # Store (a, b, e)
            ab_list.append((ab[0], ab[1], e))
            
    # Generate all pairs (x, y)
    # This is equivalent to finding all Gaussian integers Z such that |Z|^2 = N.
    # Z = prod (a_i + b_i i)^{k_i} (a_i - b_i i)^{e_i - k_i}
    # where 0 <= k_i <= e_i.
    # We can iterate through all combinations of k_i.
    
    pairs = []
    
    # Recursive generation
    def generate(idx, current_z):
        if idx == len(ab_list):
            # current_z is (real, imag)
            x = abs(current_z[0])
            y = abs(current_z[1])
            # Ensure x <= y for distinct pairs, and x,y > 0
            if x == 0 or y == 0:
                return
            if x > y:
                x, y = y, x
            # Check if this pair is already in list? 
            # The generation might produce duplicates due to symmetries?
            # No, each unique combination of exponents in Gaussian factorization 
            # gives a unique Gaussian integer up to unit multiplication.
            # We take abs, so we might get duplicates if (x,y) and (y,x) are generated separately?
            # We enforce x <= y.
            # Do we get duplicates from different factorizations?
            # No, unique factorization in Z[i].
            # However, different k_i choices might lead to same (x,y) after abs?
            # Example: Z and conjugate(Z) have same abs.
            # We should use a set to be safe, or just append and deduplicate later.
            pairs.append((x, y))
            return
            
        a, b, e = ab_list[idx]
        # We need to multiply current_z by (a+bi)^k (a-bi)^(e-k)
        # It's easier to precompute the powers.
        
        # Precompute powers of (a+bi) and (a-bi)
        # Or just iterate k from 0 to e
        
        for k in range(e + 1):
            # Multiply by (a+bi)^k * (a-bi)^(e-k)
            # This is equivalent to multiplying by (a+bi)^e * ((a-bi)/(a+bi))^(e-k)
            # Or just calculate the term.
            
            # Let's compute term = (a+bi)^k * (a-bi)^(e-k)
            # We can update current_z incrementally?
            # No, just compute the factor.
            
            # Optimization: Precompute all factors for this prime
            # factor[k] = (a+bi)^k * (a-bi)^(e-k)
            
            # Calculate factor
            real_f, imag_f = 1, 0
            # Multiply by (a+bi) k times
            for _ in range(k):
                real_f, imag_f = real_f*a - imag_f*b, real_f*b + imag_f*a
            # Multiply by (a-bi) e-k times
            for _ in range(e-k):
                real_f, imag_f = real_f*a + imag_f*b, real_f*b - imag_f*a
                
            new_real = current_z[0]*real_f - current_z[1]*imag_f
            new_imag = current_z[0]*imag_f + current_z[1]*real_f
            
            generate(idx + 1, (new_real, new_imag))
            
    generate(0, (1, 0))
    
    # Deduplicate and format
    unique_pairs = set()
    for x, y in pairs:
        if x <= y:
            unique_pairs.add((x*x, y*y))
            
    final_pairs = [[a, b] for a, b in sorted(unique_pairs)]
    
    return {'N': N, 'pairs': final_pairs}
