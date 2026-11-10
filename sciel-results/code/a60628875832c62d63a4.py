import math

def construct():
    max_n = 1000000
    min_n = 10000
    
    # 1. Generate all powerful numbers up to max_n
    # A powerful number is of the form p^a q^b ... with a,b >= 2
    # We can generate them by iterating primes and their powers.
    
    # Sieve for primes up to sqrt(max_n) is not enough, we need primes up to max_n?
    # No, if p^2 <= max_n, then p <= sqrt(max_n).
    # But powerful numbers can have large prime factors if the exponent is 2?
    # If p^2 <= max_n, then p <= 1000.
    # So all prime factors of a powerful number <= 10^6 must be <= 1000.
    
    primes = []
    is_prime = [True] * (int(math.isqrt(max_n)) + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, int(math.isqrt(max_n)) + 1, i):
                is_prime[j] = False
                
    powerful_numbers = set()
    powerful_numbers.add(1) # 1 is powerful? 1 has no prime factors. Product is empty=1.
    
    # Generate powerful numbers
    # We can use a recursive approach or iterative.
    # Let's use a list of (value, prime_index) to avoid duplicates?
    # Simpler: Iterate over all combinations of prime powers.
    # Since number of primes is small (~168 primes <= 1000), and exponents are small.
    
    # Actually, a powerful number is determined by its prime factorization.
    # We can generate them by DFS.
    
    def gen_powerful(idx, current_val):
        if current_val > max_n:
            return
        powerful_numbers.add(current_val)
        for i in range(idx, len(primes)):
            p = primes[i]
            p_pow = p * p
            if p_pow > max_n // current_val:
                break
            # Multiply by p^2, p^3, ...
            val = current_val * p_pow
            while val <= max_n:
                gen_powerful(i + 1, val) # Next prime must be larger
                val *= p
                
    gen_powerful(0, 1)
    
    powerful_list = sorted(list(powerful_numbers))
    
    # Precompute prime factors for powerful numbers to check disjointness
    # and to help with modular arithmetic?
    # We just need to check gcd(A, B) == 1.
    
    best_n = min_n
    best_val = 0
    
    # To speed up, we can iterate A and B such that A <= B
    # And A * S_1 >= min_n, A * S_1 <= max_n
    # S_1 >= ceil(min_n / A)
    # S_1 <= floor(max_n / A)
    
    # For each A, for each B > A (or B < A), check disjointness.
    # If disjoint, solve A * S_1 + 1 = B * S_2
    # S_1 = (B * S_2 - 1) / A
    # Or S_2 = (A * S_1 + 1) / B
    
    # We iterate S_1 in the valid range.
    # S_1 must be squarefree.
    
    # Optimization: Precompute squarefree check?
    # A number is squarefree if it is not divisible by any p^2.
    
    # We can precompute a boolean array is_squarefree up to max_n?
    # Or just check on the fly.
    
    # Given the small number of powerful numbers, we can afford some checks.
    
    # However, iterating all pairs (A, B) and all S_1 might be slow.
    # Number of powerful numbers ~ 3000.
    # Pairs ~ 4.5 million.
    # Range of S_1 for A=1 is 10000..1000000 (10^6 values).
    # This is too slow.
    
    # We need to be smarter.
    # We want to maximize A * B * S_1 * S_2? No, A * B is fixed for the pair.
    # We want to maximize A * B.
    # So we should prioritize pairs with large A and B.
    
    # Sort powerful numbers in descending order.
    # Iterate pairs in descending order of A*B.
    # Stop when A*B < best_val (since S_1, S_2 >= 1, max product is A*B).
    
    powerful_list_desc = sorted(powerful_list, reverse=True)
    
    # Precompute is_squarefree up to max_n // min_powerful (which is 1) -> max_n
    # Actually, we only need to check S_1 and S_2 which are <= max_n / A.
    
    # Let's just do a brute force over the powerful pairs but break early.
    
    for i in range(len(powerful_list_desc)):
        A = powerful_list_desc[i]
        if A * powerful_list_desc[i] < best_val: # Even with B=A, product is too small?
             # Actually B can be different.
             # Max product for this A is A * max_B.
             # If A * max_B < best_val, break outer loop?
             # max_B is powerful_list_desc[0].
             pass
             
        for j in range(i, len(powerful_list_desc)):
            B = powerful_list_desc[j]
            if A * B < best_val:
                break # Since B is decreasing, further B will be smaller
            
            # Check disjointness
            if math.gcd(A, B) != 1:
                continue
                
            # We need to find S_1, S_2 such that:
            # 1. A * S_1 + 1 = B * S_2
            # 2. S_1, S_2 are squarefree
            # 3. min_n <= A * S_1 <= max_n
            
            # From 1: B * S_2 - A * S_1 = 1
            # S_1 must be such that A * S_1 + 1 is divisible by B.
            # A * S_1 = -1 (mod B)
            # S_1 = -A^{-1} (mod B)
            
            try:
                inv_A = pow(A, -1, B)
            except ValueError:
                continue # gcd(A,B) != 1, but we checked.
                
            # S_1 = k * B - inv_A
            # Wait, A * (-inv_A) = -1 = B-1? No.
            # A * inv_A = 1 (mod B).
            # A * S_1 = -1 (mod B) => S_1 = -inv_A (mod B).
            # So S_1 = k * B + (B - inv_A)
            
            base_S1 = B - inv_A
            if base_S1 == 0:
                base_S1 = B
                
            # Range for S_1:
            # min_n <= A * S_1 <= max_n
            # S_1 >= ceil(min_n / A)
            # S_1 <= floor(max_n / A)
            
            min_S1 = math.ceil(min_n / A)
            max_S1 = math.floor(max_n / A)
            
            if min_S1 > max_S1:
                continue
                
            # Find smallest k such that S_1 >= min_S1
            # S_1 = k * B + base_S1
            # k * B >= min_S1 - base_S1
            k_min = math.ceil((min_S1 - base_S1) / B)
            if k_min < 0:
                k_min = 0 # S_1 must be positive? Yes, squarefree numbers are >= 1.
                
            # Iterate k
            S_1 = k_min * B + base_S1
            while S_1 <= max_S1:
                # Check if S_1 is squarefree
                # And S_2 = (A * S_1 + 1) / B is integer and squarefree
                S_2 = (A * S_1 + 1) // B
                if (A * S_1 + 1) % B != 0:
                    S_1 += B
                    continue
                    
                # Check squarefree
                # Simple check:
                def is_sqfree(x):
                    if x <= 1: return True
                    for p in primes:
                        if p * p > x:
                            break
                        if x % (p * p) == 0:
                            return False
                    return True
                    
                if is_sqfree(S_1) and is_sqfree(S_2):
                    val = A * B
                    if val > best_val:
                        best_val = val
                        best_n = A * S_1
                    S_1 += B
                    
    return best_n
