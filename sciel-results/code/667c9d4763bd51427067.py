import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    LIMIT = 10**18
    
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    # Precompute prefix products
    prefix_prod = [1]
    for p in primes:
        prefix_prod.append(prefix_prod[-1] * p)
        
    # Find max k for squarefree
    max_k = 0
    for i in range(1, len(primes)+1):
        if prefix_prod[i] <= LIMIT:
            max_k = i
        else:
            break
            
    best_n = 2
    best_score = 0
    
    def score_candidate(factors):
        """
        factors: list of (prime, exponent)
        Returns (n, score)
        """
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT:
            return 0, 0
            
        # Generate divisors
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        divs.sort()
        
        count = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                count += 1
            prev = curr
        return n, count

    def try_candidate(factors):
        nonlocal best_n, best_score
        n, score = score_candidate(factors)
        if n > 1 and score > best_score:
            best_score = score
            best_n = n
        return score

    # 1. Test squarefree with max_k primes
    if max_k > 0:
        factors = [(primes[i], 1) for i in range(max_k)]
        try_candidate(factors)
        
    # 2. Explore variations for k = max_k down to max_k - 5
    # We add exponents to the first few primes
    # We use a helper to generate exponent combinations within a limit
    
    def explore_exponents(k, limit_factor):
        """
        Explore extra exponents for the first few primes,
        given that the first k primes are already in the base (exp 1).
        limit_factor is the max multiplier allowed for the extra exponents.
        """
        if time.time() - start_time > DEADLINE:
            return
            
        # We will add extra exponents to primes 0, 1, 2, ... (2, 3, 5, ...)
        # We limit the number of primes we add exponents to to keep search space small
        # e.g. add to 2, 3, 5, 7
        
        # Recursive search for extra exponents
        def dfs(idx, current_multiplier, current_factors):
            nonlocal best_n, best_score
            if time.time() - start_time > DEADLINE:
                return
            if idx >= 4: # Stop adding exponents after 4 primes (2,3,5,7)
                # Try candidate
                # current_factors contains the base k primes + extra exponents for first idx primes
                # Actually, we need to construct the full factor list
                # Base: primes[0..k-1] with exp 1
                # Extra: added to primes[0..idx-1]
                # So factors are:
                # p_0: 1 + e_0
                # p_1: 1 + e_1
                # ...
                # p_{idx-1}: 1 + e_{idx-1}
                # p_idx ... p_{k-1}: 1
                
                # Construct factors list
                full_factors = []
                for i in range(k):
                    if i < idx:
                        # The extra exponent is stored in current_factors?
                        # Let's pass the extra exponents separately
                        pass
                    else:
                        full_factors.append((primes[i], 1))
                # This logic is getting messy. Let's pass a list of extra exponents.
                return
                
        # Simpler approach: Nested loops for extra exponents on 2, 3, 5, 7
        # Extra exp on 2: a
        # Extra exp on 3: b
        # Extra exp on 5: c
        # Extra exp on 7: d
        # Constraint: 2^a * 3^b * 5^c * 7^d <= limit_factor
        
        # Max a
        max_a = 0
        temp = limit_factor
        while temp >= 2:
            temp //= 2
            max_a += 1
            
        for a in range(max_a + 1):
            if time.time() - start_time > DEADLINE: break
            rem_a = limit_factor // (2 ** a)
            
            # Max b
            max_b = 0
            temp = rem_a
            while temp >= 3:
                temp //= 3
                max_b += 1
                
            for b in range(max_b + 1):
                if time.time() - start_time > DEADLINE: break
                rem_b = rem_a // (3 ** b)
                
                # Max c
                max_c = 0
                temp = rem_b
                while temp >= 5:
                    temp //= 5
                    max_c += 1
                    
                for c in range(max_c + 1):
                    if time.time() - start_time > DEADLINE: break
                    rem_c = rem_b // (5 ** c)
                    
                    # Max d
                    max_d = 0
                    temp = rem_c
                    while temp >= 7:
                        temp //= 7
                        max_d += 1
                        
                    for d in range(max_d + 1):
                        if time.time() - start_time > DEADLINE: break
                        # Construct factors
                        factors = []
                        # Prime 2 (idx 0)
                        factors.append((2, 1 + a))
                        # Prime 3 (idx 1)
                        if k > 1:
                            factors.append((3, 1 + b))
                        else:
                            # If k==1, we don't have 3 in base, but we added exp b?
                            # This logic assumes k is large enough to contain 2,3,5,7
                            # If k < 4, we need to handle carefully.
                            # But we are exploring k from max_k down to 10, so k >= 10 > 4.
                            pass
                        
                        # Prime 5 (idx 2)
                        if k > 2:
                            factors.append((5, 1 + c))
                        
                        # Prime 7 (idx 3)
                        if k > 3:
                            factors.append((7, 1 + d))
                            
                        # Rest of primes
                        for i in range(4, k):
                            factors.append((primes[i], 1))
                            
                        try_candidate(factors)
                        
                        # Check time
                        if time.time() - start_time > DEADLINE:
                            return

    # Run exploration for k = max_k down to max_k - 5
    for k in range(max_k, max(max_k - 6, 9), -1):
        base_prod = prefix_prod[k]
        if base_prod > LIMIT:
            continue
        limit_factor = LIMIT // base_prod
        explore_exponents(k, limit_factor)
        
        if time.time() - start_time > DEADLINE:
            break

    return best_n
