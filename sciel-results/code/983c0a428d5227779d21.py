"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import reduce

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    LIMIT = 10**18
    
    # First 16 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    def get_divisors_sorted(factors):
        """Generate all divisors sorted, from list of (prime, exponent) pairs."""
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
        return divs
    
    def tau_perp_fast(factors):
        """Count adjacent coprime pairs in sorted divisors - optimized."""
        divs = get_divisors_sorted(factors)
        count = 0
        n = len(divs)
        for i in range(n - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    best_n = 2
    best_score = 0
    
    def try_candidate(factors):
        nonlocal best_n, best_score
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT or n < 2:
            return
        try:
            score = tau_perp_fast(factors)
            if score > best_score:
                best_score = score
                best_n = n
        except Exception:
            pass
    
    # Strategy: explore configurations with many primes and small exponents
    # Key: squarefree with k primes is a strong baseline; then add exponents to small primes
    
    # 1. Squarefree with 13, 14, 15 primes
    for k in [13, 14, 15]:
        try_candidate([(p, 1) for p in primes[:k]])
    
    # 2. 15 primes with exponent on 2
    for a in range(2, 8):
        n = 2**a
        for p in primes[1:15]:
            n *= p
        if n <= LIMIT:
            try_candidate([(2, a)] + [(p, 1) for p in primes[1:15]])
    
    # 3. 14 primes with exponents on 2 and 3
    for a in range(2, 10):
        for b in range(1, 5):
            n = 2**a * 3**b
            for p in primes[2:14]:
                n *= p
            if n <= LIMIT:
                try_candidate([(2, a), (3, b)] + [(p, 1) for p in primes[2:14]])
    
    # 4. 13 primes with exponents on 2, 3, 5
    for a in range(2, 12):
        for b in range(1, 6):
            for c in range(1, 4):
                n = 2**a * 3**b * 5**c
                for p in primes[3:13]:
                    n *= p
                if n <= LIMIT:
                    try_candidate([(2, a), (3, b), (5, c)] + [(p, 1) for p in primes[3:13]])
    
    if time.time() - start_time > DEADLINE - 20:
        return best_n
    
    # 5. 12 primes with exponents on 2, 3, 5, 7
    for a in range(2, 14):
        for b in range(1, 7):
            for c in range(1, 5):
                for d in range(1, 4):
                    n = 2**a * 3**b * 5**c * 7**d
                    for p in primes[4:12]:
                        n *= p
                    if n <= LIMIT:
                        try_candidate([(2, a), (3, b), (5, c), (7, d)] + [(p, 1) for p in primes[4:12]])
    
    if time.time() - start_time > DEADLINE - 10:
        return best_n
    
    # 6. 11 primes with exponents on 2, 3, 5, 7, 11
    for a in range(2, 16):
        for b in range(1, 8):
            for c in range(1, 6):
                for d in range(1, 5):
                    for e in range(1, 4):
                        n = 2**a * 3**b * 5**c * 7**d * 11**e
                        for p in primes[5:11]:
                            n *= p
                        if n <= LIMIT:
                            try_candidate([(2, a), (3, b), (5, c), (7, d), (11, e)] + [(p, 1) for p in primes[5:11]])
    
    if time.time() - start_time > DEADLINE - 5:
        return best_n
    
    # 7. Systematic search: try various exponent combinations for first few primes
    # with remaining primes squarefree
    # Focus on configurations where we have 10-14 distinct primes
    
    # Precompute products of trailing primes
    def trailing_product(start_idx, end_idx):
        prod = 1
        for p in primes[start_idx:end_idx]:
            prod *= p
        return prod
    
    # For each number of "exponented" primes (2-5), search exponent combinations
    # and multiply by product of remaining primes
    
    # Case: 4 exponented primes (2,3,5,7) + squarefree tail
    for tail_start in range(4, 15):
        tail_prod = trailing_product(4, tail_start)
        if tail_prod > LIMIT:
            break
        for a in range(1, 15):
            n2 = 2**a
            if n2 * tail_prod > LIMIT:
                break
            for b in range(1, 8):
                n3 = n2 * 3**b
                if n3 * tail_prod > LIMIT:
                    break
                for c in range(1, 6):
                    n5 = n3 * 5**c
                    if n5 * tail_prod > LIMIT:
                        break
                    for d in range(1, 5):
                        n7 = n5 * 7**d
                        if n7 * tail_prod > LIMIT:
                            break
                        factors = [(2, a), (3, b), (5, c), (7, d)]
                        for p in primes[4:tail_start]:
                            factors.append((p, 1))
                        try_candidate(factors)
    
    if time.time() - start_time > DEADLINE - 3:
        return best_n
    
    # 8. Try replacing some primes with larger ones to create better gaps
    # This can sometimes increase coprime adjacency
    base_15 = primes[:15]
    base_prod_15 = 1
    for p in base_15:
        base_prod_15 *= p
    
    # Try swapping the largest prime with larger primes
    for swap_idx in range(10, 15):
        for new_p in primes[15:40]:
            cand_prod = base_prod_15 // base_15[swap_idx] * new_p
            if cand_prod > LIMIT:
                break
            factors = [(p, 1) for p in base_15]
            factors[swap_idx] = (new_p, 1)
            try_candidate(factors)
    
    return best_n
# EVOLVE-BLOCK-END
