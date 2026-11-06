"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    LIMIT = 10**18
    
    # First 25 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return sorted(divs)
    
    def tau_perp_from_factors(factors):
        """Count adjacent coprime pairs in sorted divisors."""
        divs = get_divisors_from_factors(factors)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    def compute_n(factors):
        """Compute n from factors."""
        n = 1
        for p, e in factors:
            n *= p ** e
        return n
    
    best_n = 0
    best_score = 0
    
    def try_candidate(factors):
        nonlocal best_n, best_score
        n = compute_n(factors)
        if n > LIMIT or n < 2:
            return
        score = tau_perp_from_factors(factors)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Try squarefree with k primes for k = 12 to 16
    for k in range(12, 17):
        if k <= len(primes):
            try_candidate([(p, 1) for p in primes[:k]])
    
    # Try increasing exponents of small primes while keeping many primes
    # 15 primes: 2^a * 3 * 5 * 7 * ... * 47
    for a in range(2, 8):
        n = 2**a
        for p in primes[1:15]:
            n *= p
        if n <= LIMIT:
            try_candidate([(2, a)] + [(p, 1) for p in primes[1:15]])
    
    # 14 primes: 2^a * 3^b * 5 * 7 * ... * 43
    for a in range(2, 10):
        for b in range(1, 5):
            n = 2**a * 3**b
            for p in primes[2:14]:
                n *= p
            if n <= LIMIT:
                try_candidate([(2, a), (3, b)] + [(p, 1) for p in primes[2:14]])
    
    # 13 primes: 2^a * 3^b * 5^c * 7 * ... * 41
    for a in range(2, 12):
        for b in range(1, 6):
            for c in range(1, 4):
                n = 2**a * 3**b * 5**c
                for p in primes[3:13]:
                    n *= p
                if n <= LIMIT:
                    try_candidate([(2, a), (3, b), (5, c)] + [(p, 1) for p in primes[3:13]])
    
    # 12 primes: 2^a * 3^b * 5^c * 7^d * 11 * ... * 37
    for a in range(2, 14):
        for b in range(1, 7):
            for c in range(1, 5):
                for d in range(1, 4):
                    n = 2**a * 3**b * 5**c * 7**d
                    for p in primes[4:12]:
                        n *= p
                    if n <= LIMIT:
                        try_candidate([(2, a), (3, b), (5, c), (7, d)] + [(p, 1) for p in primes[4:12]])
    
    # 11 primes: 2^a * 3^b * 5^c * 7^d * 11^e * 13 * ... * 31
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
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # Local search: try to improve by adjusting exponents or swapping primes
    # Start from best candidate and explore neighbors
    best_factors = None
    if best_n > 0:
        # Reconstruct factors from best_n
        temp = best_n
        factors = []
        for p in primes:
            if temp % p == 0:
                e = 0
                while temp % p == 0:
                    temp //= p
                    e += 1
                factors.append((p, e))
        best_factors = factors
    
    if best_factors is not None:
        # Try local improvements with more move types
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            current_factors = list(best_factors)
            current_primes_set = set(p for p, e in current_factors)
            
            # Generate all possible single-step moves
            moves = []
            
            # Try increasing each exponent by 1
            for i in range(len(current_factors)):
                p, e = current_factors[i]
                if e < 15:
                    new_factors = list(current_factors)
                    new_factors[i] = (p, e + 1)
                    moves.append(new_factors)
            
            # Try decreasing each exponent by 1 (if > 1)
            for i in range(len(current_factors)):
                p, e = current_factors[i]
                if e > 1:
                    new_factors = list(current_factors)
                    new_factors[i] = (p, e - 1)
                    moves.append(new_factors)
            
            # Try adding a new prime (smallest available first)
            if len(current_factors) < 18:
                for p in primes:
                    if p not in current_primes_set:
                        new_factors = list(current_factors) + [(p, 1)]
                        moves.append(new_factors)
            
            # Try removing a prime with exponent 1 (if we have many primes)
            if len(current_factors) > 10:
                for i in range(len(current_factors)):
                    p, e = current_factors[i]
                    if e == 1:
                        new_factors = [f for j, f in enumerate(current_factors) if j != i]
                        if new_factors:
                            moves.append(new_factors)
            
            # Try swapping: replace a prime with a smaller unused prime
            for i in range(len(current_factors)):
                p_old, e = current_factors[i]
                for p_new in primes:
                    if p_new < p_old and p_new not in current_primes_set:
                        new_factors = list(current_factors)
                        new_factors[i] = (p_new, e)
                        moves.append(new_factors)
            
            # Try swapping: replace a prime with a larger unused prime (to free budget)
            for i in range(len(current_factors)):
                p_old, e = current_factors[i]
                for p_new in primes:
                    if p_new > p_old and p_new not in current_primes_set:
                        new_factors = list(current_factors)
                        new_factors[i] = (p_new, e)
                        moves.append(new_factors)
            
            # Evaluate all moves, keep best improvement
            best_move_factors = None
            best_move_score = best_score
            
            for new_factors in moves:
                n = compute_n(new_factors)
                if n < 2 or n > LIMIT:
                    continue
                score = tau_perp_from_factors(new_factors)
                if score > best_move_score:
                    best_move_score = score
                    best_move_factors = new_factors
            
            if best_move_factors is not None:
                best_score = best_move_score
                best_n = compute_n(best_move_factors)
                best_factors = best_move_factors
                improved = True
    
    return best_n
# EVOLVE-BLOCK-END
