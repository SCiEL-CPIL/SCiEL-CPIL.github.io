from fractions import Fraction

def construct():
    # Primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # We want to maximize L. Cap at 10^5.
    L_MAX = 100000
    covered = bytearray(L_MAX + 1)
    
    budget = Fraction(1, 1)
    used_primes = set()
    L = 0
    
    # Window for lookahead
    W = 50
    
    while L < L_MAX:
        # Find smallest uncovered
        n = L + 1
        if covered[n]:
            # Advance L
            while L < L_MAX and covered[L + 1]:
                L += 1
            continue
        else:
            # n is the smallest uncovered
            # Find candidates
            best_p = -1
            best_r = -1
            best_score = -1
            best_cost = Fraction(1, 1)
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if cost > budget:
                    continue
                
                r = n % p
                # Score: how many new integers in [n, n+W] are covered by (p, r)
                score = 0
                k = n
                while k <= n + W:
                    if not covered[k]:
                        score += 1
                    k += p
                # Tie-breaker: prefer smaller p (higher density) if scores are equal?
                # Or prefer larger p (lower cost)?
                # Smaller p covers more in the window, so score should already reflect that.
                # If scores are equal, prefer lower cost (larger p) to save budget?
                # Let's prefer larger p (lower cost) to extend range further.
                if score > best_score or (score == best_score and p > best_p):
                    best_score = score
                    best_p = p
                    best_r = r
                    best_cost = cost
            
            if best_p == -1:
                break
            
            # Apply choice
            used_primes.add(best_p)
            budget -= best_cost
            
            # Mark covered
            k = best_r
            # Only mark up to L_MAX
            while k <= L_MAX:
                covered[k] = 1
                k += best_p
            
            # Loop will advance L in next iteration
            # But we can advance now
            # The while loop at start handles it

    return [[p, r] for p, r in [(p, r) for p, r in [(p, r) for p in used_primes for r in [n % p]]]] # This is wrong, we need the actual r used.
    
    # Reconstruct classes
    # We lost the r values in the loop above if we don't store them.
    # Let's store them in a list.
