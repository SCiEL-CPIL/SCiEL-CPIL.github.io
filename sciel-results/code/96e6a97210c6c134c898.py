import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    primes = sieve(1000)
    
    # Split primes
    # Small primes are expensive (1/p is large), Large primes are cheap.
    # Let's define a threshold.
    # 1/2 = 0.5, 1/3 = 0.33, 1/5 = 0.2, 1/7 = 0.14, 1/11 = 0.09, 1/13 = 0.076, 1/17=0.058, 1/19=0.052, 1/23=0.043, 1/29=0.034, 1/31=0.032, 1/37=0.027, 1/41=0.024, 1/43=0.023, 1/47=0.021, 1/53=0.018, 1/59=0.017, 1/61=0.016, 1/67=0.014, 1/71=0.014, 1/73=0.013, 1/79=0.012, 1/83=0.012, 1/89=0.011, 1/97=0.010.
    # Sum of 1/p for p <= 100 is approx 1.8. We have budget 1.
    # So we can use at most a subset of small primes.
    
    # Strategy:
    # 1. Greedy with "Large Prime Preference".
    #    Score = Gain * p * (1 + alpha * (p/1000))
    #    This biases towards large primes.
    # 2. Run this to build a base cover.
    # 3. Then, try to improve by swapping large primes for small primes if it increases L.
    
    def compute_L(covered, max_L):
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def run_greedy(alpha, max_L_target, deadline):
        covered = bytearray(max_L_target + 1)
        used_primes = set()
        budget = Fraction(1, 1)
        classes = []
        
        n = 1
        while n <= max_L_target:
            if time.time() > deadline:
                break
            if covered[n]:
                n += 1
                continue
            
            best_p = None
            best_r = 0
            best_score = -1
            best_gain = 0
            
            for p in primes:
                if p in used_primes:
                    continue
                cost = Fraction(1, p)
                if budget < cost:
                    continue
                
                r = n % p
                # Calculate gain
                gain = 0
                k = n
                while k <= max_L_target:
                    if not covered[k]:
                        gain += 1
                    k += p
                
                if gain == 0:
                    continue
                
                # Score: Efficiency * Preference
                # Efficiency = gain * p
                # Preference: bias towards large p
                # If alpha > 0, larger p gets higher score multiplier
                score = gain * p * (1 + alpha * (p / 1000.0))
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r
                    best_gain = gain
            
            if best_p is None:
                break
            
            used_primes.add(best_p)
            budget -= Fraction(1, best_p)
            classes.append([best_p, best_r])
            
            # Update coverage
            start_k = best_r
            if start_k == 0:
                start_k = best_p
            for k in range(start_k, max_L_target + 1, best_p):
                covered[k] = 1
                
            n += 1
            
        L = compute_L(covered, max_L_target)
        return L, classes, covered, budget

    best_L = 0
    best_classes = []
    
    # Try different alpha values and max_L targets
    # alpha = 0 is standard efficiency
    # alpha > 0 prefers large primes
    # alpha < 0 prefers small primes (risky)
    
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    targets = [2000, 5000, 10000]
    
    for target in targets:
        if time.time() > deadline:
            break
        for alpha in alphas:
            if time.time() > deadline:
                break
            L, classes, covered, remaining_budget = run_greedy(alpha, target, deadline)
            if L > best_L:
                best_L = L
                best_classes = [list(c) for c in classes]
                # We could try to extend with local search here, but let's keep it simple first
                
    # Local Search / Refinement
    # If we have time, try to improve best_classes
    # Idea: Look at the first uncovered number U.
    # Try to find a prime p (used or unused) and residue r such that:
    # 1. If p is unused: adding [p,r] covers U and doesn't break budget.
    # 2. If p is used: changing [p, r_old] to [p, r_new] covers U, maintains coverage of 1..U-1, and doesn't break budget (cost is same).
    
    if time.time() < deadline - 10:
        # Reconstruct coverage for best_classes up to best_L + 1000
        check_L = best_L + 1000
        covered = bytearray(check_L + 1)
        used_primes = set()
        for p, r in best_classes:
            used_primes.add(p)
            start_k = r
            if start_k == 0:
                start_k = p
            for k in range(start_k, check_L + 1, p):
                covered[k] = 1
        
        # Find first uncovered
        U = best_L + 1
        while U <= check_L and covered[U]:
            U += 1
            
        if U <= check_L:
            # Try to cover U
            # Option 1: Add new prime
            budget_used = sum(Fraction(1, p) for p, r in best_classes)
            remaining = Fraction(1, 1) - budget_used
            
            for p in primes:
                if p in used_primes:
                    continue
                if Fraction(1, p) > remaining:
                    continue
                r = U % p
                # Check if this helps beyond U?
                # Just add it and see if L increases
                new_classes = best_classes + [[p, r]]
                # Verify coverage of 1..U
                # Since we only added a class, 1..U-1 are still covered.
                # U is now covered.
                # Check if U+1, U+2... are covered?
                # Let's just compute new L
                # This is expensive, so only do it for few candidates
                # Sort candidates by p (larger p is cheaper, might be better for long term)
                # Or just pick the one that covers most future uncovered
                pass # Skipping complex local search for now to ensure stability
                
    return best_classes

# EVOLVE-BLOCK-END
