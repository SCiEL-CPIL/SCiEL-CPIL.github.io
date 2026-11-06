"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import gcd
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Precompute all possible values and their contributions
    # We want to maximize sum of 1/(512-a)
    # Greedy: pick numbers that give high contribution while maintaining coprimality
    
    # Strategy: Try multiple greedy orderings and keep the best
    # Ordering 1: Descending by 1/(512-a) i.e., ascending a (smaller a gives larger 1/(512-a))
    # But we also need pairwise coprime constraint
    
    def greedy_with_order(order):
        chosen = []
        for a in order:
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    def score_set(chosen):
        return sum(Fraction(1, 512 - a) for a in chosen)
    
    # Generate different orderings to try
    orderings = []
    
    # Ordering 1: Ascending a (maximizes individual contributions)
    orderings.append(list(range(1, 512)))
    
    # Ordering 2: Descending a (original)
    orderings.append(list(range(511, 0, -1)))
    
    # Ordering 3: Prioritize numbers with large prime factors (harder to be coprime with)
    # Actually, prioritize numbers that are coprime with most others
    # Let's try: sort by number of coprime partners in descending order
    coprime_count = {}
    for a in range(1, 512):
        count = sum(1 for b in range(1, 512) if b != a and gcd(a, b) == 1)
        coprime_count[a] = count
    
    orderings.append(sorted(range(1, 512), key=lambda x: -coprime_count[x]))
    
    # Ordering 4: Prioritize primes (they're coprime with all non-multiples)
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [a for a in range(2, 512) if is_prime(a)]
    composites = [a for a in range(1, 512) if not is_prime(a) and a != 1]
    orderings.append([1] + primes + composites)
    
    # Ordering 5: Mix - start with high-value small numbers, then fill
    # Sort by 1/(512-a) descending, but break ties by coprimality flexibility
    order5 = sorted(range(1, 512), key=lambda x: (-1.0/(512-x), -coprime_count[x]))
    orderings.append(order5)
    
    # Try each ordering and keep the best
    best_set = None
    best_score = Fraction(0, 1)
    
    for i, order in enumerate(orderings):
        if time.time() > deadline:
            break
        candidate = greedy_with_order(order)
        s = score_set(candidate)
        if s > best_score:
            best_score = s
            best_set = candidate
    
    # Local search: try to improve by swapping elements
    if best_set and time.time() < deadline:
        current_set = list(best_set)
        current_score = best_score
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            n = len(current_set)
            
            # Try removing one element and adding a better one
            for i in range(n):
                if time.time() > deadline:
                    break
                removed = current_set[i]
                remaining = [x for j, x in enumerate(current_set) if j != i]
                
                # Find best replacement
                best_add = None
                best_add_score = current_score - Fraction(1, 512 - removed)
                
                for a in range(1, 512):
                    if a in remaining:
                        continue
                    if any(gcd(a, b) != 1 for b in remaining):
                        continue
                    new_score = current_score - Fraction(1, 512 - removed) + Fraction(1, 512 - a)
                    if new_score > best_add_score:
                        best_add_score = new_score
                        best_add = a
                
                if best_add is not None and best_add_score > current_score:
                    current_set[i] = best_add
                    current_score = best_add_score
                    improved = True
                    break
            
            # Try adding a new element if we can
            if not improved and time.time() < deadline:
                for a in range(1, 512):
                    if a in current_set:
                        continue
                    if any(gcd(a, b) != 1 for b in current_set):
                        continue
                    new_score = current_score + Fraction(1, 512 - a)
                    if new_score > current_score:
                        current_set.append(a)
                        current_score = new_score
                        improved = True
                        break
        
        best_set = current_set
        best_score = current_score
    
    return best_set if best_set else list(range(511, 0, -1))
# EVOLVE-BLOCK-END
