"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s kill
    
    # All candidate integers in [2, 40]
    candidates = list(range(2, 41))
    
    # We'll greedily build a set with distinct subset products
    # Order candidates: primes first (safest), then prime powers, then composites
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def prime_power_score(n):
        """Lower score = more 'prime-like' = safer to add early"""
        if is_prime(n):
            return 0
        # Check if it's a prime power
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if p > n:
                break
            m = n
            while m % p == 0:
                m //= p
            if m == 1:
                return 1  # prime power
        return 2  # general composite
    
    # Sort candidates: primes first, then prime powers, then composites
    # Within same category, smaller numbers first (more likely to fit)
    candidates.sort(key=lambda x: (prime_power_score(x), x))
    
    # Greedy construction with subset product verification
    selected = []
    products = {1}  # set of all subset products of selected elements
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if len(selected) >= 18:
            break
        
        # Try adding cand: new products = products ∪ {p * cand for p in products}
        new_products = set(products)
        for p in products:
            np = p * cand
            if np in new_products:
                break
            new_products.add(np)
        else:
            # No collision, accept
            selected.append(cand)
            products = new_products
            continue
    
    # Verify the final set
    if len(selected) < 18:
        # Try a more exhaustive search with backtracking
        # But first, let's try a different ordering
        selected = []
        products = {1}
        
        # Try multiple orderings
        orderings = [
            candidates,
            list(reversed(candidates)),
            sorted(candidates, key=lambda x: (prime_power_score(x), -x)),
        ]
        
        for order in orderings:
            if time.time() > deadline:
                break
            sel = []
            prods = {1}
            for cand in order:
                if time.time() > deadline:
                    break
                if len(sel) >= 18:
                    break
                new_prods = set(prods)
                ok = True
                for p in prods:
                    np = p * cand
                    if np in new_prods:
                        ok = False
                        break
                    new_prods.add(np)
                if ok:
                    sel.append(cand)
                    prods = new_prods
            if len(sel) > len(selected):
                selected = sel
                products = prods
            if len(selected) >= 18:
                break
    
    # Final verification
    def verify(distinct_list):
        prods = {1}
        for x in distinct_list:
            new_prods = set(prods)
            for p in prods:
                np = p * x
                if np in new_prods:
                    return False
                new_prods.add(np)
            prods = new_prods
        return True
    
    if not verify(selected):
        # Fallback: return first 12 primes (known valid)
        selected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Ensure all are Python int
    return [int(x) for x in selected]
# EVOLVE-BLOCK-END
