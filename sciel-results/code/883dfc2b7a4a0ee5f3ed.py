"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

P = {'n': 40, 'maximum_size': 18}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Primes up to 40 - known valid set of size 12
    primes = [a for a in range(2, P['n'] + 1) if all(a % d for d in range(2, isqrt(a) + 1))]
    
    # Direct product-based validity check
    def is_valid(S):
        if not S:
            return True
        products = {1}
        for x in S:
            new_prods = {p * x for p in products}
            if products & new_prods:
                return False
            products |= new_prods
        return True
    
    # Fast incremental check: can we add x to current set S (maintaining products)?
    def _can_add(products, x):
        new_prods = {p * x for p in products}
        return not (products & new_prods)
    
    def get_products(S):
        products = {1}
        for x in S:
            new_prods = {p * x for p in products}
            products |= new_prods
        return products
    
    # Greedy construction with incremental product tracking
    def greedy_construct(order, randomize=True):
        S = []
        products = {1}
        remaining = list(range(2, P['n'] + 1))
        if randomize:
            random.shuffle(remaining)
        else:
            if order == 1:
                remaining.sort()
            elif order == 2:
                remaining.sort(reverse=True)
            elif order == 3:
                remaining.sort(key=lambda x: len([d for d in range(2, isqrt(x)+1) if x % d == 0]))
            else:
                remaining.sort(key=lambda x: x)
        
        for x in remaining:
            if _can_add(products, x):
                new_prods = {p * x for p in products}
                products |= new_prods
                S.append(x)
                if len(S) >= P['maximum_size']:
                    break
        return S
    
    # Start with primes as baseline
    best_set = primes[:]
    best_size = len(best_set)
    
    # Run many greedy passes with different orderings
    pass_count = 0
    while time.time() < deadline - 5:
        # Random ordering
        S = greedy_construct(0, randomize=True)
        pass_count += 1
        if len(S) > best_size:
            best_set = S[:]
            best_size = len(S)
            if best_size >= P['maximum_size']:
                break
        
        # Deterministic orderings
        for order in [1, 2, 3, 4]:
            if time.time() > deadline - 5:
                break
            S = greedy_construct(order, randomize=False)
            pass_count += 1
            if len(S) > best_size:
                best_set = S[:]
                best_size = len(S)
                if best_size >= P['maximum_size']:
                    break
    
    # Local search: try to improve best_set via swaps
    if best_size < P['maximum_size'] and time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            S = best_set[:]
            products = get_products(S)
            candidates = [x for x in range(2, P['n'] + 1) if x not in S]
            random.shuffle(candidates)
            
            for to_remove in S[:]:
                if time.time() > deadline - 2:
                    break
                if to_remove not in S:
                    continue
                
                # Temporarily remove to_remove
                S.remove(to_remove)
                # Recompute products without to_remove
                products = get_products(S)
                
                for to_add in candidates:
                    if _can_add(products, to_add):
                        # Valid swap
                        S.append(to_add)
                        new_products = {p * to_add for p in products}
                        products |= new_products
                        
                        # Try to add more
                        added = True
                        while added and time.time() < deadline - 2:
                            added = False
                            rem = [x for x in range(2, P['n'] + 1) if x not in S]
                            random.shuffle(rem)
                            for x in rem:
                                if _can_add(products, x):
                                    new_prods = {p * x for p in products}
                                    products |= new_prods
                                    S.append(x)
                                    added = True
                                    break
                        
                        if len(S) > best_size:
                            best_set = S[:]
                            best_size = len(S)
                            improved = True
                            if best_size >= P['maximum_size']:
                                break
                        break  # Only one swap per to_remove attempt
                if best_size >= P['maximum_size']:
                    break
    
    # Final verification
    if is_valid(best_set):
        return best_set
    else:
        # Fallback to primes
        return primes

# EVOLVE-BLOCK-END
