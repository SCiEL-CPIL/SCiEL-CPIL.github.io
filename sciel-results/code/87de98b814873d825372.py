"""
Dense increasing sequences with unique block products.
Strategy: Greedy construction with backtracking and time management.
The seed program uses a simple greedy approach. We improve by:
1. Using a smarter ordering (not just ascending) to find longer sequences.
2. Implementing limited backtracking to escape local optima.
3. Tracking all block products efficiently.
4. Time-bounded search with deadline guard.
"""

import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin before 60s limit
    
    def elapsed():
        return time.time() - start_time
    
    def check_time():
        return elapsed() < deadline
    
    def all_block_products(seq):
        """Compute all nonempty consecutive block products of seq."""
        prods = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                prods.add(prod)
        return prods
    
    def can_add(seq, x, prods):
        """Check if x can be appended to seq without product collision."""
        # Compute new block products ending at x
        new_prods = set()
        prod = x
        for a in reversed(seq):
            new_prods.add(prod)
            prod *= a
        new_prods.add(x)
        # Check intersection with existing products
        if new_prods & prods:
            return False, None
        return True, new_prods
    
    def greedy_construct(start_val=2, order=None):
        """Greedy construction with given ordering of candidates."""
        if order is None:
            order = list(range(2, 129))
        
        values = []
        products = set()
        
        for x in order:
            if not check_time():
                break
            # Check if x can be added
            new_prods = set()
            prod = x
            for a in reversed(values):
                new_prods.add(prod)
                prod *= a
            new_prods.add(x)
            
            if not (new_prods & products):
                values.append(x)
                products.update(new_prods)
        
        return values
    
    def backtrack_search(candidates, current_seq, current_prods, max_depth=None):
        """
        Backtracking search to find a longer sequence.
        candidates: sorted list of candidate values not yet used
        current_seq: current sequence
        current_prods: set of all block products
        Returns best sequence found.
        """
        best = list(current_seq)
        
        if not check_time():
            return best
        
        if max_depth is not None and len(current_seq) >= max_depth:
            return best
        
        for i, x in enumerate(candidates):
            if not check_time():
                break
            
            # Try adding x
            new_prods = set()
            prod = x
            for a in reversed(current_seq):
                new_prods.add(prod)
                prod *= a
            new_prods.add(x)
            
            if new_prods & current_prods:
                continue
            
            # Add x and recurse
            new_seq = current_seq + [x]
            new_prod_set = current_prods | new_prods
            remaining = candidates[i+1:]
            
            result = backtrack_search(remaining, new_seq, new_prod_set, max_depth)
            if len(result) > len(best):
                best = result
        
        return best
    
    # Strategy 1: Simple greedy (baseline)
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    if not check_time():
        return best_seq
    
    # Strategy 2: Try different orderings
    # Order by "primality" - primes first, then composites with large prime factors
    def score_value(x):
        """Score a value for ordering preference."""
        # Prefer numbers with large prime factors (less likely to collide)
        # and numbers that are "far" from 1 in terms of prime factorization
        n = x
        max_prime = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                max_prime = max(max_prime, d)
                n //= d
            d += 1
        if n > 1:
            max_prime = max(max_prime, n)
        # Also prefer larger values (they create larger products, less collision)
        return max_prime * 1000 + x
    
    # Try several orderings
    orderings = [
        list(range(2, 129)),  # ascending
        list(range(128, 1, -1)),  # descending
        sorted(range(2, 129), key=score_value, reverse=True),  # by max prime factor
        sorted(range(2, 129), key=lambda x: (x % 2, -x)),  # evens first then odds desc
        sorted(range(2, 129), key=lambda x: (x % 3, -x)),  # multiples of 3 first
    ]
    
    for order in orderings:
        if not check_time():
            break
        seq = greedy_construct(order=order)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
    
    if not check_time():
        return best_seq
    
    # Strategy 3: Local improvement - try swapping elements
    def try_improve(seq):
        """Try to improve sequence by local swaps and insertions."""
        if not check_time():
            return seq
        
        best = list(seq)
        best_len = len(best)
        prods = all_block_products(best)
        
        # Try inserting unused numbers
        used = set(best)
        unused = [x for x in range(2, 129) if x not in used]
        
        for pos in range(len(best) + 1):
            if not check_time():
                break
            for x in unused:
                if not check_time():
                    break
                # Try inserting x at position pos
                new_seq = best[:pos] + [x] + best[pos:]
                new_prods = all_block_products(new_seq)
                if len(new_prods) == sum(range(1, len(new_seq)+1)):
                    # All products distinct (count should be n*(n+1)/2)
                    if len(new_seq) > best_len:
                        best = new_seq
                        best_len = len(best)
                        prods = new_prods
        
        return best
    
    best_seq = try_improve(best_seq)
    best_len = len(best_seq)
    
    if not check_time():
        return best_seq
    
    # Strategy 4: More aggressive backtracking from a good starting point
    if check_time():
        # Use the best sequence found so far as a guide
        # Try to build a sequence using backtracking with pruning
        used_in_best = set(best_seq)
        # Try a fresh backtracking search with limited depth
        candidates = list(range(2, 129))
        
        # Limit search: only explore up to best_len + 5 depth
        max_d = best_len + 5
        result = backtrack_search(candidates, [], set(), max_depth=max_d)
        if len(result) > best_len:
            best_seq = result
            best_len = len(best)
    
    return best_seq
