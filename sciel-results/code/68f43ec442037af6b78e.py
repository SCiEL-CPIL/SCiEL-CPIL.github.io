"""
Strategy: Multi-start greedy with incremental product tracking.
Each greedy pass tries elements in a different order, maintaining a running
product set. Only O(n) new products are checked per insertion. After greedy,
we do a targeted insertion pass trying to add excluded elements.
Multiple random orderings + deterministic sweeps give diverse starting points.
"""
import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 128
    all_elements = list(range(2, 129))
    n_elem = len(all_elements)  # 127

    def greedy_with_order(order):
        """
        Greedy: process elements in given order. For each, check if adding it
        (at its sorted position) causes any product collision. Use incremental
        product tracking for efficiency.
        """
        values = []  # sorted list
        product_set = set()
        
        for x in order:
            if x in product_set:
                # x itself is already a product (singleton of x, or product of some block)
                # Actually we need to check all new products involving x
                pass
            
            # Find insertion position (maintain sorted order)
            lo, hi = 0, len(values)
            while lo < hi:
                mid = (lo + hi) // 2
                if values[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            pos = lo
            
            # Compute all new products that would be created by inserting x at pos
            # New products: x * (product of values[pos+j] ... values[pos+k]) for j=0.., k>=j
            # Also: x * (product of values[pos-1]...values[pos-1+j]) * ... 
            # Actually: any consecutive block that includes x and some neighbors.
            # A block [i..j] that includes position pos (0-indexed in new array).
            # In old array, this corresponds to blocks [i..pos-1], [pos..j], [i..j] where
            # i <= pos <= j in new indexing. In old indexing, i..pos-1 maps to old[i..pos-1],
            # pos..j maps to x * old[pos..j-1].
            # 
            # New products are:
            # For left extension L (0..pos elements to the left) and right extension R (0..len-pos elements to right):
            # product = (prod of values[pos-L..pos-1]) * x * (prod of values[pos..pos+R-1])
            # where L=0 means no left, R=0 means no right (just x).
            
            # Efficient check: compute left products and right products
            # left_prod[L] = product of values[pos-L : pos] for L = 0, 1, 2, ...
            # right_prod[R] = product of values[pos : pos+R] for R = 0, 1, 2, ...
            
            left_prods = [1]  # L=0
            p = 1
            for L in range(1, pos + 1):
                p *= values[pos - L]
                left_prods.append(p)
            
            right_prods = [1]  # R=0
            p = 1
            for R in range(1, len(values) - pos + 1):
                p *= values[pos + R - 1]
                right_prods.append(p)
            
            # Check all combinations
            conflict = False
            for lp in left_prods:
                for rp in right_prods:
                    prod = lp * x * rp
                    if prod in product_set:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                # Insert x
                values.insert(pos, x)
                # Update product_set with all new products
                # Recompute left and right prods for the new position
                # Actually we already have them; just add all
                left_prods2 = [1]
                p = 1
                for L in range(1, pos + 1):
                    p *= values[pos - L]
                    left_prods2.append(p)
                
                right_prods2 = [1]
                p = 1
                for R in range(1, len(values) - pos):
                    p *= values[pos + R]
                    right_prods2.append(p)
                
                for lp in left_prods2:
                    for rp in right_prods2:
                        product_set.add(lp * x * rp)
        
        return values

    def is_valid(seq):
        """Full validation check."""
        if not seq:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        n = len(seq)
        products = set()
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return len(products) == n * (n + 1) // 2

    def targeted_insertion(seq):
        """Try to insert each excluded element into the sequence."""
        seq_set = set(seq)
        excluded = [x for x in all_elements if x not in seq_set]
        improved = True
        while improved:
            improved = False
            for x in excluded[:]:
                # Try inserting x
                lo, hi = 0, len(seq)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if seq[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                pos = lo
                
                # Check for conflicts
                left_prods = [1]
                p = 1
                for L in range(1, pos + 1):
                    p *= seq[pos - L]
                    left_prods.append(p)
                
                right_prods = [1]
                p = 1
                for R in range(1, len(seq) - pos + 1):
                    p *= seq[pos + R - 1]
                    right_prods.append(p)
                
                conflict = False
                # We need to check against all existing products
                # But we don't have the full product set here... 
                # Let's just do a quick check: compute all new products and see
                # if any equals an existing product. For efficiency, compute
                # the existing product set once.
                # Actually, let's use a different approach: just try and validate.
                # For small sequences this is fine.
                new_seq = seq[:pos] + [x] + seq[pos:]
                if is_valid(new_seq):
                    seq = new_seq
                    seq_set.add(x)
                    excluded.remove(x)
                    improved = True
                    break  # restart the loop
        return seq

    # Phase 1: Deterministic greedy (natural order)
    best_seq = greedy_with_order(all_elements)
    best_len = len(best_seq)
    
    if time.time() - start_time > DEADLINE:
        return best_seq

    # Phase 2: Multi-start with random orderings
    rng = random.Random(42)
    num_starts = 200  # we'll do as many as time allows
    
    for trial in range(num_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Different ordering strategies
        strategy = trial % 5
        if strategy == 0:
            order = all_elements[:]
            rng.shuffle(order)
        elif strategy == 1:
            # Reverse
            order = all_elements[::-1]
        elif strategy == 2:
            # Primes first
            def is_prime(n):
                if n < 2: return False
                if n < 4: return True
                if n % 2 == 0 or n % 3 == 0: return False
                d = 5
                while d * d <= n:
                    if n % d == 0 or n % (d+2) == 0:
                        return False
                    d += 6
                return True
            primes = [x for x in all_elements if is_prime(x)]
            composites = [x for x in all_elements if not is_prime(x)]
            order = primes + composites
            rng.shuffle(primes)
            rng.shuffle(composites)
            order = primes + composites
        elif strategy == 3:
            # Large first
            order = all_elements[::-1]
            rng.shuffle(order)
        else:
            # Random with bias toward smaller numbers
            order = sorted(all_elements, key=lambda x: rng.random() * x)
        
        seq = greedy_with_order(order)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
    
    if time.time() - start_time > DEADLINE - 3:
        return best_seq

    # Phase 3: Targeted insertion on best sequence
    best_seq = targeted_insertion(best_seq)
    best_len = len(best_seq)

    # Phase 4: More random restarts with targeted insertion
    for trial in range(100):
        if time.time() - start_time > DEADLINE - 3:
            break
        
        order = all_elements[:]
        rng.shuffle(order)
        seq = greedy_with_order(order)
        seq = targeted_insertion(seq)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]

    # Final validation
    if not is_valid(best_seq):
        # Fallback to simple greedy
        best_seq = greedy_with_order(all_elements)

    return best_seq
