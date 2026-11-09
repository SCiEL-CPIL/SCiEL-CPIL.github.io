import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill
    
    # Precompute primes up to 128
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    # Count number of prime factors (with multiplicity) - fewer factors = less conflict potential
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Sort composites by number of prime factors (ascending) - simpler numbers first
    composites_sorted = sorted(composites, key=lambda x: prime_factor_count(x))
    
    def get_product_set(seq):
        """Return set of all consecutive block products."""
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods.add(p)
                if p > 10**18:  # cap to avoid huge integers
                    break
        return prods
    
    def try_insertion_fast(seq, c, prod_set):
        """Try to insert c into seq, using precomputed prod_set for speed.
        Returns (new_seq, new_prod_set, True) if successful, else (None, None, False)."""
        n = len(seq)
        
        # Find valid insertion positions (where seq[i-1] < c < seq[i])
        for i in range(n + 1):
            if i > 0 and seq[i-1] >= c:
                continue
            if i < n and c >= seq[i]:
                continue
            
            # Compute new products introduced by inserting c at position i
            # These are blocks that include c
            new_prods = set()
            valid = True
            
            # Blocks starting at i (c is first element)
            p = 1
            for j in range(i, n):
                if j == i:
                    p = c
                else:
                    p *= seq[j]
                if p > 10**18:
                    break
                if p in prod_set or p in new_prods:
                    valid = False
                    break
                new_prods.add(p)
            if not valid:
                continue
            
            # Blocks ending at i (c is last element), excluding the single-element block [c]
            p = c
            for j in range(i-1, -1, -1):
                p *= seq[j]
                if p > 10**18:
                    break
                if p in prod_set or p in new_prods:
                    valid = False
                    break
                new_prods.add(p)
            if not valid:
                continue
            
            # Blocks spanning across c: start before i, end after i
            # For each start position s < i, extend forward past i
            for s in range(i-1, -1, -1):
                p = c
                # First multiply by elements from s to i-1 (in reverse order, but product is commutative)
                for k in range(s, i):
                    p *= seq[k]
                    if p > 10**18:
                        break
                else:
                    # Now extend forward from i
                    for k in range(i, n):
                        p *= seq[k]
                        if p > 10**18:
                            break
                        if p in prod_set or p in new_prods:
                            valid = False
                            break
                        new_prods.add(p)
                    if not valid:
                        break
            if not valid:
                continue
            
            # All new products are unique and don't conflict with existing ones
            new_seq = seq[:i] + [c] + seq[i:]
            new_prod_set = prod_set | new_prods
            return new_seq, new_prod_set, True
        
        return None, None, False
    
    def build_seq(seed_order):
        """Build a sequence starting with primes, then greedily add composites."""
        seq = primes[:]
        prod_set = get_product_set(seq)
        used = set(seq)
        remaining = [x for x in composites_sorted if x not in used]
        
        # Apply seed_order shuffle
        random.Random(seed_order).shuffle(remaining)
        
        progress = True
        while progress:
            progress = False
            for c in remaining[:]:
                new_seq, new_prod_set, success = try_insertion_fast(seq, c, prod_set)
                if success:
                    seq = new_seq
                    prod_set = new_prod_set
                    remaining.remove(c)
                    progress = True
                    break
        return seq, prod_set
    
    best_seq = primes[:]
    best_prod_set = get_product_set(best_seq)
    best_len = len(best_seq)
    
    # Try many different random seeds/orders
    while time.time() < deadline:
        # Generate a new random seed
        seed_val = random.randint(0, 2**31 - 1)
        seq, prod_set = build_seq(seed_val)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            best_prod_set = prod_set
            
            # Try to further improve best_seq with local search
            improved = True
            while improved and time.time() < deadline:
                improved = False
                used = set(best_seq)
                remaining2 = [x for x in composites_sorted if x not in used]
                random.shuffle(remaining2)
                for c in remaining2:
                    new_seq, new_prod_set, success = try_insertion_fast(best_seq, c, best_prod_set)
                    if success:
                        best_seq = new_seq
                        best_prod_set = new_prod_set
                        best_len = len(best_seq)
                        improved = True
                        break
    
    return best_seq
