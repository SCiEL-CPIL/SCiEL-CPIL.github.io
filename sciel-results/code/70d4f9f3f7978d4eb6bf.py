import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 82.0

    # Precompute prime factorizations for 2..128
    def factorize(n):
        f = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                f.append(d)
                n //= d
            d += 1
        if n > 1:
            f.append(n)
        return f

    # Represent each number as a frozenset of (prime, exponent) pairs
    # Actually use a tuple of (prime, exp) sorted by prime for hashing
    num_vec = {}
    for n in range(2, 129):
        factors = factorize(n)
        # Count exponents
        from collections import Counter
        cnt = Counter(factors)
        num_vec[n] = tuple(sorted(cnt.items()))

    # For product of a sequence of numbers, the exponent vector is the sum
    # We represent a product vector as a frozenset of (prime, total_exp)
    # But for efficiency, we'll use a dict-based approach with a hash

    # Key insight: instead of computing full products (which can be huge),
    # we track exponent vectors. Two products are equal iff their exponent vectors match.
    # We can represent a vector as a frozenset of (prime, exp) where exp > 0.

    def vec_add(a, b):
        """Add two exponent vectors (as dicts or frozensets)."""
        # Convert to dict for addition
        d = dict(a)
        for p, e in b:
            d[p] = d.get(p, 0) + e
        # Remove zeros
        d = {p: e for p, e in d.items() if e > 0}
        return frozenset(d.items())

    # Precompute vec for each number
    num_fvec = {}
    for n in range(2, 129):
        num_fvec[n] = frozenset(num_vec[n])

    # Fast check: given a sequence (list of ints), check if all consecutive block products are distinct
    # We do this by tracking all suffix-product vectors from each starting position
    def is_valid_seq(seq):
        n = len(seq)
        seen = set()
        for i in range(n):
            # Build product vector from i to j for j = i, i+1, ...
            prod_vec = frozenset()  # empty vector = product 1
            for j in range(i, n):
                prod_vec = vec_add(prod_vec, num_fvec[seq[j]])
                if prod_vec in seen:
                    return False
                seen.add(prod_vec)
        return True

    # Faster: incremental check when appending x to seq
    # New products are those ending at the new position: seq[i]*...*seq[-1]*x for i=0..n
    # We need to check these against all existing products
    def check_append(seq, x):
        """Check if appending x to seq keeps all products distinct. Returns True if valid."""
        n = len(seq)
        # First, collect all existing products
        existing = set()
        for i in range(n):
            prod_vec = frozenset()
            for j in range(i, n):
                prod_vec = vec_add(prod_vec, num_fvec[seq[j]])
                existing.add(prod_vec)
        
        # Now check new products: those that include x (the last element)
        # These are seq[i]*...*seq[n-1]*x for i = 0, 1, ..., n
        # For i = n: just x
        # For i < n: seq[i]*...*seq[n-1]*x
        new_prods = []
        prod_vec = num_fvec[x]  # product of just x
        new_prods.append(prod_vec)
        
        for i in range(n - 1, -1, -1):
            prod_vec = vec_add(num_fvec[seq[i]], prod_vec)
            new_prods.append(prod_vec)
        
        # Check each new product against existing and other new products
        new_set = set()
        for pv in new_prods:
            if pv in existing:
                return False
            if pv in new_set:
                return False
            new_set.add(pv)
        return True

    # Even faster: maintain a running set of all products as we build the sequence
    # State: (seq, all_products_set)
    # When appending x, we only need to check the new products (those ending at x)
    # and add them to the set
    
    def greedy_build(order, all_products):
        """Build sequence greedily in given order, maintaining all_products set."""
        seq = []
        for x in order:
            if time.time() > deadline - 2:
                break
            # Check if x can be appended
            n = len(seq)
            # New products: x, seq[n-1]*x, seq[n-2]*seq[n-1]*x, ..., seq[0]*...*seq[n-1]*x
            prod_vec = num_fvec[x]
            valid = True
            if prod_vec in all_products:
                valid = False
            else:
                for i in range(n - 1, -1, -1):
                    prod_vec = vec_add(num_fvec[seq[i]], prod_vec)
                    if prod_vec in all_products:
                        valid = False
                        break
            if valid:
                # Add all new products to all_products
                # Recompute them (we lost track of which were new)
                prod_vec = num_fvec[x]
                all_products.add(prod_vec)
                for i in range(n - 1, -1, -1):
                    prod_vec = vec_add(num_fvec[seq[i]], prod_vec)
                    all_products.add(prod_vec)
                seq.append(x)
        return seq, all_products

    # Local search with swap and delete+insert
    def local_search(seq, max_iters=500):
        """Improve seq via local moves: delete one, insert one; or swap adjacent."""
        best = seq[:]
        best_len = len(best)
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 2:
                break
            
            # Compute all products for best
            all_prods = set()
            n = len(best)
            for i in range(n):
                pv = frozenset()
                for j in range(i, n):
                    pv = vec_add(pv, num_fvec[best[j]])
                    all_prods.add(pv)
            
            # Try: delete element at position i, then try to insert some unused x
            used = set(best)
            for i in range(len(best)):
                if time.time() > deadline - 2:
                    break
                trial = best[:i] + best[i+1:]
                # Recompute products for trial
                trial_prods = set()
                tn = len(trial)
                for a in range(tn):
                    pv = frozenset()
                    for b in range(a, tn):
                        pv = vec_add(pv, num_fvec[trial[b]])
                        trial_prods.add(pv)
                
                # Try inserting each unused x
                for x in range(2, 129):
                    if x in used:
                        continue
                    # Check if x can be appended to trial
                    pv = num_fvec[x]
                    if pv in trial_prods:
                        continue
                    ok = True
                    for k in range(tn - 1, -1, -1):
                        pv = vec_add(num_fvec[trial[k]], pv)
                        if pv in trial_prods:
                            ok = False
                            break
                    if ok:
                        new_seq = trial + [x]
                        if len(new_seq) > len(best):
                            best = new_seq
                            best_len = len(best)
                            improved = True
                            break
                if improved:
                    break
            
            # Try: swap adjacent elements
            if not improved:
                for i in range(len(best) - 1):
                    if time.time() > deadline - 2:
                        break
                    trial = best[:]
                    trial[i], trial[i+1] = trial[i+1], trial[i]
                    # Check validity
                    if is_valid_seq(trial):
                        best = trial
                        improved = True
                        break
        
        return best

    # Main: try multiple strategies
    best_seq = []
    
    # Strategy 1: Greedy in natural order
    seq1, prods1 = greedy_build(range(2, 129), set())
    if len(seq1) > len(best_seq):
        best_seq = seq1
    
    # Local search on strategy 1
    if time.time() < deadline - 5:
        improved = local_search(best_seq, max_iters=200)
        if len(improved) > len(best_seq):
            best_seq = improved
    
    # Strategy 2: Multiple random restarts
    rng = random.Random(12345)
    for restart in range(20):
        if time.time() > deadline - 3:
            break
        order = list(range(2, 129))
        rng.shuffle(order)
        seq_r, prods_r = greedy_build(order, set())
        if len(seq_r) > len(best_seq):
            best_seq = seq_r
        if time.time() < deadline - 5:
            improved = local_search(best_seq, max_iters=100)
            if len(improved) > len(best_seq):
                best_seq = improved
    
    # Strategy 3: Greedy with priority on primes first (primes have unique factorization)
    if time.time() < deadline - 5:
        primes = [n for n in range(2, 129) if all(n % d != 0 for d in range(2, int(n**0.5)+1))]
        composites = [n for n in range(2, 129) if n not in primes]
        order3 = primes + composites
        seq3, prods3 = greedy_build(order3, set())
        if len(seq3) > len(best_seq):
            best_seq = seq3
        if time.time() < deadline - 5:
            improved = local_search(best_seq, max_iters=100)
            if len(improved) > len(best_seq):
                best_seq = improved

    # Final verification
    if not is_valid_seq(best_seq):
        # Fallback: build a valid sequence from scratch
        best_seq, _ = greedy_build(range(2, 129), set())
    
    return best_seq
