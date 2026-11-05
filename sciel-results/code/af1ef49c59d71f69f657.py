import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def get_prime_factorization(n):
        """Return dict of prime->exponent for n."""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def is_valid(s):
        """Check if all consecutive block products are distinct."""
        n = len(s)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= s[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def is_valid_fast(s):
        """Faster validity check with early product magnitude pruning."""
        n = len(s)
        if n == 0:
            return True
        prods = set()
        # Precompute prefix products to speed up block product computation
        # But for now, use early termination: if product exceeds a threshold,
        # it's unlikely to collide with earlier small products
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= s[j]
                if p in prods:
                    return False
                prods.add(p)
                # Early exit: if p is very large, remaining extensions will only be larger
                # and won't collide with existing smaller products
                # But we still need to check all for correctness
        return True

    all_numbers = list(range(2, 129))
    primes = [x for x in all_numbers if is_prime(x)]
    composites = [x for x in all_numbers if not is_prime(x)]
    
    best_seq = primes[:]
    best_len = len(primes)
    
    random.seed(42)
    
    # Multi-Start Greedy Strategy with multiple insertion order strategies
    
    def greedy_insert(seed, check_order_strategy, deadline):
        """
        Given a valid seed, try to insert remaining numbers.
        check_order_strategy: 'ascending', 'descending', 'random', 'middle_out'
        """
        seed = sorted(seed)
        if not is_valid(seed):
            return seed, len(seed)
            
        current_seq = seed[:]
        current_len = len(current_seq)
        current_set = set(current_seq)
        
        remaining = [x for x in all_numbers if x not in current_set]
        
        if check_order_strategy == 'ascending':
            check_order = remaining[:]
        elif check_order_strategy == 'descending':
            check_order = remaining[::-1]
        elif check_order_strategy == 'random':
            check_order = remaining[:]
            random.shuffle(check_order)
        elif check_order_strategy == 'middle_out':
            # Start from middle, expand outward
            mid = len(remaining) // 2
            check_order = []
            for d in range(mid + 1):
                if mid - d >= 0:
                    check_order.append(remaining[mid - d])
                if d > 0 and mid + d < len(remaining):
                    check_order.append(remaining[mid + d])
        else:
            check_order = remaining[:]
        
        for c in check_order:
            if time.time() > deadline - 5:
                break
                
            # Find position to insert c to keep sequence sorted
            pos = 0
            while pos < len(current_seq) and current_seq[pos] < c:
                pos += 1
            
            if pos < len(current_seq) and current_seq[pos] == c:
                continue
                
            new_seq = current_seq[:pos] + [c] + current_seq[pos:]
            
            if is_valid(new_seq):
                current_seq = new_seq
                current_set.add(c)
                current_len += 1
        
        return current_seq, current_len

    # Define multiple strategies to try
    strategies = [
        ('primes_asc', primes[:], 'ascending'),
        ('primes_desc', primes[:], 'descending'),
        ('primes_rand', primes[:], 'random'),
        ('small_asc', all_numbers[:20], 'ascending'),
        ('small_desc', all_numbers[:20], 'descending'),
        ('small_rand', all_numbers[:20], 'random'),
        ('small_mid', all_numbers[:20], 'middle_out'),
    ]
    
    # Add a random seed
    rand_seed = []
    for x in sorted(random.sample(all_numbers, 15)):
        test = rand_seed + [x]
        pos = 0
        while pos < len(test) and test[pos] < x:
            pos += 1
        test.insert(pos, x)
        if is_valid(test):
            rand_seed = test
        else:
            test.pop(pos)
    if rand_seed:
        strategies.append(('rand_asc', rand_seed[:], 'ascending'))
        strategies.append(('rand_desc', rand_seed[:], 'descending'))
        strategies.append(('rand_rand', rand_seed[:], 'random'))
    
    # Also try a "primes + selected composites" seed
    # Add composites that are products of two distinct primes (square-free composites)
    sq_free_composites = [x for x in composites if all(get_prime_factorization(x).get(p, 0) == 1 for p in get_prime_factorization(x))]
    seed_pf = primes[:] + sq_free_composites[:10]
    seed_pf.sort()
    if is_valid(seed_pf):
        strategies.append(('pf_asc', seed_pf[:], 'ascending'))
        strategies.append(('pf_desc', seed_pf[:], 'descending'))
    
    for name, seed, strategy in strategies:
        if time.time() > deadline - 10:
            break
            
        result, result_len = greedy_insert(seed, strategy, deadline)
        
        if result_len > best_len:
            best_len = result_len
            best_seq = result[:]
    
    # Second phase: local search / swap-based improvement
    # Try swapping elements or removing one and re-adding differently
    if time.time() < deadline - 15:
        # Try to improve by removing one element and trying to add two different ones
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_best = best_seq[:]
            current_best_set = set(current_best)
            
            # Try removing each element and see if we can add back two others
            for idx in range(len(current_best)):
                if time.time() > deadline - 10:
                    break
                removed = current_best[idx]
                without = current_best[:idx] + current_best[idx+1:]
                if not is_valid(without):
                    continue
                
                # Try to add two numbers not in current_best
                missing = [x for x in all_numbers if x not in current_best_set]
                # Try pairs
                for i in range(len(missing)):
                    if time.time() > deadline - 10:
                        break
                    for j in range(i+1, len(missing)):
                        if time.time() > deadline - 10:
                            break
                        candidate = without + [missing[i], missing[j]]
                        candidate.sort()
                        if is_valid(candidate) and len(candidate) > len(best_seq):
                            best_seq = candidate
                            best_len = len(candidate)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    
    return best_seq
