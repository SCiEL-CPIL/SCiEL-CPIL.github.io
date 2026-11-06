import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    # All numbers in [2, 128]
    all_numbers = list(range(2, 129))
    primes = [p for p in all_numbers if is_prime(p)]
    composites = [c for c in all_numbers if not is_prime(c)]
    
    # Precompute prime factorization for all numbers
    # Vector of exponents for primes [2, 3, 5, ..., 127]
    prime_list = primes
    prime_index = {p: i for i, p in enumerate(prime_list)}
    factorizations = {}
    for n in all_numbers:
        vec = [0] * len(prime_list)
        temp = n
        for i, p in enumerate(prime_list):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        if temp > 1:
            # temp must be a prime in our list
            if temp in prime_index:
                vec[prime_index[temp]] += 1
        factorizations[n] = tuple(vec)
        
    def vec_add(v1, v2):
        return tuple(a + b for a, b in zip(v1, v2))
        
    def is_valid_fast(seq):
        """Check if all consecutive block products are distinct using exponent vectors."""
        products = set()
        n = len(seq)
        for i in range(n):
            curr_vec = [0] * len(prime_list)
            for j in range(i, n):
                # Add factorization of seq[j]
                f = factorizations[seq[j]]
                for k in range(len(prime_list)):
                    curr_vec[k] += f[k]
                prod_tuple = tuple(curr_vec)
                if prod_tuple in products:
                    return False
                products.add(prod_tuple)
        return True

    # Initial best: all primes
    best_seq = sorted(primes)
    best_len = len(best_seq)
    
    # Helper to greedily fill a sequence
    def greedy_fill(base_seq, candidates, order_key=None):
        current_seq = list(base_seq)
        # Sort candidates based on key
        if order_key:
            sorted_candidates = sorted(candidates, key=order_key)
        else:
            sorted_candidates = sorted(candidates)
            
        for c in sorted_candidates:
            if time.time() > deadline:
                break
            if c in current_seq:
                continue
                
            # Binary search for insertion point
            lo, hi = 0, len(current_seq)
            while lo < hi:
                mid = (lo + hi) // 2
                if current_seq[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            insert_pos = lo
            
            if insert_pos < len(current_seq) and current_seq[insert_pos] == c:
                continue
            
            new_seq = current_seq[:insert_pos] + [c] + current_seq[insert_pos:]
            
            if is_valid_fast(new_seq):
                current_seq = new_seq
                
        return current_seq

    # Strategy 1: Multiple Greedy Passes with different orderings
    # Order by number of distinct prime factors
    def num_distinct_prime_factors(n):
        count = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            count += 1
        return count
        
    # Pre-compute distinct factors for sorting
    distinct_factors = {c: num_distinct_prime_factors(c) for c in composites}
    
    # Trial 1: Sort by distinct factors (ascending)
    seq1 = greedy_fill(primes, composites, key=lambda x: distinct_factors[x])
    if len(seq1) > best_len:
        best_len = len(seq1)
        best_seq = seq1
        
    # Trial 2: Sort by distinct factors (descending)
    seq2 = greedy_fill(primes, composites, key=lambda x: -distinct_factors[x])
    if len(seq2) > best_len:
        best_len = len(seq2)
        best_seq = seq2
        
    # Trial 3: Sort by value
    seq3 = greedy_fill(primes, composites)
    if len(seq3) > best_len:
        best_len = len(seq3)
        best_seq = seq3
        
    # Strategy 2: Local Search / Destroy and Repair
    # We will try to improve the best_seq found so far
    current_best = list(best_seq)
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
            
        # Pick a random subset of elements to remove (destroy)
        # Remove 1 to 3 random elements
        num_remove = random.randint(1, 3)
        if len(current_best) <= num_remove:
            continue
            
        removed = random.sample(current_best, num_remove)
        remaining = [x for x in current_best if x not in removed]
        
        # Candidates are all composites not in remaining
        # We can also re-add the removed ones
        candidates = [c for c in composites if c not in remaining]
        
        # Randomize the order of candidates slightly to escape local optima
        # Or use a specific heuristic
        # Let's try a mix: some random, some heuristic
        if random.random() < 0.5:
             # Heuristic: sort by distinct factors
             ordered_candidates = sorted(candidates, key=lambda x: distinct_factors[x])
        else:
             # Random order
             ordered_candidates = candidates
             random.shuffle(ordered_candidates)
             
        new_seq = greedy_fill(remaining, ordered_candidates)
        
        if len(new_seq) > best_len:
            best_len = len(new_seq)
            best_seq = new_seq
            current_best = new_seq
            
    return sorted(best_seq)
