"""Stronger program: backtracking search with exponent-vector conflict detection."""
import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Precompute prime factorizations for 2..128
    # Primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # For each number 2..128, store its exponent vector as a tuple
    exponent_vectors = {}
    for n in range(2, 129):
        vec = [0] * len(primes)
        temp = n
        for idx, p in enumerate(primes):
            while temp % p == 0:
                vec[idx] += 1
                temp //= p
        # If temp > 1, it's a prime > 23 (i.e., 29, 31, ... but those don't exist <= 128 that aren't in our list)
        # Actually 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127
        # These are primes, so they'd have temp = p > 1. But since they're single prime powers, we need to handle them.
        # Let me extend the prime list to include all primes up to 128
        if temp > 1:
            # temp is a prime factor not in our list
            pass  # We'll handle this differently
    
    # Let me redo: include all primes up to 128
    def get_primes_up_to(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    all_primes = get_primes_up_to(128)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}
    num_primes = len(all_primes)
    
    exponent_vectors = {}
    for n in range(2, 129):
        vec = [0] * num_primes
        temp = n
        for idx, p in enumerate(all_primes):
            while temp % p == 0:
                vec[idx] += 1
                temp //= p
        exponent_vectors[n] = tuple(vec)
    
    # For a sequence, a consecutive block product's exponent vector is the sum of exponent vectors of elements in the block.
    # Two blocks have the same product iff their exponent vectors are equal.
    
    # We need: for any two distinct nonempty consecutive blocks, their product exponent vectors must differ.
    # Equivalently, all block product exponent vectors must be distinct.
    
    # Key insight: if we maintain the current sequence, adding a new element x at the end,
    # we need to check that all new blocks ending at x have products not seen before,
    # AND that these new products don't collide with each other.
    
    # But also, blocks not ending at x already had distinct products among themselves.
    # However, a new block ending at x could have the same product as an old block not ending at x.
    
    # So when adding x to the end:
    # - New blocks: suffixes of the extended sequence that include x
    # - These are: [x], [last, x], [second_last, last, x], ..., [whole_sequence, x]
    # - We need none of these to collide with any existing block product OR with each other.
    
    # We can maintain a set of all block product exponent vectors (as tuples).
    # When adding x, compute the new block vectors and check for collisions.
    
    def is_valid_seq(seq):
        """Full validation of a sequence."""
        if not seq:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        n = len(seq)
        products = set()
        for i in range(n):
            vec_sum = [0] * num_primes
            for j in range(i, n):
                ev = exponent_vectors[seq[j]]
                for k in range(num_primes):
                    vec_sum[k] += ev[k]
                key = tuple(vec_sum)
                if key in products:
                    return False
                products.add(key)
        return True
    
    def greedy_construct():
        """Greedy: try numbers 2..128 in order, adding if valid."""
        seq = []
        products = set()  # set of exponent vector tuples for all block products
        
        for x in range(2, 129):
            # Check if we can append x
            # New blocks are suffixes including x
            new_blocks = []
            vec_sum = [0] * num_primes
            for i in range(len(seq) - 1, -1, -1):
                ev = exponent_vectors[seq[i]]
                for k in range(num_primes):
                    vec_sum[k] += ev[k]
                new_blocks.append(tuple(vec_sum))
            # Add x itself
            new_blocks.append(tuple(exponent_vectors[x]))
            
            # Check collisions
            valid = True
            for nb in new_blocks:
                if nb in products:
                    valid = False
                    break
            if not valid:
                continue
            # Check new blocks against each other
            if len(set(new_blocks)) != len(new_blocks):
                continue
            
            # Valid: add x
            seq.append(x)
            for nb in new_blocks:
                products.add(nb)
        
        return seq
    
    # Initial greedy
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # Backtracking search: build sequence incrementally, try to extend
    # Use depth-first search with pruning
    
    def backtrack(seq, products, remaining):
        """
        Try to extend seq by adding elements from remaining (in increasing order).
        Returns the best extension found.
        """
        if time.time() - start_time > DEADLINE:
            return seq, products
        
        current_len = len(seq)
        best_here = (seq[:], set(products))
        
        for i, x in enumerate(remaining):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try adding x to the end of seq
            # Compute new block vectors
            new_blocks = []
            vec_sum = [0] * num_primes
            for j in range(len(seq) - 1, -1, -1):
                ev = exponent_vectors[seq[j]]
                for k in range(num_primes):
                    vec_sum[k] += ev[k]
                new_blocks.append(tuple(vec_sum))
            new_blocks.append(tuple(exponent_vectors[x]))
            
            # Check against existing products
            valid = True
            for nb in new_blocks:
                if nb in products:
                    valid = False
                    break
            if not valid:
                continue
            # Check among new_blocks
            if len(set(new_blocks)) != len(new_blocks):
                continue
            
            # Valid: recurse
            new_products = products.copy()
            for nb in new_blocks:
                new_products.add(nb)
            
            new_seq = seq + [x]
            # Remaining after x: elements > x not yet used
            # Since we're going in increasing order, remaining is already sorted
            new_remaining = remaining[i+1:]
            
            extended_seq, _ = backtrack(new_seq, new_products, new_remaining)
            
            if len(extended_seq) > len(best_here[0]):
                best_here = (extended_seq[:], set(products))
            
            # If we found a very long sequence, return early
            if len(best_here[0]) > best_len + 5:
                break
        
        return best_here
    
    # Full backtracking from empty
    full_result = backtrack([], set(), list(range(2, 129)))
    if len(full_result[0]) > best_len:
        best_seq = full_result[0]
        best_len = best_len if False else len(full_result[0])
    
    # Local search: try to insert or swap elements
    used = set(best_seq)
    remaining = [x for x in range(2, 129) if x not in used]
    
    def try_insert(seq, new_elem, pos):
        """Try inserting new_elem at position pos."""
        if pos < 0 or pos > len(seq):
            return None
        test_seq = seq[:pos] + [new_elem] + seq[pos:]
        # Check strictly increasing
        for i in range(len(test_seq) - 1):
            if test_seq[i] >= test_seq[i+1]:
                return None
        # Check range
        if any(x < 2 or x > 128 for x in test_seq):
            return None
        # Check block products
        if not is_valid_seq(test_seq):
            return None
        return test_seq
    
    # Iterative improvement
    max_rounds = 20
    for _ in range(max_rounds):
        if time.time() - start_time > DEADLINE:
            break
        
        improved = False
        for new_elem in remaining[:]:
            if time.time() - start_time > DEADLINE:
                break
            for pos in range(len(best_seq) + 1):
                result = try_insert(best_seq, new_elem, pos)
                if result is not None:
                    best_seq = result
                    used.add(new_elem)
                    remaining.remove(new_elem)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Try swaps
    for _ in range(10):
        if time.time() - start_time > DEADLINE:
            break
        swapped = False
        for i in range(len(best_seq)):
            if time.time() - start_time > DEADLINE:
                break
            old_val = best_seq[i]
            for new_val in remaining:
                if time.time() - start_time > DEADLINE:
                    break
                test_seq = best_seq[:]
                test_seq[i] = new_val
                test_seq.sort()
                if is_valid_seq(test_seq):
                    # Try to insert old_val
                    for pos in range(len(test_seq) + 1):
                        result = try_insert(test_seq, old_val, pos)
                        if result is not None:
                            best_seq = result
                            used.discard(new_val)
                            used.add(old_val)
                            if new_val not in remaining:
                                remaining.append(new_val)
                            if old_val in remaining:
                                remaining.remove(old_val)
                            swapped = True
                            break
                if swapped:
                    break
            if swapped:
                break
        if not swapped:
            break
    
    # Final validation
    if not is_valid_seq(best_seq):
        best_seq = greedy_construct()
    
    return best_seq
