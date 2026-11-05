"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute prime factorizations for numbers 2..128
    # Use a fixed list of primes up to 128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def factorize(n):
        """Return a tuple of exponents for each prime in the fixed prime list."""
        exponents = [0] * num_primes
        for i, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                exponents[i] += 1
                n //= p
        if n > 1:
            # n is a prime > sqrt(original_n), find its index
            if n in prime_to_idx:
                exponents[prime_to_idx[n]] += 1
        return tuple(exponents)
    
    # Precompute factorizations as tuples of exponents
    factor_cache = {}
    for i in range(2, 129):
        factor_cache[i] = factorize(i)
    
    def add_vecs(v1, v2):
        """Add two exponent vectors element-wise."""
        return tuple(a + b for a, b in zip(v1, v2))
    
    def compute_tails_from_tuple(seq_tuple, x):
        """Compute all consecutive-block product factor vectors ending at x.
        Returns list of exponent vectors."""
        tails = []
        vec = factor_cache[x]
        tails.append(vec)
        for a in reversed(seq_tuple):
            vec = add_vecs(vec, factor_cache[a])
            tails.append(vec)
        return tails
    
    def check_collision(tails, product_set):
        """Check if any tail vector is in product_set."""
        for t in tails:
            if t in product_set:
                return True
        return False
    
    # Greedy baseline
    def run_greedy():
        values = []
        product_set = set()
        for x in range(2, 129):
            if time.time() > deadline:
                break
            seq_tuple = tuple(values)
            tails = compute_tails_from_tuple(seq_tuple, x)
            if not check_collision(tails, product_set):
                values.append(x)
                product_set.update(tails)
        return values, product_set
    
    best_seq, best_prods = run_greedy()
    best_len = len(best_seq)
    
    # Beam search with better pruning
    beam = {(): set()}
    
    for x in range(2, 129):
        if time.time() > deadline - 5:
            break
        
        next_beam = {}
        
        for seq, prods in beam.items():
            # Option 1: Skip x
            if seq not in next_beam:
                next_beam[seq] = prods
            
            # Option 2: Include x
            tails = compute_tails_from_tuple(seq, x)
            
            if not check_collision(tails, prods):
                new_seq = seq + (x,)
                new_prods = set(prods)
                new_prods.update(tails)
                next_beam[new_seq] = new_prods
        
        # Prune beam: keep top W by length, then by smaller sum (more room for future)
        if next_beam:
            sorted_seqs = sorted(next_beam.keys(), key=lambda s: (-len(s), sum(s) if s else 0))
            W = 300  # Reduced for speed
            beam = {s: next_beam[s] for s in sorted_seqs[:W]}
            
            # Update best
            if sorted_seqs and len(sorted_seqs[0]) > best_len:
                best_len = len(sorted_seqs[0])
                best_seq = list(sorted_seqs[0])
                best_prods = next_beam[sorted_seqs[0]]
        else:
            break
    
    # Hill-climbing with replace moves
    if time.time() < deadline - 10:
        current = list(best_seq)
        
        def compute_all_products(seq):
            """Compute set of all consecutive block product vectors for seq."""
            prods = set()
            n = len(seq)
            for i in range(n):
                vec = ()
                for j in range(i, n):
                    vec = add_vecs(vec, factor_cache[seq[j]])
                    prods.add(vec)
            return prods
        
        def is_valid_sequence(seq):
            """Check if sequence is strictly increasing in [2,128] with distinct block products."""
            if len(seq) == 0:
                return True
            if seq[0] < 2 or seq[-1] > 128:
                return False
            for i in range(len(seq) - 1):
                if seq[i] >= seq[i + 1]:
                    return False
            prods = set()
            n = len(seq)
            for i in range(n):
                vec = ()
                for j in range(i, n):
                    vec = add_vecs(vec, factor_cache[seq[j]])
                    if vec in prods:
                        return False
                    prods.add(vec)
            return True
        
        current_prods = compute_all_products(current)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try replacing each element with a different value
            for pos in range(len(current)):
                if time.time() > deadline - 3:
                    break
                old_val = current[pos]
                
                # Try all possible replacements in order of preference
                # Prefer values close to old_val to maintain local structure
                candidates = []
                for new_val in range(2, 129):
                    if new_val == old_val:
                        continue
                    # Check bounds
                    if pos > 0 and new_val <= current[pos - 1]:
                        continue
                    if pos < len(current) - 1 and new_val >= current[pos + 1]:
                        continue
                    candidates.append(new_val)
                
                # Sort candidates by distance from old_val (prefer small changes)
                candidates.sort(key=lambda v: abs(v - old_val))
                
                for new_val in candidates:
                    if time.time() > deadline - 3:
                        break
                    
                    test = list(current)
                    test[pos] = new_val
                    
                    # Quick check: is it strictly increasing?
                    valid_increasing = True
                    for i in range(len(test) - 1):
                        if test[i] >= test[i + 1]:
                            valid_increasing = False
                            break
                    if not valid_increasing:
                        continue
                    
                    # Check product distinctness
                    test_prods = set()
                    n = len(test)
                    valid = True
                    for i in range(n):
                        vec = ()
                        for j in range(i, n):
                            vec = add_vecs(vec, factor_cache[test[j]])
                            if vec in test_prods:
                                valid = False
                                break
                            test_prods.add(vec)
                        if not valid:
                            break
                    
                    if valid:
                        # Accept the move
                        current = test
                        current_prods = test_prods
                        improved = True
                        break  # Break inner loop, continue outer
        
        best_seq = current
        best_len = len(current)
    
    # Try to extend the sequence by adding elements at the end
    if time.time() < deadline - 5:
        current = list(best_seq)
        current_prods = compute_all_products(current)
        
        # Try adding elements in increasing order from the end
        last_val = current[-1] if current else 1
        for x in range(last_val + 1, 129):
            if time.time() > deadline - 3:
                break
            seq_tuple = tuple(current)
            tails = compute_tails_from_tuple(seq_tuple, x)
            if not check_collision(tails, current_prods):
                current.append(x)
                current_prods.update(tails)
        
        if len(current) > best_len:
            best_seq = current
            best_len = len(current)
    
    return best_seq
# EVOLVE-BLOCK-END
