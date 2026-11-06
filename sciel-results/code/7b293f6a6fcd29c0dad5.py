import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Precompute primes in [2, 128]
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    P = len(primes)

    # Precompute signatures for all numbers 2..128 as fixed-length tuples of exponents
    sigs_fixed = [None] * 129
    for i in range(2, 129):
        sig = [0] * P
        n = i
        for idx, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                sig[idx] += 1
                n //= p
        if n > 1:
            idx = prime_to_idx[n]
            sig[idx] += 1
        sigs_fixed[i] = tuple(sig)

    def add_sig(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def is_valid_addition(seq_sigs, x_sig, products_set):
        """
        Check if appending x_sig to seq_sigs is valid.
        Returns (is_valid, new_tails_sigs)
        """
        new_tails = [x_sig]
        current_prod = x_sig
        for s in reversed(seq_sigs):
            current_prod = add_sig(current_prod, s)
            new_tails.append(current_prod)

        for tail in new_tails:
            if tail in products_set:
                return False, None

        return True, new_tails

    def greedy_construct(skip_set):
        """
        Greedily construct a sequence by iterating 2..128 and including x if valid,
        unless x is in skip_set.
        """
        seq_vals = []
        seq_sigs = []
        products_set = set()

        for x in range(2, 129):
            if x in skip_set:
                continue

            x_sig = sigs_fixed[x]
            valid, new_tails = is_valid_addition(seq_sigs, x_sig, products_set)

            if valid:
                seq_vals.append(x)
                seq_sigs.append(x_sig)
                products_set.update(new_tails)

        return seq_vals

    def validate_sequence(seq):
        """Verify that a sequence is strictly increasing in [2,128] and all block products are distinct."""
        if not seq:
            return True
        if any(x < 2 or x > 128 for x in seq):
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        # Check all consecutive block products
        products = set()
        for i in range(len(seq)):
            prod_sig = [0] * P
            for j in range(i, len(seq)):
                for k in range(P):
                    prod_sig[k] += sigs_fixed[seq[j]][k]
                prod_tuple = tuple(prod_sig)
                if prod_tuple in products:
                    return False
                products.add(prod_tuple)
        return True

    def try_insert(seq, x):
        """Try to insert x into seq at any valid position. Returns new seq or None."""
        for pos in range(len(seq) + 1):
            if pos > 0 and seq[pos-1] >= x:
                continue
            if pos < len(seq) and seq[pos] <= x:
                continue
            
            candidate = seq[:pos] + [x] + seq[pos:]
            if validate_sequence(candidate):
                return candidate
        return None

    def polish_sequence(seq):
        """
        Iterative local search: try insertions, swaps, and deletions.
        """
        best = list(seq)
        best_len = len(best)
        n = len(best)
        improved = True
        
        while improved and time.time() < deadline - 2.0:
            improved = False
            
            # 1. Try single insertions
            for x in range(2, 129):
                if x in best:
                    continue
                candidate = try_insert(best, x)
                if candidate:
                    best = candidate
                    best_len += 1
                    n = len(best)
                    improved = True
                    if time.time() > deadline - 2.0:
                        return best

            # 2. Try swaps: remove element i, insert at position j
            # This is expensive, so we limit it or do it less frequently if time is tight
            if time.time() < deadline - 5.0:
                for i in range(len(best)):
                    x = best[i]
                    rest = best[:i] + best[i+1:]
                    # Try inserting x into 'rest' at various positions
                    # We already know x is not in rest
                    for pos in range(len(rest) + 1):
                        if pos > 0 and rest[pos-1] >= x:
                            continue
                        if pos < len(rest) and rest[pos] <= x:
                            continue
                        
                        # Skip if it's the same position (i.e., pos == i)
                        if pos == i:
                            continue
                            
                        candidate = rest[:pos] + [x] + rest[pos:]
                        if validate_sequence(candidate):
                            # Check if it's actually different from best (it should be)
                            if candidate != best:
                                best = candidate
                                improved = True
                                break # Move to next i
                    if time.time() > deadline - 2.0:
                        return best

        return best

    best_seq = []

    # Strategy 1: Standard Greedy (no skips)
    seq = greedy_construct(set())
    if len(seq) > len(best_seq):
        best_seq = seq

    # Strategy 2: Local Search / Hill Climbing on the "Skip Set"
    current_skip_set = set()
    current_seq = list(best_seq)
    current_len = len(current_seq)

    # Expand candidates to [2, 30]
    candidates = list(range(2, 31))

    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        for x in candidates:
            if time.time() > deadline - 10.0:
                break
            # Try adding x to skip set
            new_skip = current_skip_set | {x}
            new_seq = greedy_construct(new_skip)
            if len(new_seq) > current_len:
                current_skip_set = new_skip
                current_seq = new_seq
                current_len = len(new_seq)
                improved = True

            # Try removing x from skip set (if present)
            if x in current_skip_set:
                new_skip = current_skip_set - {x}
                new_seq = greedy_construct(new_skip)
                if len(new_seq) > current_len:
                    current_skip_set = new_skip
                    current_seq = new_seq
                    current_len = len(new_seq)
                    improved = True

    if len(current_seq) > len(best_seq):
        best_seq = current_seq

    # Strategy 3: Randomized Restart with Random Skip Sets
    random.seed(42)

    for _ in range(2000):
        if time.time() > deadline - 10.0:
            break

        skip_set = set()
        for x in range(2, 21):
            if x < 10:
                if random.random() < 0.5:
                    skip_set.add(x)
            elif x < 20:
                if random.random() < 0.2:
                    skip_set.add(x)

        seq = greedy_construct(skip_set)
        if len(seq) > len(best_seq):
            best_seq = seq

    # Strategy 4: More aggressive random search
    for _ in range(2000):
        if time.time() > deadline - 10.0:
            break

        skip_set = set()
        for x in range(2, 31):
            if x < 10:
                if random.random() < 0.6:
                    skip_set.add(x)
            elif x < 20:
                if random.random() < 0.3:
                    skip_set.add(x)
            else:
                if random.random() < 0.1:
                    skip_set.add(x)

        seq = greedy_construct(skip_set)
        if len(seq) > len(best_seq):
            best_seq = seq

    # Strategy 5: Polish the best sequence found
    if time.time() < deadline - 2.0:
        polished = polish_sequence(best_seq)
        if len(polished) > len(best_seq):
            best_seq = polished

    # Final validation
    if not validate_sequence(best_seq):
        # Fallback to a known valid greedy result
        best_seq = greedy_construct(set())

    return best_seq
