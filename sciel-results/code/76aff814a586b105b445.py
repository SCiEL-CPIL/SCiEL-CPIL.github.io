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
        """Verify that a sequence is strictly increasing in [2,128] and all block products are distinct.
        Uses prefix-sum signatures for O(n^2) total work instead of O(n^2 * P)."""
        if not seq:
            return True
        if any(x < 2 or x > 128 for x in seq):
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        n = len(seq)
        # Build prefix sum of signatures: prefix[i][k] = sum of exponents in seq[0..i-1]
        # Use a list of lists for speed
        prefix = [[0] * P for _ in range(n + 1)]
        for i in range(n):
            s = sigs_fixed[seq[i]]
            prev = prefix[i]
            cur = prefix[i + 1]
            for k in range(P):
                cur[k] = prev[k] + s[k]
        # Check all block products
        products = set()
        for i in range(n):
            pi = prefix[i]
            for j in range(i + 1, n + 1):
                pj = prefix[j]
                # Block product signature = prefix[j] - prefix[i]
                # Build tuple for set lookup
                prod_tuple = tuple(pj[k] - pi[k] for k in range(P))
                if prod_tuple in products:
                    return False
                products.add(prod_tuple)
        return True

    def polish_sequence(seq, deadline=None):
        """
        Try single insertions at all valid positions to improve the sequence.
        Uses the fast prefix-sum validator. Iterates until no more improvements.
        """
        if not seq:
            return seq

        best = list(seq)
        best_len = len(best)
        best_set = set(best)

        improved = True
        while improved:
            if deadline is not None and time.time() > deadline:
                break
            improved = False
            for x in range(2, 129):
                if x in best_set:
                    continue
                # Find valid insertion positions
                # x must go where best[pos-1] < x < best[pos]
                # Use binary search to find the range
                lo, hi = 0, len(best)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if best[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                # lo is the first index where best[lo] >= x
                # Valid positions are from lo (insert before best[lo]) 
                # but we need best[pos-1] < x and best[pos] > x (or boundaries)
                # Position lo: insert before best[lo] (or at end)
                # Position lo-1: insert after best[lo-1] (if lo-1 >= 0)
                # Actually, since the sequence is strictly increasing, x can only be inserted
                # at position lo (right after all elements < x, before all elements >= x)
                # But we also need to check if x could go at lo-1 if best[lo-1] < x
                # Since best is sorted, all elements at indices < lo are < x,
                # and all elements at indices >= lo are >= x.
                # So the only valid position is lo (between best[lo-1] and best[lo])
                # But we should also consider that skipping some elements might allow insertion
                # For single insertion, the position is uniquely determined: lo
                pos = lo
                if pos > 0 and best[pos - 1] >= x:
                    continue
                if pos < len(best) and best[pos] <= x:
                    continue
                candidate = best[:pos] + [x] + best[pos:]
                if validate_sequence(candidate):
                    best = candidate
                    best_set.add(x)
                    best_len = len(best)
                    improved = True

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
    while improved and time.time() < deadline - 5.0:
        improved = False
        for x in candidates:
            if time.time() > deadline - 5.0:
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

    for _ in range(3000):
        if time.time() > deadline - 5.0:
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
    for _ in range(3000):
        if time.time() > deadline - 5.0:
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

    # Strategy 5: Polish the best sequence found (fast validator enables many iterations)
    if time.time() < deadline - 2.0:
        polished = polish_sequence(best_seq, deadline=deadline - 1.0)
        if len(polished) > len(best_seq):
            best_seq = polished

    # Strategy 6: Iterative polish - try inserting one more element after each improvement
    # The fast validator makes this feasible
    if time.time() < deadline - 1.0:
        polished2 = polish_sequence(best_seq, deadline=deadline - 0.5)
        if len(polished2) > len(best_seq):
            best_seq = polished2

    # Final validation
    if not validate_sequence(best_seq):
        # Fallback to a known valid greedy result
        best_seq = greedy_construct(set())

    return best_seq
