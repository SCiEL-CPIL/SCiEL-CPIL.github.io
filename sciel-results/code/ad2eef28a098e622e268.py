import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85

    def is_valid_full(s):
        """Full validation using prefix products."""
        n = len(s)
        if n == 0:
            return True
        pref = [1] * (n + 1)
        for i in range(n):
            pref[i + 1] = pref[i] * s[i]
        prods = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                p = pref[j] // pref[i]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def try_insert_incremental(seq, prods, c):
        """Try to insert c into seq at the correct position. Returns (new_seq, new_prods) or None."""
        n = len(seq)
        # Find position
        pos = 0
        while pos < n and seq[pos] < c:
            pos += 1
        if pos < n and seq[pos] == c:
            return None

        # Compute all new products that involve c
        # A block containing c: [i..j] where i <= pos <= j
        # Product = (product of seq[i..pos-1]) * c * (product of seq[pos+1..j])
        # We need to check all such products against existing prods and among themselves

        new_prods = set()
        # Products of blocks ending at c (i.e., [i..pos] for i in 0..pos)
        left_prod = 1
        for i in range(pos, -1, -1):
            left_prod *= seq[i] if i < pos else 1
            # Actually let me recompute: block [i..pos] = seq[i]*...*seq[pos-1]*c
            # Let me compute suffix products from pos-1 down to 0
            pass

        # Better approach: compute all new blocks containing c
        # Block [i..j] contains c iff i <= pos <= j
        # Product = pref[j+1]//pref[i] where pref is prefix of new_seq
        # But we can compute incrementally

        # Let's compute left suffix products: for i from pos down to 0:
        # L[i] = seq[i] * seq[i+1] * ... * seq[pos-1]
        # R[j] = seq[pos+1] * seq[pos+2] * ... * seq[j]
        # Block [i..j] product = L[i] * c * R[j]

        # L[0] = seq[0]*...*seq[pos-1], L[1] = seq[1]*...*seq[pos-1], ..., L[pos] = 1
        L = [1] * (pos + 1)
        for i in range(pos - 1, -1, -1):
            L[i] = L[i + 1] * seq[i]

        # R[k] for k = 0, 1, ..., n-pos represents product of first k elements to the right of pos
        # Elements to right: seq[pos], seq[pos+1], ..., seq[n-1]
        # R[0] = 1
        # R[1] = seq[pos]
        # R[2] = seq[pos] * seq[pos+1]
        # ...
        # R[n-pos] = seq[pos] * ... * seq[n-1]
        
        R = [1]
        for k in range(1, n - pos + 1):
            R.append(R[-1] * seq[pos + k - 1])

        # Now check all blocks containing c
        seen_new = set()
        for i in range(0, pos + 1):
            for k in range(0, n - pos + 1):
                p = L[i] * c * R[k]
                if p in prods or p in seen_new:
                    return None
                seen_new.add(p)

        # Also need to check that no two new products collide with each other (done via seen_new)
        # And no new product collides with existing (done via prods check)

        new_seq = seq[:pos] + [c] + seq[pos:]
        new_prods = prods | seen_new
        return (new_seq, new_prods)

    def greedy_from_start(order, deadline):
        """Build a sequence by greedily adding elements in the given order."""
        seq = []
        prods = set()
        for c in order:
            if time.time() > deadline - 5:
                break
            result = try_insert_incremental(seq, prods, c)
            if result is not None:
                seq, prods = result
        return seq

    def greedy_insert_repair(seq, prods, deadline):
        """Try to insert all remaining numbers using incremental checks."""
        current_seq = seq[:]
        current_prods = prods.copy()
        current_set = set(current_seq)
        all_numbers = list(range(2, 129))
        remaining = [x for x in all_numbers if x not in current_set]

        # Try inserting each remaining number
        for c in remaining:
            if time.time() > deadline - 5:
                break
            result = try_insert_incremental(current_seq, current_prods, c)
            if result is not None:
                current_seq, current_prods = result
                current_set.add(c)

        return current_seq, current_prods

    def local_search_swap(seq, prods, deadline):
        """Try removing one element and adding another to improve length."""
        current_seq = seq[:]
        current_prods = prods.copy()
        current_set = set(current_seq)
        all_numbers = list(range(2, 129))
        improved = True

        while improved and time.time() < deadline - 8:
            improved = False
            missing = [x for x in all_numbers if x not in current_set]

            # Try adding a missing element
            for add_val in missing:
                if time.time() > deadline - 8:
                    break
                result = try_insert_incremental(current_seq, current_prods, add_val)
                if result is not None:
                    current_seq, current_prods = result
                    current_set.add(add_val)
                    improved = True
                    break

            # If no simple addition works, try remove+add
            if not improved:
                for idx in range(len(current_seq)):
                    if time.time() > deadline - 8:
                        break
                    removed_val = current_seq[idx]
                    without = current_seq[:idx] + current_seq[idx + 1:]
                    # Recompute prods for 'without' - this is expensive
                    # Skip for now; only do simple additions
                    break

        return current_seq, current_prods

    # --- Multi-start greedy with different orderings ---
    best_seq = []
    best_len = 0
    best_prods = set()

    # Start 1: Natural order
    seq1 = greedy_from_start(list(range(2, 129)), deadline)
    if len(seq1) > best_len:
        best_seq = seq1
        best_len = len(seq1)
        best_prods = set()
        # Recompute prods for best_seq
        n = len(best_seq)
        pref = [1] * (n + 1)
        for i in range(n):
            pref[i + 1] = pref[i] * best_seq[i]
        for i in range(n):
            for j in range(i + 1, n + 1):
                best_prods.add(pref[j] // pref[i])

    # Start 2: Reverse order (largest first, but we need increasing, so this means we try to build from high values)
    # Actually for increasing sequences, we can only add in increasing order of value.
    # So different orderings mean: which subset to include, in what priority.
    # Let's try: prioritize smaller numbers first vs larger numbers first
    # Actually the sequence must be increasing, so the order is determined by values.
    # The "ordering" is really which values to include.

    # Start 3: Randomized greedy - skip some values randomly
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        # Create a random permutation of the available numbers
        nums = list(range(2, 129))
        random.shuffle(nums)
        # But we must insert in increasing order of value to maintain increasing property
        # So the "order" is really the priority: which values to try first
        # Since the sequence must be increasing, we can only add values in increasing order.
        # So we need a different strategy: which values to SKIP.

        # Strategy: randomly decide to skip each value with some probability
        skip_prob = 0.1 + 0.3 * (trial / 50.0)
        seq = []
        prods = set()
        for c in range(2, 129):
            if time.time() > deadline - 10:
                break
            if random.random() < skip_prob:
                continue
            result = try_insert_incremental(seq, prods, c)
            if result is not None:
                seq, prods = result

        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)
            best_prods = prods.copy()

    # Start 4: Greedy with bias toward prime numbers (primes have fewer multiplicative relations)
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [x for x in range(2, 129) if is_prime(x)]
    composites = [x for x in range(2, 129) if not is_prime(x)]

    # Try: primes first, then composites
    seq_p = []
    prods_p = set()
    for c in primes + composites:
        if time.time() > deadline - 10:
            break
        result = try_insert_incremental(seq_p, prods_p, c)
        if result is not None:
            seq_p, prods_p = result
    if len(seq_p) > best_len:
        best_seq = seq_p
        best_len = len(seq_p)
        best_prods = prods_p.copy()

    # Try: composites first, then primes
    seq_c = []
    prods_c = set()
    for c in composites + primes:
        if time.time() > deadline - 10:
            break
        result = try_insert_incremental(seq_c, prods_c, c)
        if result is not None:
            seq_c, prods_c = result
    if len(seq_c) > best_len:
        best_seq = seq_c
        best_len = len(seq_c)
        best_prods = prods_c.copy()

    # Start 5: Numbers with many prime factors first (more "constrained" early)
    def num_prime_factors(n):
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

    by_factors = sorted(range(2, 129), key=lambda x: -num_prime_factors(x))
    seq_f = []
    prods_f = set()
    for c in by_factors:
        if time.time() > deadline - 10:
            break
        result = try_insert_incremental(seq_f, prods_f, c)
        if result is not None:
            seq_f, prods_f = result
    if len(seq_f) > best_len:
        best_seq = seq_f
        best_len = len(seq_f)
        best_prods = prods_f.copy()

    # Start 6: Numbers with few prime factors first
    by_few_factors = sorted(range(2, 129), key=lambda x: num_prime_factors(x))
    seq_ff = []
    prods_ff = set()
    for c in by_few_factors:
        if time.time() > deadline - 10:
            break
        result = try_insert_incremental(seq_ff, prods_ff, c)
        if result is not None:
            seq_ff, prods_ff = result
    if len(seq_ff) > best_len:
        best_seq = seq_ff
        best_len = len(seq_ff)
        best_prods = prods_ff.copy()

    # Phase 2: Insertion repair on the best sequence
    if time.time() < deadline - 8:
        repaired_seq, repaired_prods = greedy_insert_repair(best_seq, best_prods, deadline)
        if len(repaired_seq) > best_len:
            best_seq = repaired_seq
            best_len = len(repaired_seq)
            best_prods = repaired_prods

    # Phase 3: Local search - try to add more elements
    if time.time() < deadline - 8:
        ls_seq, ls_prods = local_search_swap(best_seq, best_prods, deadline)
        if len(ls_seq) > best_len:
            best_seq = ls_seq
            best_len = len(ls_seq)
            best_prods = ls_prods

    # Final validation
    if not is_valid_full(best_seq):
        # Fallback
        best_seq = []
        for x in range(2, 129):
            result = try_insert_incremental(best_seq, set(), x)
            if result is not None:
                best_seq = result[0]
        # Recompute prods
        n = len(best_seq)
        best_prods = set()
        if n > 0:
            pref = [1] * (n + 1)
            for i in range(n):
                pref[i + 1] = pref[i] * best_seq[i]
            for i in range(n):
                for j in range(i + 1, n + 1):
                    best_prods.add(pref[j] // pref[i])

    return best_seq
