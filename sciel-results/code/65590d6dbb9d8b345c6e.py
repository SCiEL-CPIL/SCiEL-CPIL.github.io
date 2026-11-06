import time
import random
import sys

def construct():
    start_time = time.time()
    # Keep a safety margin before the 90s wall-clock limit (contract says 60s; use 50s to be safe)
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

    best_seq = []

    # Strategy 1: Standard Greedy (no skips)
    seq = greedy_construct(set())
    if len(seq) > len(best_seq):
        best_seq = seq

    # Strategy 2: Local Search / Hill Climbing on the "Skip Set"
    # We try to improve the sequence by toggling skips for numbers in [2, 20].
    # This is a form of local search where the "state" is the skip set.

    # Initial state: the skip set that produced best_seq (which is empty for now)
    current_skip_set = set()
    current_seq = list(best_seq)
    current_len = len(current_seq)

    # Numbers to consider for skipping
    candidates = list(range(2, 21))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates:
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
    # We generate random skip sets for small numbers and run greedy.
    # This helps escape local optima of the hill climbing.

    random.seed(42)  # Deterministic for reproducibility

    for _ in range(2000):
        if time.time() > deadline:
            break

        # Randomly select a set of numbers to skip from [2, 20]
        skip_set = set()
        for x in range(2, 21):
            # Probability of skipping: higher for smaller numbers
            # p = 0.5 * (1 - (x-2)/18) ? Let's just use a uniform random with bias
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
    # Try skipping more numbers, including up to 30.
    for _ in range(2000):
        if time.time() > deadline:
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

    # Final check: ensure we return a valid sequence
    # (The greedy construct ensures validity, so this is just a sanity check)
    return best_seq
