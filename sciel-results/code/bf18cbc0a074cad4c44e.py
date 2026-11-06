import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    P = len(primes)

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

    # Precompute all pairwise signature sums for speed
    # sig_sum[i][j] = sigs_fixed[i] + sigs_fixed[j] for i, j in 2..128
    sig_sum = {}
    for i in range(2, 129):
        for j in range(i, 129):
            sig_sum[(i, j)] = tuple(x + y for x, y in zip(sigs_fixed[i], sigs_fixed[j]))

    def add_sig(a, b):
        return tuple(x + y for x, y in zip(a, b))

    def is_valid_addition(seq_sigs, x_sig, products_set):
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

    # Strategy 1: Standard Greedy
    seq = greedy_construct(set())
    if len(seq) > len(best_seq):
        best_seq = seq

    # Strategy 2: Local Search on Skip Set with broader candidates
    current_skip_set = set()
    current_seq = list(best_seq)
    current_len = len(current_seq)
    candidates = list(range(2, 31))

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in candidates:
            new_skip = current_skip_set | {x}
            new_seq = greedy_construct(new_skip)
            if len(new_seq) > current_len:
                current_skip_set = new_skip
                current_seq = new_seq
                current_len = len(new_seq)
                improved = True
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

    # Strategy 3: Randomized restarts with varied skip probabilities
    random.seed(42)
    
    for trial in range(3000):
        if time.time() > deadline:
            break
        
        # Vary the strategy: different ranges and probabilities
        strategy = trial % 5
        skip_set = set()
        
        if strategy == 0:
            # Light skipping of small numbers
            for x in range(2, 15):
                if random.random() < 0.3:
                    skip_set.add(x)
        elif strategy == 1:
            # Medium skipping
            for x in range(2, 25):
                if random.random() < 0.25:
                    skip_set.add(x)
        elif strategy == 2:
            # Skip specific conflict-prone numbers
            conflict_prone = [2, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28]
            for x in conflict_prone:
                if random.random() < 0.4:
                    skip_set.add(x)
        elif strategy == 3:
            # Skip multiples of 2 or 3 in a pattern
            for x in range(2, 40):
                if x % 2 == 0 and random.random() < 0.3:
                    skip_set.add(x)
                elif x % 3 == 0 and x % 2 != 0 and random.random() < 0.2:
                    skip_set.add(x)
        else:
            # Heavy random skipping
            for x in range(2, 50):
                if random.random() < 0.15:
                    skip_set.add(x)
        
        seq = greedy_construct(skip_set)
        if len(seq) > len(best_seq):
            best_seq = seq

    def verify_sequence(sig_seq):
        """Check product distinctness using precomputed prime signatures (fast)."""
        n = len(sig_seq)
        prod_set = set()
        for j in range(n):
            prod = (0,) * P
            for k in range(j, n):
                prod = tuple(prod[d] + sig_seq[k][d] for d in range(P))
                if prod in prod_set:
                    return False
                prod_set.add(prod)
        return True

    # Strategy 4: Local search on the best sequence found - try swapping elements
    # For each position, try replacing the element with a nearby number
    if time.time() < deadline:
        best_copy = list(best_seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_copy)):
                orig_val = best_copy[i]
                # Try replacing with values near orig_val
                for delta in [-3, -2, -1, 1, 2, 3, -5, 5]:
                    new_val = orig_val + delta
                    if new_val < 2 or new_val > 128:
                        continue
                    # Check if new_val is already in the sequence
                    if new_val in best_copy:
                        continue
                    # Build candidate sequence
                    candidate = best_copy.copy()
                    candidate[i] = new_val
                    # Check if still strictly increasing
                    valid_inc = True
                    for j in range(len(candidate) - 1):
                        if candidate[j] >= candidate[j + 1]:
                            valid_inc = False
                            break
                    if not valid_inc:
                        continue
                    # Check product distinctness using signatures (fast)
                    cand_sigs = [sigs_fixed[v] for v in candidate]
                    if verify_sequence(cand_sigs) and len(candidate) > len(best_seq):
                        best_seq = candidate
                        best_copy = candidate.copy()
                        improved = True
                        break
                if improved:
                    break

    # Strategy 4b: Deletion-based local search - remove one element, re-greedy
    if time.time() < deadline:
        base_seq = list(best_seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(base_seq)):
                if time.time() > deadline:
                    break
                # Build skip set: all elements in base_seq except position i
                skip_set = set(base_seq[:i] + base_seq[i+1:])
                new_seq = greedy_construct(skip_set)
                if len(new_seq) > len(best_seq):
                    best_seq = new_seq
                    base_seq = list(new_seq)
                    improved = True
                    break

    # Strategy 5: More aggressive random search with different seeds
    for seed in [123, 456, 789, 1011, 2023]:
        if time.time() > deadline:
            break
        random.seed(seed)
        for _ in range(500):
            if time.time() > deadline:
                break
            skip_set = set()
            for x in range(2, 35):
                if random.random() < 0.2:
                    skip_set.add(x)
            seq = greedy_construct(skip_set)
            if len(seq) > len(best_seq):
                best_seq = seq

    # Strategy 6: Final polish - try single-element insertions
    if time.time() < deadline:
        best_copy = list(best_seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_copy) + 1):
                # Try inserting a new element at position i
                for new_val in range(2, 129):
                    if new_val in best_copy:
                        continue
                    # Check if insertion maintains strictly increasing order
                    if i > 0 and new_val <= best_copy[i - 1]:
                        continue
                    if i < len(best_copy) and new_val >= best_copy[i]:
                        continue
                    # Build candidate sequence
                    candidate = best_copy.copy()
                    candidate.insert(i, new_val)
                    # Check product distinctness using signatures (fast)
                    cand_sigs = [sigs_fixed[v] for v in candidate]
                    if verify_sequence(cand_sigs) and len(candidate) > len(best_seq):
                        best_seq = candidate
                        best_copy = candidate.copy()
                        improved = True
                        break
                if improved:
                    break

    return best_seq
