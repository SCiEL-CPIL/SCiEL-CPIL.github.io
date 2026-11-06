import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

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
        # Precompute suffix sums: suffix[k] = sum of sigs from index k to end
        n = len(seq_sigs)
        if n == 0:
            new_tails = [x_sig]
            if new_tails[0] in products_set:
                return False, None
            return True, new_tails
        suffix = [None] * (n + 1)
        suffix[n] = (0,) * P
        for k in range(n - 1, -1, -1):
            suffix[k] = add_sig(seq_sigs[k], suffix[k + 1])
        new_tails = [x_sig]
        for k in range(n):
            new_tails.append(add_sig(x_sig, suffix[k]))
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
        # 1-element moves
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
        # 2-element moves (add two at once)
        if not improved and time.time() < deadline:
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if time.time() > deadline:
                        break
                    x, y = candidates[i], candidates[j]
                    new_skip = current_skip_set | {x, y}
                    new_seq = greedy_construct(new_skip)
                    if len(new_seq) > current_len:
                        current_skip_set = new_skip
                        current_seq = new_seq
                        current_len = len(new_seq)
                        improved = True
                        break
                if improved:
                    break

    if len(current_seq) > len(best_seq):
        best_seq = current_seq

    # Strategy 3: Randomized restarts with varied skip probabilities
    random.seed(42)
    
    for trial in range(5000):
        if time.time() > deadline:
            break
        
        # Vary the strategy: different ranges and probabilities
        strategy = trial % 7
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
        elif strategy == 4:
            # Heavy random skipping
            for x in range(2, 50):
                if random.random() < 0.15:
                    skip_set.add(x)
        elif strategy == 5:
            # Skip numbers with many small prime factors (high "conflict potential")
            for x in range(2, 30):
                small_pf = sum(1 for p in primes if p <= 11 and x % p == 0)
                if small_pf >= 2 and random.random() < 0.5:
                    skip_set.add(x)
        else:
            # Random permutation-based: skip first k in a random order of small numbers
            small_nums = list(range(2, 20))
            random.shuffle(small_nums)
            k = random.randint(2, 6)
            skip_set = set(small_nums[:k])
        
        seq = greedy_construct(skip_set)
        if len(seq) > len(best_seq):
            best_seq = seq

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
                for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, -7, 7]:
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
                    # Check product distinctness using signatures
                    cand_sigs = [sigs_fixed[v] for v in candidate]
                    prod_set = set()
                    valid_prod = True
                    for j in range(len(candidate)):
                        cur = (0,) * P
                        for k in range(j, len(candidate)):
                            cur = add_sig(cur, cand_sigs[k])
                            if cur in prod_set:
                                valid_prod = False
                                break
                            prod_set.add(cur)
                        if not valid_prod:
                            break
                    if valid_prod and len(candidate) > len(best_seq):
                        best_seq = candidate
                        best_copy = candidate.copy()
                        improved = True
                        break
                if improved:
                    break

    # Strategy 5: More aggressive random search with different seeds
    for seed in [123, 456, 789, 1011, 2023, 3141, 2718, 1618]:
        if time.time() > deadline:
            break
        random.seed(seed)
        for _ in range(800):
            if time.time() > deadline:
                break
            # Vary skip pattern per iteration
            pat = random.randint(0, 3)
            skip_set = set()
            if pat == 0:
                for x in range(2, 35):
                    if random.random() < 0.2:
                        skip_set.add(x)
            elif pat == 1:
                for x in range(2, 25):
                    if random.random() < 0.3:
                        skip_set.add(x)
            elif pat == 2:
                for x in range(2, 40):
                    if (x % 2 == 0) and random.random() < 0.35:
                        skip_set.add(x)
            else:
                nums = list(range(2, 22))
                random.shuffle(nums)
                k = random.randint(3, 8)
                skip_set = set(nums[:k])
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
                    # Check product distinctness using signatures
                    cand_sigs = [sigs_fixed[v] for v in candidate]
                    prod_set = set()
                    valid_prod = True
                    for j in range(len(candidate)):
                        cur = (0,) * P
                        for k in range(j, len(candidate)):
                            cur = add_sig(cur, cand_sigs[k])
                            if cur in prod_set:
                                valid_prod = False
                                break
                            prod_set.add(cur)
                        if not valid_prod:
                            break
                    if valid_prod and len(candidate) > len(best_seq):
                        best_seq = candidate
                        best_copy = candidate.copy()
                        improved = True
                        break
                if improved:
                    break

    # Strategy 7: Replace-move polish - swap one element for another to fix conflicts
    if time.time() < deadline:
        best_copy = list(best_seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_copy)):
                orig = best_copy[i]
                for new_val in range(2, 129):
                    if new_val == orig or new_val in best_copy:
                        continue
                    candidate = best_copy.copy()
                    candidate[i] = new_val
                    # Check strictly increasing
                    valid_inc = True
                    for j in range(len(candidate) - 1):
                        if candidate[j] >= candidate[j + 1]:
                            valid_inc = False
                            break
                    if not valid_inc:
                        continue
                    # Check product distinctness
                    cand_sigs = [sigs_fixed[v] for v in candidate]
                    prod_set = set()
                    valid_prod = True
                    for j in range(len(candidate)):
                        cur = (0,) * P
                        for k in range(j, len(candidate)):
                            cur = add_sig(cur, cand_sigs[k])
                            if cur in prod_set:
                                valid_prod = False
                                break
                            prod_set.add(cur)
                        if not valid_prod:
                            break
                    if valid_prod:
                        # Same length but different arrangement might enable future insertions
                        # Only accept if it leads to a longer sequence via insertion
                        # For now, skip same-length swaps (they don't increase score)
                        pass
                if improved:
                    break

    return best_seq
