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

    def seq_to_sig_set(seq):
        """Convert a value sequence to its set of block-product signatures."""
        sigs = [sigs_fixed[x] for x in seq]
        prod_set = set()
        for i in range(len(sigs)):
            cur = None
            for j in range(i, len(sigs)):
                if cur is None:
                    cur = sigs[j]
                else:
                    cur = add_sig(cur, sigs[j])
                prod_set.add(cur)
        return prod_set

    def check_valid_seq(seq):
        """Verify strictly increasing and distinct block products."""
        if len(seq) == 0:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        prod_set = set()
        for i in range(len(seq)):
            prod = 1
            for j in range(i, len(seq)):
                prod *= seq[j]
                if prod in prod_set:
                    return False
                prod_set.add(prod)
        return True

    # Strategy 1: Standard Greedy
    best_seq = greedy_construct(set())
    best_len = len(best_seq)

    # Strategy 2: Local Search on Skip Set
    current_skip_set = set()
    current_seq = list(best_seq)
    current_len = best_len
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

    if current_len > best_len:
        best_seq = current_seq
        best_len = current_len

    # Strategy 3: Randomized restarts
    random.seed(42)
    for trial in range(3000):
        if time.time() > deadline:
            break
        strategy = trial % 5
        skip_set = set()
        if strategy == 0:
            for x in range(2, 15):
                if random.random() < 0.3:
                    skip_set.add(x)
        elif strategy == 1:
            for x in range(2, 25):
                if random.random() < 0.25:
                    skip_set.add(x)
        elif strategy == 2:
            conflict_prone = [2, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28]
            for x in conflict_prone:
                if random.random() < 0.4:
                    skip_set.add(x)
        elif strategy == 3:
            for x in range(2, 40):
                if x % 2 == 0 and random.random() < 0.3:
                    skip_set.add(x)
                elif x % 3 == 0 and x % 2 != 0 and random.random() < 0.2:
                    skip_set.add(x)
        else:
            for x in range(2, 50):
                if random.random() < 0.15:
                    skip_set.add(x)
        seq = greedy_construct(skip_set)
        if len(seq) > best_len:
            best_seq = seq
            best_len = len(seq)

    # Strategy 4: CSP-style backtracking with MRV heuristic
    # This is the structural change: instead of greedy + local search on skip sets,
    # we do a guided backtracking that picks the next element based on which
    # choice leaves the most future options open (MRV-inspired).
    def csp_backtrack(current_seq, current_prod_set, remaining, depth, max_depth):
        """
        Backtracking search guided by MRV: at each step, try candidates in order
        of how many future elements they conflict with (fewer conflicts = more remaining).
        """
        if time.time() > deadline:
            return current_seq
        if depth >= max_depth:
            return current_seq

        if not remaining:
            return current_seq

        # MRV heuristic: for each candidate, count how many of the remaining
        # elements would be blocked. Pick the one that blocks the fewest.
        # But computing this fully is expensive, so we approximate:
        # Try candidates in order, but prefer those that are "safer" (larger primes, etc.)
        
        best_candidate = None
        best_future_count = -1
        best_new_prod_set = None
        best_new_remaining = None

        # To keep this fast, only evaluate a subset of candidates per level
        # Sort remaining by a heuristic: prefer numbers with fewer small prime factors
        def score_candidate(x):
            sig = sigs_fixed[x]
            # Count non-zero entries (number of distinct prime factors)
            nz = sum(1 for v in sig if v > 0)
            # Prefer numbers with fewer distinct prime factors (less likely to conflict)
            return nz

        # Limit candidates to keep backtracking tractable
        limited_remaining = list(remaining)
        limited_remaining.sort(key=score_candidate)
        # Only consider top candidates to limit branching
        candidates_to_try = limited_remaining[:15]

        for x in candidates_to_try:
            if x in current_seq:
                continue
            # Check strictly increasing
            if current_seq and x <= current_seq[-1]:
                continue

            x_sig = sigs_fixed[x]
            new_tails = [x_sig]
            current_prod = x_sig
            for s in reversed([sigs_fixed[v] for v in current_seq]):
                current_prod = add_sig(current_prod, s)
                new_tails.append(current_prod)

            valid = True
            for tail in new_tails:
                if tail in current_prod_set:
                    valid = False
                    break

            if not valid:
                continue

            new_prod_set = current_prod_set | set(new_tails)
            new_seq = current_seq + [x]
            new_remaining = [v for v in remaining if v > x]

            # Recurse
            result = csp_backtrack(new_seq, new_prod_set, new_remaining, depth + 1, max_depth)
            if len(result) > len(current_seq):
                return result

        return current_seq

    # Run CSP backtracking from the start
    if time.time() < deadline:
        initial_remaining = list(range(2, 129))
        csp_result = csp_backtrack([], set(), initial_remaining, 0, 127)
        if len(csp_result) > best_len:
            best_seq = csp_result
            best_len = len(csp_result)

    # Strategy 5: Iterated local search with deletion + reinsertion
    # From the best sequence, try removing one element and then greedily
    # reinserting elements to see if we can get a longer sequence.
    if time.time() < deadline:
        best_copy = list(best_seq)
        best_copy_len = len(best_copy)
        
        for removal_iter in range(200):
            if time.time() > deadline:
                break
            
            # Try removing a random element from the first half of the sequence
            if len(best_copy) < 2:
                break
            idx = random.randint(0, min(len(best_copy) // 2, len(best_copy) - 1))
            removed_val = best_copy[idx]
            modified = best_copy[:idx] + best_copy[idx + 1:]
            
            # Now greedily try to add back elements that were skipped
            # Build the set of used values
            used_set = set(modified)
            modified_sigs = [sigs_fixed[x] for x in modified]
            prod_set = seq_to_sig_set(modified)
            
            # Try adding elements in order
            new_modified = list(modified)
            new_modified_sigs = list(modified_sigs)
            new_prod_set = set(prod_set)
            
            for x in range(2, 129):
                if x in used_set:
                    continue
                x_sig = sigs_fixed[x]
                # Check if x fits at the end (since we're rebuilding greedily)
                # Actually, we need to check if it can be inserted anywhere
                # For simplicity, just try appending
                valid, new_tails = is_valid_addition(new_modified_sigs, x_sig, new_prod_set)
                if valid:
                    new_modified.append(x)
                    new_modified_sigs.append(x_sig)
                    new_prod_set.update(new_tails)
                    used_set.add(x)
            
            if len(new_modified) > best_copy_len:
                best_copy = new_modified
                best_copy_len = len(new_modified)
                best_seq = new_modified
                best_len = best_copy_len

    # Strategy 6: Final polish - try single-element insertions
    if time.time() < deadline:
        best_copy = list(best_seq)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_copy) + 1):
                for new_val in range(2, 129):
                    if new_val in best_copy:
                        continue
                    if i > 0 and new_val <= best_copy[i - 1]:
                        continue
                    if i < len(best_copy) and new_val >= best_copy[i]:
                        continue
                    candidate = best_copy.copy()
                    candidate.insert(i, new_val)
                    prod_set = set()
                    valid_prod = True
                    for j in range(len(candidate)):
                        prod = 1
                        for k in range(j, len(candidate)):
                            prod *= candidate[k]
                            if prod in prod_set:
                                valid_prod = False
                                break
                            prod_set.add(prod)
                        if not valid_prod:
                            break
                    if valid_prod and len(candidate) > best_len:
                        best_seq = candidate
                        best_copy = candidate.copy()
                        best_len = len(candidate)
                        improved = True
                        break
                if improved:
                    break

    # Final validation
    if not check_valid_seq(best_seq):
        # Fallback to a known valid sequence
        best_seq = greedy_construct(set())

    return best_seq
