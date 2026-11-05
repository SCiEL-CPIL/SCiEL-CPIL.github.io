"""Vectorized incremental hill-climbing solver for the covering problem."""
import time
import numpy as np


def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = [False] * len(sieve[i * i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def _compute_counts(residues, primes, max_L):
    """Compute coverage counts array for given residues."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        pos = np.arange(start, max_L + 1, p)
        if len(pos) > 0:
            counts[pos] += 1
    return counts


def _find_L(counts, max_L):
    """Find largest L such that counts[1..L] >= 2."""
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_L
    else:
        return below_two[0]


def _get_positions(r, p, max_L):
    """Get positions covered by residue r mod p in 1..max_L."""
    if r == 0:
        start = p
    else:
        start = r
    if start > max_L:
        return np.array([], dtype=np.int32)
    return np.arange(start, max_L + 1, p, dtype=np.int32)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit

    primes = _get_primes(1000)
    n_primes = len(primes)
    max_L = 100000

    # Initial solution: try to be smart about initial residues
    # Use a greedy forward construction: for each prime, pick the residue
    # that maximizes coverage given the primes already placed.
    # But start with a reasonable initial guess and refine.
    initial_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_residues = initial_residues[:]
    counts = _compute_counts(best_residues, primes, max_L)
    best_L = _find_L(counts, max_L)

    # Precompute position arrays for each prime and each residue
    # positions[i][r] = positions covered by residue r mod primes[i]
    # This is a list of lists: positions[i] is a list of length primes[i]
    positions = []
    for i, p in enumerate(primes):
        pos_list = []
        for r in range(p):
            pos_list.append(_get_positions(r, p, max_L))
        positions.append(pos_list)

    # Also precompute the "old positions" for the current residue of each prime
    # and the lengths, to speed up delta computation

    # Vectorized hill-climbing:
    # For each prime i, we want to find the best new_r that maximizes L.
    # The key insight: for prime i with current residue r_i, changing to new_r
    # gives: counts_new = counts - 1 at old_pos + 1 at new_pos
    # We can evaluate all new_r simultaneously by:
    #   1. Start with a base = counts - 1 at old_pos (this is the "removed" state)
    #   2. For each candidate new_r, add 1 at new_pos[new_r]
    #   3. Find L for each candidate
    #
    # To do this vectorized:
    #   base = counts.copy()
    #   base[old_pos] -= 1
    #   Then for each new_r: candidate = base.copy(); candidate[new_pos] += 1; L = find_L(candidate)
    #
    # But we can do even better: note that the only difference between candidates
    # is which positions get +1. We can precompute for each position x, which
    # candidates (new_r values) would give x a +1.
    #
    # Actually, the simplest vectorized approach:
    # For prime i, create an array of shape (p, max_L+1) where row r is the
    # count array if we set residue to r. But that's p * (max_L+1) = 1000 * 100001
    # = 100M entries * 4 bytes = 400MB. Too much.
    #
    # Alternative: evaluate candidates in batches.
    # For a batch of B candidates, create a (B, max_L+1) array.
    # B = 100 gives 100 * 100001 * 4 = 40MB. Manageable.
    #
    # Actually, let's think more carefully. The bottleneck is _find_L, which
    # does np.where(counts[1:] < 2). If we have B candidates, we can stack them
    # and do a single np.where on the stacked array.
    #
    # Let's try: for prime i, evaluate all p candidates in batches of B.

    BATCH_SIZE = 50  # tune this

    improved = True
    iteration = 0
    while improved and (time.time() - start_time) < deadline:
        improved = False
        iteration += 1
        for i in range(n_primes):
            if (time.time() - start_time) > deadline - 5:
                break
            p = primes[i]
            current_r = best_residues[i]

            # Compute the "base" counts: current counts minus the old residue's coverage
            old_pos = positions[i][current_r]
            base = counts.copy()
            if len(old_pos) > 0:
                base[old_pos] -= 1

            # Now evaluate all new_r in 0..p-1 (skip current_r)
            best_new_r = current_r
            best_new_L = best_L

            # Process in batches
            candidates = [r for r in range(p) if r != current_r]
            for batch_start in range(0, len(candidates), BATCH_SIZE):
                if (time.time() - start_time) > deadline - 5:
                    break
                batch = candidates[batch_start:batch_start + BATCH_SIZE]
                B = len(batch)

                # Create (B, max_L+1) array
                # Each row is base + 1 at the positions for that candidate
                # We can do this by: start with base broadcast, then add 1 at specific positions
                # For efficiency: create the batch array as a copy of base for each row,
                # then add 1 at the appropriate positions.

                # Method: create a 2D array of shape (B, max_L+1)
                # Initialize all rows to base
                batch_arr = np.tile(base, (B, 1))  # (B, max_L+1)

                # For each candidate in the batch, add 1 at its positions
                for j, new_r in enumerate(batch):
                    new_pos = positions[i][new_r]
                    if len(new_pos) > 0:
                        batch_arr[j, new_pos] += 1

                # Find L for each candidate: first position where count < 2
                # We want the first x in 1..max_L where batch_arr[j, x] < 2
                # Equivalently: for each row j, find min x >= 1 where batch_arr[j, x] < 2
                # If no such x, L = max_L

                below_two_mask = batch_arr[:, 1:] < 2  # (B, max_L)
                # For each row, find the first True
                # np.argmax returns 0 if no True, so we need to be careful
                first_below = np.argmax(below_two_mask, axis=1)  # (B,)
                # Where below_two_mask[j] has no True, argmax returns 0
                # We need to check if there was actually a True
                has_below = below_two_mask.any(axis=1)  # (B,)
                # L for candidate j: if has_below[j], then first_below[j] + 1, else max_L
                # But wait: first_below is 0-indexed in the [1:] slice, so position = first_below + 1
                Ls = np.where(has_below, first_below + 1, max_L)  # (B,)

                # Find the best L in this batch
                best_idx_in_batch = np.argmax(Ls)
                best_L_in_batch = Ls[best_idx_in_batch]
                if best_L_in_batch > best_new_L:
                    best_new_L = best_L_in_batch
                    best_new_r = batch[best_idx_in_batch]

            if best_new_r != current_r:
                # Apply the best change
                old_pos = positions[i][current_r]
                new_pos = positions[i][best_new_r]
                if len(old_pos) > 0:
                    counts[old_pos] -= 1
                if len(new_pos) > 0:
                    counts[new_pos] += 1
                best_residues[i] = best_new_r
                best_L = best_new_L
                improved = True

    return best_residues
