"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
import numpy as np

N = 256
MAX_SUM = 2 * N

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 55.0
    rng = random.Random(12345)

    # --- Fast bitmask-based Sidon maintenance ---
    # We represent the set of pair sums as a Python int bitmask.
    # Bit s is 1 iff sum s is present.
    # Sums range from 2 to 2*N = 512, so we need ~513 bits.

    def greedy_bitmask(order):
        """Greedy: add elements in given order if no sum conflict."""
        S = []
        sums = 0
        for x in order:
            # Check if 2*x is already in sums
            if (sums >> (2 * x)) & 1:
                continue
            # Check if x+a is in sums for any a in S
            conflict = False
            for a in S:
                if (sums >> (x + a)) & 1:
                    conflict = True
                    break
            if not conflict:
                S.append(x)
                sums |= 1 << (2 * x)
                for a in S[:-1]:
                    sums |= 1 << (x + a)
        return S, sums

    def is_maximal(S, sums):
        """Check inclusion-maximality: no missing x can be added."""
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Can x be added?
            if (sums >> (2 * x)) & 1:
                continue
            blocked = False
            for a in S:
                if (sums >> (x + a)) & 1:
                    blocked = True
                    break
            if not blocked:
                return False
        return True

    def remove_element(S, sums, idx):
        """Remove S[idx] from S and update sums bitmask. Returns new S, sums or None if invalid."""
        x = S[idx]
        new_S = S[:idx] + S[idx+1:]
        # Recompute sums for new_S (small enough for N=256)
        new_sums = 0
        n = len(new_S)
        for i in range(n):
            new_sums |= 1 << (2 * new_S[i])
            for j in range(i + 1, n):
                new_sums |= 1 << (new_S[i] + new_S[j])
        return new_S, new_sums

    def prune_to_minimal(S, sums):
        """Repeatedly try to remove elements while maintaining maximality."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n = len(S)
            # Try removing in random order to escape local optima
            order = list(range(n))
            rng.shuffle(order)
            for idx in order:
                if time.time() > deadline:
                    break
                new_S, new_sums = remove_element(S, sums, idx)
                if not new_S:
                    continue
                if is_maximal(new_S, new_sums):
                    S, sums = new_S, new_sums
                    changed = True
                    break
        return S, sums

    # --- Phase 1: Deterministic greedy + prune ---
    S, sums = greedy_bitmask(range(1, N + 1))
    S, sums = prune_to_minimal(S, sums)
    best_S = S
    best_sums = sums
    best_size = len(S)

    base_order = list(range(1, N + 1))

    # --- Phase 2: Randomized restarts with diverse orderings ---
    trial = 0
    while time.time() < deadline:
        trial += 1
        order = base_order[:]
        strategy = trial % 12

        if strategy == 0:
            rng.shuffle(order)
        elif strategy == 1:
            order = order[::-1]
        elif strategy == 2:
            # Sort by random hash
            h = [rng.randint(0, 10**9) for _ in range(N)]
            order = [x for _, x in sorted(zip(h, order))]
        elif strategy == 3:
            # Bias toward large numbers
            order.sort(reverse=True)
            # Partial shuffle of first half
            half = len(order) // 2
            for i in range(half - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        elif strategy == 4:
            # Alternate low/high
            low = [x for x in order if x <= N // 2]
            high = [x for x in order if x > N // 2]
            rng.shuffle(low)
            rng.shuffle(high)
            new_order = []
            li, hi = 0, 0
            while li < len(low) or hi < len(high):
                if li < len(low):
                    new_order.append(low[li]); li += 1
                if hi < len(high):
                    new_order.append(high[hi]); hi += 1
            order = new_order
        elif strategy == 5:
            # Quadratic residue ordering (structural)
            order = sorted(range(1, N + 1), key=lambda x: (x * x) % (N + 1))
        elif strategy == 6:
            # Reverse quadratic
            order = sorted(range(1, N + 1), key=lambda x: (x * x) % (N + 1), reverse=True)
        elif strategy == 7:
            # Random with gap: skip some small numbers early
            small = [x for x in order if x <= 32]
            rest = [x for x in order if x > 32]
            rng.shuffle(small)
            rng.shuffle(rest)
            order = rest + small
        elif strategy == 8:
            # Golden ratio ordering
            phi = 0.6180339887
            order = sorted(range(1, N + 1), key=lambda x: (x * phi * N) % N)
        elif strategy == 9:
            # Prime ordering
            def is_prime(n):
                if n < 2: return False
                for p in range(2, int(n**0.5) + 1):
                    if n % p == 0: return False
                return True
            primes = [x for x in range(1, N + 1) if is_prime(x)]
            nonprimes = [x for x in range(1, N + 1) if not is_prime(x)]
            rng.shuffle(primes)
            rng.shuffle(nonprimes)
            order = primes + nonprimes
        elif strategy == 10:
            # Random with clustering
            clusters = [rng.randint(1, N) for _ in range(8)]
            order = sorted(range(1, N + 1), key=lambda x: min(abs(x - c) for c in clusters))
            rng.shuffle(order)
        elif strategy == 11:
            # Fibonacci-like ordering
            fibs = [1]
            while fibs[-1] < N:
                fibs.append(fibs[-1] + fibs[-2] if len(fibs) >= 2 else 1)
            fib_set = set(fibs)
            fib_order = [x for x in range(1, N + 1) if x in fib_set]
            non_fib = [x for x in range(1, N + 1) if x not in fib_set]
            rng.shuffle(non_fib)
            order = fib_order + non_fib

        S, sums = greedy_bitmask(order)
        if not S:
            continue
        S, sums = prune_to_minimal(S, sums)
        if len(S) < best_size:
            best_size = len(S)
            best_S = S
            best_sums = sums

    # --- Phase 3: Final aggressive local search on best ---
    # Try swapping elements
    for _ in range(200):
        if time.time() > deadline:
            break
        S_cur, sums_cur = best_S[:], best_sums
        n = len(S_cur)
        if n == 0:
            break
        # Try replacing one element with a missing one
        missing = [x for x in range(1, N + 1) if x not in S_cur]
        rng.shuffle(missing)
        for new_x in missing[:50]:
            if time.time() > deadline:
                break
            # Try each position to remove
            for i in range(n):
                old_x = S_cur[i]
                new_S = S_cur[:i] + S_cur[i+1:] + [new_x]
                # Recompute sums
                new_sums = 0
                m = len(new_S)
                for a in range(m):
                    new_sums |= 1 << (2 * new_S[a])
                    for b in range(a + 1, m):
                        new_sums |= 1 << (new_S[a] + new_S[b])
                # Check Sidon
                # Verify all sums distinct
                sum_list = []
                for a in range(m):
                    sum_list.append(2 * new_S[a])
                    for b in range(a + 1, m):
                        sum_list.append(new_S[a] + new_S[b])
                if len(set(sum_list)) != len(sum_list):
                    continue
                if is_maximal(new_S, new_sums):
                    if len(new_S) < best_size:
                        best_size = len(new_S)
                        best_S = new_S
                        best_sums = new_sums
                    break

    # --- Phase 4: Final prune ---
    best_S, best_sums = prune_to_minimal(best_S, best_sums)

    return best_S

# EVOLVE-BLOCK-END
