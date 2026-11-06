import time
import numpy as np

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _first_start(r, p):
    """First x >= 1 such that x ≡ r (mod p)"""
    if r == 0:
        return p
    return r

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    nprimes = n

    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    # Work on a window; we will grow it as L grows.
    # Use a generous max window so we can measure L up to ~20000.
    MAXW = 20000

    def _compute_counts(residues, max_check):
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for r, p in zip(residues, primes):
            start = _first_start(r, p)
            if start <= max_check:
                counts[start::p] += 1
        return counts

    def _find_L(counts):
        # largest L such that counts[1..L] >= 2
        below = np.where(counts[1:] < 2)[0]
        if len(below) == 0:
            return len(counts) - 1
        return int(below[0])

    def _setcover_greedy(max_check, time_limit):
        """Process primes in increasing order; pick residue to maximize coverage
        of positions that are currently single-covered (count==1) in [1, max_check].
        Ties broken by coverage of uncovered (count==0) positions, then by residue value."""
        residues = [0] * n
        counts = np.zeros(max_check + 1, dtype=np.int32)

        for i, p in enumerate(primes):
            if time.time() > time_limit:
                break
            # For each candidate residue r in [0, p), compute coverage gain
            # on positions with count==1 (primary) and count==0 (secondary).
            # We only care about positions in [1, max_check].
            # The positions covered by residue r mod p are: start, start+p, ...
            # We want to maximize: (# of those positions where count==1) * 1000
            #                     + (# of those positions where count==0) * 1
            # (weighting single-covered much higher since they're the bottleneck)

            best_r = 0
            best_score = -1

            # Vectorize over residues: for each r, the covered positions are
            # first_start(r,p) + k*p. We can compute the count values at those
            # positions for each r.
            # For efficiency with large p, we iterate but use numpy slicing.
            for r in range(p):
                s = _first_start(r, p)
                if s > max_check:
                    # No positions in [1, max_check] covered
                    score = 0
                else:
                    idx = np.arange(s, max_check + 1, p)
                    cvals = counts[idx]
                    n_single = np.sum(cvals == 1)
                    n_zero = np.sum(cvals == 0)
                    score = n_single * 1000 + n_zero
                if score > best_score:
                    best_score = score
                    best_r = r
                elif score == best_score and r < best_r:
                    best_r = r

            residues[i] = best_r
            s = _first_start(best_r, p)
            if s <= max_check:
                counts[s::p] += 1

        return residues, _find_L(counts)

    def _local_search(residues, max_check, time_limit, max_rounds=50):
        """Hill-climb: try changing each residue to maximize L."""
        residues = list(residues)
        counts = _compute_counts(residues, max_check)
        best_L = _find_L(counts)

        for _ in range(max_rounds):
            if time.time() > time_limit:
                break
            improved = False
            for i, p in enumerate(primes):
                if time.time() > time_limit:
                    break
                old_r = residues[i]
                # Try a few candidate residues: old_r, and residues that cover
                # the first gap position
                target_x = best_L + 1
                candidates = set()
                candidates.add(target_x % p)
                # Also try a few random ones
                for _ in range(3):
                    candidates.add(int(np.random.randint(0, p)))

                best_new_r = old_r
                best_new_L = best_L

                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    # Apply
                    old_s = _first_start(old_r, p)
                    new_s = _first_start(new_r, p)
                    if old_s <= max_check:
                        counts[old_s::p] -= 1
                    if new_s <= max_check:
                        counts[new_s::p] += 1

                    new_L = _find_L(counts)

                    # Revert
                    if new_s <= max_check:
                        counts[new_s::p] -= 1
                    if old_s <= max_check:
                        counts[old_s::p] += 1

                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_new_r = new_r

                if best_new_r != old_r:
                    # Apply the best change
                    old_s = _first_start(old_r, p)
                    new_s = _first_start(best_new_r, p)
                    if old_s <= max_check:
                        counts[old_s::p] -= 1
                    if new_s <= max_check:
                        counts[new_s::p] += 1
                    residues[i] = best_new_r
                    best_L = best_new_L
                    improved = True

            if not improved:
                break

        return residues, best_L

    def _grow_and_optimize(residues, max_check, time_limit):
        """Grow max_check as L approaches it, re-running local search."""
        current_M = max_check
        current_L = _find_L(_compute_counts(residues, max_check))

        while time.time() < time_limit:
            if current_L < current_M:
                # Stuck, do local search
                residues, current_L = _local_search(residues, current_M, time_limit)
                if current_L < current_M:
                    break
            # Extend range
            current_M = min(current_M * 2, MAXW)
            residues, current_L = _local_search(residues, current_M, time_limit)

        return residues, current_L

    # Phase 1: Set-cover greedy to get a good initial solution
    t1 = deadline - 60  # spend up to 60s on set-cover
    residues, L = _setcover_greedy(MAXW, t1)

    # Phase 2: Local search to polish
    t2 = deadline - 10
    residues, L = _local_search(residues, MAXW, t2)

    # Phase 3: If time remains, try a few random restarts of local search
    if time.time() < t2:
        for _ in range(3):
            if time.time() > t2:
                break
            # Perturb some residues
            perturbed = list(residues)
            for _ in range(20):
                idx = int(np.random.randint(0, nprimes))
                p = primes[idx]
                perturbed[idx] = int(np.random.randint(0, p))
            res2, L2 = _local_search(perturbed, MAXW, t2)
            if L2 > L:
                residues, L = res2, L2

    return [int(r) for r in residues]
