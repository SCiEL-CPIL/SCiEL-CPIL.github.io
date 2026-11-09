import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n_primes = len(primes)  # 168

    # Precompute: for each prime p_i, the mask of integers 1..MAX_N that are ≡ r (mod p)
    # We'll use a coverage-count approach: for a given set of residues, compute how many
    # congruences each integer satisfies.
    MAX_N = 5000  # We search up to this; the answer won't exceed this realistically for n=1000

    # Precompute modular patterns: for prime p, integer x (1-indexed), x mod p
    # We need: for each prime index i and each residue r, which positions in 1..MAX_N satisfy x ≡ r (mod p)
    # This is expensive to precompute fully. Instead, we compute coverage on the fly.

    def evaluate(residues, n_max=MAX_N):
        """Return the largest L such that every integer 1..L has coverage >= 2."""
        # coverage[x] = number of primes p_i such that x ≡ residues[i] (mod p_i)
        # Use vectorized approach: for each prime, mark which positions it covers
        coverage = np.zeros(n_max, dtype=np.int32)  # 0-indexed: position j corresponds to integer j+1
        for i, p in enumerate(primes):
            r = residues[i]
            # Integers x in 1..n_max with x ≡ r (mod p):
            # x = r, r+p, r+2p, ... but x >= 1
            # If r == 0, x = p, 2p, 3p, ...
            # If r > 0, x = r, r+p, r+2p, ...
            if r == 0:
                # x = p, 2p, ...
                positions = np.arange(p - 1, n_max, p)  # 0-indexed: p-1, 2p-1, ...
            else:
                # x = r, r+p, r+2p, ...
                # 0-indexed: r-1, r-1+p, ...
                if r - 1 < n_max:
                    positions = np.arange(r - 1, n_max, p)
                else:
                    positions = np.empty(0, dtype=np.int64)
            if len(positions) > 0:
                coverage[positions] += 1
        # Find largest L such that coverage[0..L-1] all >= 2
        # Find first position where coverage < 2
        if n_max == 0:
            return 0
        below = np.where(coverage < 2)[0]
        if len(below) == 0:
            return n_max
        return int(below[0])

    def seed_solution():
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]

    def random_solution():
        return [random.randint(0, p - 1) for p in primes]

    # Local search: try changing one residue at a time, keep best
    def local_search(residues, time_limit):
        residues = list(residues)
        best_score = evaluate(residues)
        no_improve = 0
        max_no_improve = 50

        while time.time() < time_limit:
            improved = False
            # Try each prime position
            for i in range(n_primes):
                if time.time() >= time_limit:
                    break
                p = primes[i]
                old_r = residues[i]
                # Try all possible residues for this prime
                # We can't try all p residues if p is large (up to 997)
                # Instead, try a subset: the current, and some promising ones
                # Actually for small primes we can try all; for large ones, sample
                if p <= 50:
                    candidates = list(range(p))
                else:
                    # Sample: include old, 0, and random samples
                    candidates = [old_r, 0]
                    if p <= 200:
                        candidates = list(range(p))
                    else:
                        candidates = [old_r, 0] + [random.randint(0, p - 1) for _ in range(20)]
                    candidates = list(set(candidates))

                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    residues[i] = new_r
                    score = evaluate(residues)
                    if score > best_score:
                        best_score = score
                        no_improve = 0
                        improved = True
                    else:
                        residues[i] = old_r

                if improved:
                    break  # restart scanning from beginning
                else:
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break

            if not improved:
                break

        return residues, best_score

    # Multi-start search
    best_residues = seed_solution()
    best_score = evaluate(best_residues)

    # Try seed first
    r, s = local_search(best_residues, deadline - 10)
    if s > best_score:
        best_score = s
        best_residues = r

    # Random restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() >= deadline - 10:
            break
        # Mix: sometimes random, sometimes perturbation of best
        if restart % 2 == 0:
            init = random_solution()
        else:
            init = list(best_residues)
            # Perturb a few residues
            for _ in range(random.randint(3, 10)):
                idx = random.randint(0, n_primes - 1)
                init[idx] = random.randint(0, primes[idx] - 1)

        r, s = local_search(init, deadline - 10)
        if s > best_score:
            best_score = s
            best_residues = r

    # Final: try to squeeze more from best with more local search time
    if time.time() < deadline - 5:
        r, s = local_search(best_residues, deadline - 2)
        if s > best_score:
            best_score = s
            best_residues = r

    return best_residues
