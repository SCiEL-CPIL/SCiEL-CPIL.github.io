import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 125

    def get_primes(n):
        sieve = bytearray([1]) * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i in range(2, n+1) if sieve[i]]

    primes = get_primes(1000)
    n_primes = len(primes)
    p_list = primes

    # Greedy construction: assign residues one prime at a time
    # For each prime p_i, choose r_i to maximize the number of integers in [1, LIMIT]
    # that are currently covered exactly once (singly-covered) and would become doubly-covered.
    # Also consider the boundary: we want to extend the prefix where coverage >= 2.

    LIMIT = 20000  # working window for greedy decisions

    # coverage[x] = number of primes (among those assigned so far) that cover x
    coverage = np.zeros(LIMIT + 1, dtype=np.int16)
    residues = np.zeros(n_primes, dtype=np.int32)

    for i, p in enumerate(p_list):
        if time.time() > deadline - 20:
            break

        # Find the current "doubly-covered prefix": largest L such that coverage[x] >= 2 for all x in 1..L
        # We'll compute this once per prime assignment for efficiency
        # But for greedy choice, we focus on: which residue r maximizes the gain?

        # For a given residue r mod p, the integers covered are r, r+p, r+2p, ... (with r=0 meaning p, 2p, ...)
        # The gain is: for each such integer x in [1, LIMIT], if coverage[x] == 1, it becomes 2 (good).
        # If coverage[x] == 0, it becomes 1 (neutral for our goal of >=2, but helps future primes).
        # If coverage[x] >= 2, it becomes 3+ (wasted).

        # We want to maximize the number of x in [1, LIMIT] where coverage[x] == 1 and x ≡ r (mod p).
        # Additionally, we want to ensure the prefix is extended, so we should prioritize
        # covering integers near the current boundary.

        # Compute the current boundary
        # Find first x >= 1 where coverage[x] < 2
        bad = np.where(coverage[1:LIMIT+1] < 2)[0]
        if len(bad) == 0:
            current_boundary = LIMIT
        else:
            current_boundary = int(bad[0])  # 0-indexed, so x = bad[0]+1 is first bad, boundary = bad[0]

        # Focus on integers in [current_boundary, min(LIMIT, current_boundary + 5*p)] 
        # Actually, let's just count singly-covered integers in the whole window for each residue.
        # But to be efficient, we can precompute for this prime p:
        # For each residue r in 0..p-1, count how many x in [1, LIMIT] with x ≡ r (mod p) have coverage[x] == 1.

        # Efficient approach: for this p, create an array of length p
        # gain[r] = number of x in [1, LIMIT] with x ≡ r (mod p) and coverage[x] == 1

        # We can do this by iterating over x where coverage[x] == 1 and accumulating into gain[x % p]
        # But that's O(LIMIT) per prime, which is 168 * 20000 = 3.36M, acceptable.

        gain = np.zeros(p, dtype=np.int32)
        # Also track: for boundary extension, we care about integers near current_boundary
        # Let's add a bonus for covering integers in [current_boundary+1, current_boundary + p]
        # Actually, let's just use the raw count of singly-covered integers.

        # Iterate through all x in 1..LIMIT where coverage[x] == 1
        # This is O(LIMIT) per prime
        for x in range(1, LIMIT + 1):
            if coverage[x] == 1:
                gain[x % p] += 1

        # Choose the residue with maximum gain
        best_r = int(np.argmax(gain))

        # But we also want to ensure we're extending the boundary.
        # Let's check: does this residue cover any integer in [current_boundary+1, current_boundary + 2*p]?
        # If not, maybe we should prefer a residue that does, even with slightly lower gain.
        # For now, let's stick with max gain and see.

        residues[i] = best_r

        # Update coverage
        r = best_r
        if r == 0:
            start = p
        else:
            start = r
        if start <= LIMIT:
            coverage[start::p] += 1

    # Now do a local repair pass: try to improve the boundary
    # Find current boundary
    bad = np.where(coverage[1:LIMIT+1] < 2)[0]
    if len(bad) == 0:
        boundary = LIMIT
    else:
        boundary = int(bad[0])

    # Local search: for primes near the end (larger primes), try changing their residues
    # to see if we can extend the boundary
    for _ in range(50):  # up to 50 rounds of local improvement
        if time.time() > deadline - 10:
            break

        improved = False
        # Try changing residues for the last 50 primes (largest primes have most flexibility)
        for i in range(n_primes - 50, n_primes):
            if time.time() > deadline - 10:
                break

            p = p_list[i]
            old_r = int(residues[i])

            # Try a few random alternatives
            for _ in range(5):
                new_r = np.random.randint(0, p)
                if new_r == old_r:
                    continue

                # Update coverage
                if old_r == 0:
                    old_start = p
                else:
                    old_start = old_r
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r

                if old_start <= LIMIT:
                    coverage[old_start::p] -= 1
                if new_start <= LIMIT:
                    coverage[new_start::p] += 1

                # Check new boundary
                bad_new = np.where(coverage[1:LIMIT+1] < 2)[0]
                if len(bad_new) == 0:
                    new_boundary = LIMIT
                else:
                    new_boundary = int(bad_new[0])

                if new_boundary > boundary:
                    residues[i] = new_r
                    boundary = new_boundary
                    improved = True
                else:
                    # Revert
                    if new_start <= LIMIT:
                        coverage[new_start::p] -= 1
                    if old_start <= LIMIT:
                        coverage[old_start::p] += 1

                if improved:
                    break

            if improved:
                break

        if not improved:
            break

    # Final full evaluation
    def full_score(residues, limit=100000):
        cov = np.zeros(limit + 1, dtype=np.int16)
        for i, p in enumerate(p_list):
            r = int(residues[i])
            if r == 0:
                start = p
            else:
                start = r
            if start <= limit:
                cov[start::p] += 1
        bad = np.where(cov[1:limit+1] < 2)[0]
        if len(bad) == 0:
            return limit
        return int(bad[0])

    final_score = full_score(residues, 100000)

    return residues.tolist()
