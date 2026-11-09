"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n = len(primes)  # 168

    # Fast evaluator using difference array.
    # Returns the largest L (capped at max_check) such that every x in 1..L has count>=2.
    def evaluate(residues, max_check=100000):
        # We'll expand limit until we find a failure or reach max_check.
        limit = 256
        while limit <= max_check:
            # diff array of size limit+2
            diff = [0] * (limit + 2)
            for i in range(n):
                p = primes[i]
                r = residues[i]
                # x = r + k*p in [1, limit]
                if r == 0:
                    k_min = 1
                else:
                    k_min = 0
                k_max = (limit - r) // p
                if k_min > k_max:
                    continue
                start = r + k_min * p
                # positions: start, start+p, start+2p, ... up to <= limit
                # Use diff: add 1 at each covered position via step pattern
                # Efficient: iterate
                x = start
                while x <= limit:
                    diff[x] += 1
                    x += p
            # prefix sum
            # find first x in 1..limit with count < 2
            running = 0
            for x in range(1, limit + 1):
                running += diff[x]
                if running < 2:
                    return x - 1
            # all passed
            if limit >= max_check:
                return max_check
            limit = min(limit * 2, max_check)
        return max_check

    # Precompute prime list as a global-ish for speed
    P = primes

    # Seed: assign prime i to cover ceil((i+1)/2)
    def seed_solution():
        return [((i + 1) // 2) % p for i, p in enumerate(P)]

    def random_solution():
        return [random.randint(0, p - 1) for p in P]

    # Local search with fast single-move evaluation.
    # We maintain the current residues and a cached evaluation.
    # For a single change at index i, we can quickly recompute by re-evaluating
    # but to be fast we just call evaluate (which is reasonably fast for small L).
    # To speed up, we do a "delta" evaluation: recompute counts only around the changed prime.
    # Simpler: call evaluate but it's O(n*L/p). For L~1000 that's fine.

    def local_search(residues, max_time):
        current = list(residues)
        current_score = evaluate(current)
        if current_score >= 1000:
            return current, current_score

        improved = True
        # Track how many times we've done a full pass
        while improved and time.time() < max_time:
            improved = False
            # Randomize order to avoid getting stuck
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if time.time() >= max_time:
                    break
                p = P[i]
                old_r = current[i]
                # Try several candidates
                candidates = set()
                # Always try a few random ones and a few near old
                for _ in range(8):
                    candidates.add(random.randint(0, p - 1))
                # Also try old_r +/- 1
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    current[i] = new_r
                    new_score = evaluate(current)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        current[i] = old_r
                # restore
                current[i] = old_r
        return current, current_score

    # Main
    random.seed(12345)

    best_residues = seed_solution()
    best_score = evaluate(best_residues)

    # First local search from seed
    best_residues, best_score = local_search(best_residues, deadline - 60.0)
    if best_score >= 1000:
        return best_residues

    # Multiple random restarts
    while time.time() < deadline - 30.0:
        cand = random_solution()
        cand, score = local_search(cand, deadline - 30.0)
        if score > best_score:
            best_score = score
            best_residues = cand
            if best_score >= 1000:
                break

    # Final polish on best
    if time.time() < deadline - 5.0:
        best_residues, best_score = local_search(best_residues, deadline - 5.0)

    return best_residues
# EVOLVE-BLOCK-END
