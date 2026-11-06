"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    num_primes = len(primes)

    # Precompute reciprocals as Fractions for exact budget checking
    # But for speed, use float for search and Fraction only for final validation
    recip_float = [1.0 / p for p in primes]
    recip_frac = [Fraction(1, p) for p in primes]

    def coverage_length(residues, primes_list):
        """Compute the largest L such that every integer 1..L is covered."""
        # Use a boolean array for coverage up to a reasonable cap
        cap = 100000
        covered = bytearray(cap + 1)  # index 0 unused, 1..cap
        for i, p in enumerate(primes_list):
            r = residues[i]
            # Mark all numbers n in [1, cap] where n ≡ r (mod p)
            # First number >= 1 with n ≡ r (mod p):
            if r == 0:
                first = p
            else:
                first = r
            if first < 1:
                first += p * ((1 - first + p - 1) // p)
            for n in range(first, cap + 1, p):
                covered[n] = 1
        # Find the largest L such that all of 1..L are covered
        L = 0
        for i in range(1, cap + 1):
            if covered[i]:
                L = i
            else:
                break
        return L

    def total_budget(residues_used, primes_list):
        """Sum of 1/p for used classes."""
        s = 0.0
        for i in range(len(residues_used)):
            if residues_used[i] is not None:
                s += recip_float[i]
        return s

    def is_valid(residues, primes_list):
        """Check distinct primes (always true by construction) and budget."""
        budget = total_budget(residues, primes_list)
        return budget <= 1.0

    # Greedy initial solution: use small primes first, assign residues to cover 1..L greedily
    def greedy_initial():
        # Sort primes ascending
        order = sorted(range(num_primes), key=lambda i: primes[i])
        residues = [None] * num_primes
        # We'll build coverage incrementally
        # Target: cover 1, 2, 3, ... as far as possible
        # For each prime in order, assign it to cover the smallest uncovered number
        covered = set()
        current_L = 0
        for idx in order:
            p = primes[idx]
            # Find smallest uncovered number
            n = current_L + 1
            while n in covered:
                n += 1
            if n > 100000:
                break
            # Assign residue so that n ≡ r (mod p)
            r = n % p
            residues[idx] = r
            # Update covered set (approximate: just mark n and nearby)
            # For speed, we'll recompute coverage at the end
            covered.add(n)
            current_L = n
        return residues

    # Simulated annealing
    def simulated_annealing(initial_residues, time_limit):
        random.seed(42)
        current_residues = initial_residues[:]
        current_L = coverage_length(current_residues, primes)
        best_residues = current_residues[:]
        best_L = current_L

        # Temperature schedule
        T_start = 1.0
        T_end = 0.001
        steps = 0
        total_steps = 500000  # adjust based on time

        for step in range(total_steps):
            if time.time() > time_limit:
                break

            # Choose a random prime index to modify
            idx = random.randint(0, num_primes - 1)
            old_r = current_residues[idx]
            new_r = random.randint(0, primes[idx] - 1)

            if old_r == new_r:
                continue

            # Try the move
            current_residues[idx] = new_r
            new_L = coverage_length(current_residues, primes)

            # Acceptance criterion
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = current_residues[:]
            else:
                # Accept with probability exp((new_L - current_L) / T)
                T = T_start * (T_end / T_start) ** (step / max(1, total_steps - 1))
                delta = new_L - current_L
                if random.random() < pow(2.718281828, delta / max(T, 1e-10)):
                    current_L = new_L
                else:
                    current_residues[idx] = old_r

        return best_residues, best_L

    # Local search polish: try to extend coverage by adjusting residues
    def local_polish(residues, time_limit):
        """Try to extend coverage by finding gaps and reassigning residues."""
        current_residues = residues[:]
        current_L = coverage_length(current_residues, primes)
        best_L = current_L

        # Find the first gap
        def find_first_gap(res):
            cap = 100000
            covered = bytearray(cap + 1)
            for i, p in enumerate(primes):
                r = res[i]
                if r is None:
                    continue
                if r == 0:
                    first = p
                else:
                    first = r
                if first < 1:
                    first += p * ((1 - first + p - 1) // p)
                for n in range(first, cap + 1, p):
                    covered[n] = 1
            for i in range(1, cap + 1):
                if not covered[i]:
                    return i
            return cap + 1

        while time.time() < time_limit:
            gap = find_first_gap(current_residues)
            if gap > 100000:
                break

            # Try to cover the gap by reassigning some residue
            improved = False
            for idx in range(num_primes):
                if current_residues[idx] is None:
                    continue
                p = primes[idx]
                # Try to set residue so that gap ≡ r (mod p)
                new_r = gap % p
                if new_r == current_residues[idx]:
                    continue
                old_r = current_residues[idx]
                current_residues[idx] = new_r
                new_L = coverage_length(current_residues, primes)
                if new_L > best_L:
                    best_L = new_L
                    improved = True
                else:
                    current_residues[idx] = old_r
                if improved:
                    break
            if not improved:
                # Try random perturbations
                for _ in range(10):
                    if time.time() > time_limit:
                        break
                    idx = random.randint(0, num_primes - 1)
                    if current_residues[idx] is None:
                        continue
                    old_r = current_residues[idx]
                    new_r = random.randint(0, primes[idx] - 1)
                    current_residues[idx] = new_r
                    new_L = coverage_length(current_residues, primes)
                    if new_L > best_L:
                        best_L = new_L
                        improved = True
                    else:
                        current_residues[idx] = old_r
                    if improved:
                        break
            if not improved:
                break

        return current_residues, best_L

    # Run greedy initial
    initial_residues = greedy_initial()
    initial_L = coverage_length(initial_residues, primes)

    # Run simulated annealing
    best_residues, best_L = simulated_annealing(initial_residues, deadline - 20)

    # Polish
    if time.time() < deadline - 5:
        best_residues, best_L = local_polish(best_residues, deadline - 5)

    # Final validation with exact rationals
    final_residues = best_residues
    final_classes = []
    budget = Fraction(0, 1)
    for i, p in enumerate(primes):
        r = final_residues[i]
        if r is not None:
            final_classes.append([p, r])
            budget += recip_frac[i]

    if budget > 1:
        # Fallback to greedy if budget exceeded
        final_residues = greedy_initial()
        final_classes = []
        budget = Fraction(0, 1)
        for i, p in enumerate(primes):
            r = final_residues[i]
            if r is not None:
                final_classes.append([p, r])
                budget += recip_frac[i]

    return final_classes
# EVOLVE-BLOCK-END
