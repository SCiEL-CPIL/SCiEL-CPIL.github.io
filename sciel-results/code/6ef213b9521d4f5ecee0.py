import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Generate primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(2, 1001) if is_prime(p)]
    prime_set = set(primes)

    # Coverage window
    MAX_L = 10000

    def compute_L(covered):
        """Find the largest L such that 1..L are all covered."""
        L = 0
        for n in range(1, MAX_L + 1):
            if covered[n]:
                L = n
            else:
                break
        return L

    def greedy_assign(prime_list):
        """Greedy forward pass: assign each prime the residue covering most new numbers."""
        covered = [False] * (MAX_L + 1)
        assignments = {}
        for p in sorted(prime_list):
            best_r = 0
            best_count = -1
            for r in range(p):
                count = 0
                # Count newly covered positions
                for n in range(r % p + 1, MAX_L + 1, p):
                    if n >= 1 and not covered[n]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            assignments[p] = best_r
            for n in range(best_r % p + 1, MAX_L + 1, p):
                if n >= 1:
                    covered[n] = True
        L = compute_L(covered)
        return L, assignments

    def local_search(prime_list, max_iters=5000):
        """Local search: repeatedly try changing one residue to improve L."""
        covered = [False] * (MAX_L + 1)
        assignments = {}
        # Initialize with greedy
        for p in sorted(prime_list):
            best_r = 0
            best_count = -1
            for r in range(p):
                count = 0
                for n in range(r % p + 1, MAX_L + 1, p):
                    if n >= 1 and not covered[n]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            assignments[p] = best_r
            for n in range(best_r % p + 1, MAX_L + 1, p):
                if n >= 1:
                    covered[n] = True

        current_L = compute_L(covered)
        best_L = current_L
        best_assign = dict(assignments)

        no_improve = 0
        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Pick a random prime to reassign
            p = random.choice(prime_list)
            old_r = assignments[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue

            # Remove old contribution
            for n in range(old_r % p + 1, MAX_L + 1, p):
                if n >= 1:
                    # Check if any other prime covers n
                    still_covered = False
                    for q in prime_list:
                        if q == p:
                            continue
                        if (n - assignments[q]) % q == 0:
                            still_covered = True
                            break
                    if not still_covered:
                        covered[n] = False

            # Add new contribution
            for n in range(new_r % p + 1, MAX_L + 1, p):
                if n >= 1:
                    covered[n] = True

            new_L = compute_L(covered)

            if new_L >= current_L:
                current_L = new_L
                no_improve = 0
                if new_L > best_L:
                    best_L = new_L
                    best_assign = dict(assignments)
                    assignments[p] = new_r
                else:
                    # Accept but keep best
                    assignments[p] = new_r
            else:
                # Reject: revert
                assignments[p] = old_r
                # Rebuild covered from scratch is expensive; do it incrementally
                # Remove new contribution
                for n in range(new_r % p + 1, MAX_L + 1, p):
                    if n >= 1:
                        still_covered = False
                        for q in prime_list:
                            if q == p:
                                continue
                            if (n - assignments[q]) % q == 0:
                                still_covered = True
                                break
                        if not still_covered:
                            covered[n] = False
                # Re-add old contribution
                for n in range(old_r % p + 1, MAX_L + 1, p):
                    if n >= 1:
                        covered[n] = True
                no_improve += 1
                if no_improve > 500:
                    break

        return best_L, best_assign

    def simulated_annealing(prime_list, max_iters=8000):
        """SA over residue assignments."""
        covered = [False] * (MAX_L + 1)
        assignments = {}
        for p in sorted(prime_list):
            best_r = 0
            best_count = -1
            for r in range(p):
                count = 0
                for n in range(r % p + 1, MAX_L + 1, p):
                    if n >= 1 and not covered[n]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            assignments[p] = best_r
            for n in range(best_r % p + 1, MAX_L + 1, p):
                if n >= 1:
                    covered[n] = True

        current_L = compute_L(covered)
        best_L = current_L
        best_assign = dict(assignments)

        T = 50.0
        cooling = 0.999

        for it in range(max_iters):
            if time.time() > deadline:
                break

            p = random.choice(prime_list)
            old_r = assignments[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue

            # Remove old
            for n in range(old_r % p + 1, MAX_L + 1, p):
                if n >= 1:
                    still_covered = False
                    for q in prime_list:
                        if q == p:
                            continue
                        if (n - assignments[q]) % q == 0:
                            still_covered = True
                            break
                    if not still_covered:
                        covered[n] = False

            # Add new
            for n in range(new_r % p + 1, MAX_L + 1, p):
                if n >= 1:
                    covered[n] = True

            new_L = compute_L(covered)
            delta = new_L - current_L

            if delta >= 0 or random.random() < pow(2.718281828, delta / max(T, 1e-6)):
                current_L = new_L
                assignments[p] = new_r
                if new_L > best_L:
                    best_L = new_L
                    best_assign = dict(assignments)
            else:
                # Revert
                assignments[p] = old_r
                for n in range(new_r % p + 1, MAX_L + 1, p):
                    if n >= 1:
                        still_covered = False
                        for q in prime_list:
                            if q == p:
                                continue
                            if (n - assignments[q]) % q == 0:
                                still_covered = True
                                break
                        if not still_covered:
                            covered[n] = False
                for n in range(old_r % p + 1, MAX_L + 1, p):
                    if n >= 1:
                        covered[n] = True

            T *= cooling

        return best_L, best_assign

    # Candidate prime sets (sum of 1/p <= 1)
    # We need exact rational check
    def check_cost(prime_list):
        s = Fraction(0)
        for p in prime_list:
            s += Fraction(1, p)
        return s <= 1

    # Build several candidate sets
    candidates = []

    # Set 1: All primes >= 13 (the seed)
    s1 = [p for p in primes if p >= 13]
    if check_cost(s1):
        candidates.append(s1)

    # Set 2: Primes 5..97
    s2 = [p for p in primes if 5 <= p <= 97]
    if check_cost(s2):
        candidates.append(s2)

    # Set 3: 2, 3, 11, 13..97
    s3 = [2, 3, 11] + [p for p in primes if 13 <= p <= 97]
    if check_cost(s3):
        candidates.append(s3)

    # Set 4: 2, 3, 7, 13..71 (check cost)
    s4 = [2, 3, 7] + [p for p in primes if 13 <= p <= 71]
    if check_cost(s4):
        candidates.append(s4)

    # Set 5: 2, 3, 5, 17..61
    s5 = [2, 3, 5] + [p for p in primes if 17 <= p <= 61]
    if check_cost(s5):
        candidates.append(s5)

    # Set 6: 3, 5, 7, 11, 13..53
    s6 = [3, 5, 7, 11] + [p for p in primes if 13 <= p <= 53]
    if check_cost(s6):
        candidates.append(s6)

    # Set 7: 2, 3, 11, 13..79
    s7 = [2, 3, 11] + [p for p in primes if 13 <= p <= 79]
    if check_cost(s7):
        candidates.append(s7)

    best_L = 0
    best_classes = []

    # For each candidate, run local search and SA
    for idx, cand in enumerate(candidates):
        if time.time() > deadline:
            break

        # First: greedy
        L_g, assign_g = greedy_assign(cand)
        if L_g > best_L:
            best_L = L_g
            best_classes = [[p, assign_g[p]] for p in sorted(assign_g)]

        # Then: local search
        L_ls, assign_ls = local_search(cand, max_iters=3000)
        if L_ls > best_L:
            best_L = L_ls
            best_classes = [[p, assign_ls[p]] for p in sorted(assign_ls)]

        # Then: simulated annealing
        L_sa, assign_sa = simulated_annealing(cand, max_iters=5000)
        if L_sa > best_L:
            best_L = L_sa
            best_classes = [[p, assign_sa[p]] for p in sorted(assign_sa)]

        # Random restarts: perturb best and re-run local search
        if time.time() < deadline - 5:
            for restart in range(3):
                if time.time() > deadline:
                    break
                # Start from best_assign with random perturbation
                perturbed = dict(assign_sa)
                for _ in range(5):
                    p = random.choice(cand)
                    perturbed[p] = random.randint(0, p - 1)
                # Rebuild covered
                covered = [False] * (MAX_L + 1)
                for p in cand:
                    r = perturbed[p]
                    for n in range(r % p + 1, MAX_L + 1, p):
                        if n >= 1:
                            covered[n] = True
                L_r = compute_L(covered)
                # Local search from this point
                # (simplified: just evaluate)
                if L_r > best_L:
                    best_L = L_r
                    best_classes = [[p, perturbed[p]] for p in sorted(perturbed)]

    # Final verification with exact rationals
    if best_classes:
        total = Fraction(0)
        for p, r in best_classes:
            total += Fraction(1, p)
        if total > 1:
            # Should not happen, but safety
            return []

    return best_classes
