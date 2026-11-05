import time
from fractions import Fraction
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    small_primes = [2, 3, 5, 7, 11]
    large_primes = [p for p in primes if p > 11]

    # Generate candidate prime sets:
    # For each subset of small primes (that fits in budget), fill with smallest large primes
    candidate_sets = []
    seen_sets = set()

    # Include the original seed: primes >= 13
    seed_set = [p for p in primes if p >= 13]
    candidate_sets.append(seed_set)
    seen_sets.add(tuple(sorted(seed_set)))

    # Systematically try subsets of small primes
    for r in range(len(small_primes) + 1):
        for subset in itertools.combinations(small_primes, r):
            current_sum = sum(Fraction(1, p) for p in subset)
            if current_sum > 1:
                continue

            current_set = list(subset)
            # Fill remaining budget with smallest large primes
            for p in large_primes:
                if current_sum + Fraction(1, p) <= 1:
                    current_set.append(p)
                    current_sum += Fraction(1, p)

            key = tuple(sorted(current_set))
            if key not in seen_sets:
                seen_sets.add(key)
                candidate_sets.append(current_set)

    # Also try some targeted swaps: replace some medium primes with smaller ones
    # e.g., remove 13 and add 11; remove 13,17 and add 7; etc.
    extra_candidates = []

    # Try: seed minus some primes plus 11
    for remove_list in [[53], [59], [53, 59], [61], [67]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 11) <= 1:
            s.append(11)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)

    # Try: seed minus some primes plus 7
    for remove_list in [[13], [13, 17], [17], [19]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 7) <= 1:
            s.append(7)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)

    # Try: seed minus some primes plus 5
    for remove_list in [[13, 17], [17, 19], [13, 19], [13, 17, 19]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 5) <= 1:
            s.append(5)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)

    # Try: seed minus some primes plus 3
    for remove_list in [[13, 17, 19, 23], [17, 19, 23, 29], [13, 17, 19, 23, 29]]:
        s = [p for p in seed_set if p not in remove_list]
        s_sum = sum(Fraction(1, p) for p in s)
        if s_sum + Fraction(1, 3) <= 1:
            s.append(3)
            key = tuple(sorted(s))
            if key not in seen_sets:
                seen_sets.add(key)
                extra_candidates.append(s)

    candidate_sets.extend(extra_candidates)

    MAX_L = 100000
    best_L = 0
    best_classes = seed_set and [[p, (i + 1) % p] for i, p in enumerate(seed_set)] or []

    def evaluate_set(P):
        """Improved greedy: assign primes to maximize new coverage at each step."""
        if time.time() > deadline - 5:
            return 0, {}

        P_sorted = sorted(P)
        n_primes = len(P_sorted)
        if n_primes == 0:
            return 0, {}

        covered = bytearray(MAX_L + 1)  # 1-indexed
        residues = {}
        used_primes = set()
        remaining = list(P_sorted)

        for n in range(1, MAX_L + 1):
            if time.time() > deadline - 3:
                break
            if covered[n]:
                continue

            # Among unused primes, pick the one whose best residue covers most new numbers
            best_p = None
            best_r = None
            best_new = -1
            for p in remaining:
                # Try the residue that covers n
                r = n % p
                # Count how many new numbers this would cover
                new_count = 0
                start_k = r if r > 0 else p
                for k in range(start_k, MAX_L + 1, p):
                    if not covered[k]:
                        new_count += 1
                        if new_count > best_new + 100:  # Early exit if clearly better
                            break
                if new_count > best_new:
                    best_new = new_count
                    best_p = p
                    best_r = r

            if best_p is None:
                break

            residues[best_p] = best_r
            used_primes.add(best_p)
            remaining.remove(best_p)

            # Mark all k in [1, MAX_L] with k % best_p == best_r
            start_k = best_r if best_r > 0 else best_p
            for k in range(start_k, MAX_L + 1, best_p):
                covered[k] = 1

        # Compute L
        L = 0
        for i in range(1, MAX_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L, residues

    # Evaluate candidates, prioritizing those with smaller primes first
    # Sort candidates by number of small primes (more small primes likely better)
    def small_prime_count(P):
        return sum(1 for p in P if p <= 11)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), len(P)))

    for P in candidate_sets:
        if time.time() > deadline - 10:
            break

        # Verify budget
        s = sum(Fraction(1, p) for p in P)
        if s > 1:
            continue

        L, residues = evaluate_set(P)
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]

    # Local improvement: try swapping residues of pairs of primes
    if time.time() < deadline - 20 and best_classes:
        current_classes = best_classes[:]
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for i in range(len(current_classes)):
                if time.time() > deadline - 15:
                    break
                for j in range(i + 1, len(current_classes)):
                    if time.time() > deadline - 15:
                        break
                    p1, r1 = current_classes[i]
                    p2, r2 = current_classes[j]
                    # Try swapping residues
                    old_classes = current_classes[:]
                    current_classes[i] = [p1, r2]
                    current_classes[j] = [p2, r1]

                    # Check if this improves coverage
                    covered = bytearray(MAX_L + 1)
                    for p, r in current_classes:
                        start_k = r if r > 0 else p
                        for k in range(start_k, MAX_L + 1, p):
                            covered[k] = 1

                    L_new = 0
                    for k in range(1, min(best_L + 50, MAX_L + 1)):
                        if covered[k]:
                            L_new = k
                        else:
                            break

                    if L_new > best_L:
                        best_L = L_new
                        best_classes = current_classes[:]
                        improved = True
                    else:
                        current_classes = old_classes

    # Fallback: if best_L is 0, use seed
    if best_L == 0:
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Final verification
    final_sum = sum(Fraction(1, p) for p, r in best_classes)
    if final_sum > 1:
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Verify distinct primes
    prime_list = [p for p, r in best_classes]
    if len(prime_list) != len(set(prime_list)):
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    return best_classes
