import time
from fractions import Fraction
import itertools

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
        """Greedy residue assignment: for n=1..MAX_L, if uncovered, assign a prime."""
        if time.time() > deadline - 5:
            return 0, {}

        P_sorted = sorted(P)
        n_primes = len(P_sorted)
        if n_primes == 0:
            return 0, {}

        covered = bytearray(MAX_L + 1)  # 1-indexed, covered[i] = 1 if i is covered
        residues = {}
        used_primes = set()

        for n in range(1, MAX_L + 1):
            if time.time() > deadline - 3:
                break
            if covered[n]:
                continue

            # Find smallest unused prime to assign
            assigned = False
            for p in P_sorted:
                if p in used_primes:
                    continue
                r = n % p
                residues[p] = r
                used_primes.add(p)

                # Mark all k in [1, MAX_L] with k % p == r
                start_k = r if r > 0 else p
                for k in range(start_k, MAX_L + 1, p):
                    covered[k] = 1

                assigned = True
                break

            if not assigned:
                break

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
