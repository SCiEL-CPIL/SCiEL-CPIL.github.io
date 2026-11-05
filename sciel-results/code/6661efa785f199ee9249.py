import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Sieve primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)

    # Precompute prime indices for quick lookup
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    def calc_L(classes, max_check=100000):
        """Calculate the largest L such that 1..L are all covered."""
        if not classes:
            return 0
        covered = np.zeros(max_check + 1, dtype=bool)
        for p, r in classes:
            start = p if r == 0 else r
            if start <= max_check:
                idx = np.arange(start, max_check + 1, p)
                covered[idx] = True
        # Find first uncovered
        uncovered = np.where(~covered[1:])[0]
        if len(uncovered) == 0:
            return max_check
        return int(uncovered[0])

    def check_coverage(classes, L):
        """Check if 1..L are all covered by the given classes."""
        if not classes:
            return L <= 0
        covered = np.zeros(L + 1, dtype=bool)
        for p, r in classes:
            start = p if r == 0 else r
            if start <= L:
                idx = np.arange(start, L + 1, p)
                covered[idx] = True
        return bool(np.all(covered[1:]))

    def budget_ok(primes_set):
        """Check if sum of 1/p for primes in set is <= 1."""
        s = Fraction(0)
        for p in primes_set:
            s += Fraction(1, p)
        return s <= 1

    def greedy_assign_by_conflict(primes_set, L_target):
        """
        Assign residues to primes to maximize coverage of 1..L_target.
        Process primes in order of 'constraint' - those that cover fewer options first.
        """
        primes_sorted = sorted(primes_set, key=lambda p: L_target // p)  # Fewer hits first
        residues = {}
        covered = np.zeros(L_target + 1, dtype=bool)

        for p in primes_sorted:
            best_r = 0
            best_gain = -1
            # Sample residues if p is large
            if p > 50:
                candidates_r = list(range(p))
                np.random.shuffle(candidates_r)
                candidates_r = candidates_r[:min(p, 200)]
            else:
                candidates_r = list(range(p))

            for r in candidates_r:
                start = p if r == 0 else r
                if start > L_target:
                    continue
                idx = np.arange(start, L_target + 1, p)
                gain = np.sum(~covered[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r

            residues[p] = best_r
            start = p if best_r == 0 else best_r
            if start <= L_target:
                idx = np.arange(start, L_target + 1, p)
                covered[idx] = True

        return [[p, residues[p]] for p in primes_sorted]

    def simulated_annealing(classes, L_target, max_iter=500, t_start=10.0, t_end=0.1):
        """
        Simulated annealing on residue assignments to maximize coverage of 1..L_target.
        """
        current_classes = [list(c) for c in classes]
        covered = np.zeros(L_target + 1, dtype=bool)
        for p, r in current_classes:
            start = p if r == 0 else r
            if start <= L_target:
                idx = np.arange(start, L_target + 1, p)
                covered[idx] = True
        current_score = np.sum(covered[1:])  # Number of covered integers in 1..L_target

        best_classes = [list(c) for c in current_classes]
        best_score = current_score

        n_primes = len(current_classes)
        if n_primes == 0:
            return current_classes, 0

        for it in range(max_iter):
            t = t_start * (t_end / t_start) ** (it / max_iter)
            # Pick a random prime to change
            i = np.random.randint(n_primes)
            p = current_classes[i][0]
            old_r = current_classes[i][1]
            new_r = np.random.randint(p)

            # Compute delta in coverage
            # Remove old residue
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r

            # Count how many positions are covered ONLY by old residue
            # and how many are newly covered by new residue
            old_positions = np.arange(old_start, L_target + 1, p) if old_start <= L_target else np.array([], dtype=int)
            new_positions = np.arange(new_start, L_target + 1, p) if new_start <= L_target else np.array([], dtype=int)

            # Positions that were covered by old but not by new
            only_old = set(old_positions.tolist()) - set(new_positions.tolist())
            # Positions that will be covered by new but not by old
            only_new = set(new_positions.tolist()) - set(old_positions.tolist())

            # For only_old: check if still covered by other primes
            delta = 0
            for pos in only_old:
                if pos >= 1 and pos <= L_target:
                    # Check if covered by other primes
                    still_covered = False
                    for j, (pj, rj) in enumerate(current_classes):
                        if j == i:
                            continue
                        s_j = pj if rj == 0 else rj
                        if s_j <= pos and (pos - s_j) % pj == 0:
                            still_covered = True
                            break
                    if not still_covered:
                        delta -= 1

            for pos in only_new:
                if pos >= 1 and pos <= L_target:
                    # Check if not already covered by other primes
                    already_covered = False
                    for j, (pj, rj) in enumerate(current_classes):
                        if j == i:
                            continue
                        s_j = pj if rj == 0 else rj
                        if s_j <= pos and (pos - s_j) % pj == 0:
                            already_covered = True
                            break
                    if not already_covered:
                        delta += 1

            new_score = current_score + delta

            if new_score > current_score or np.random.random() < np.exp(delta / max(t, 1e-10)):
                current_classes[i][1] = new_r
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_classes = [list(c) for c in current_classes]

        # Recompute actual L for best
        L = calc_L(best_classes, L_target)
        return best_classes, L

    def multi_start_search(primes_set, L_target, n_starts=3, sa_iters=300):
        """Try multiple random starts and simulated annealing."""
        best_classes = None
        best_L = 0

        for s in range(n_starts):
            # Random initial assignment
            np.random.seed(s * 12345 + 678)
            classes = [[p, np.random.randint(p)] for p in primes_set]

            # Greedy improvement
            classes = greedy_assign_by_conflict(primes_set, L_target)

            # Simulated annealing
            classes, L = simulated_annealing(classes, L_target, max_iter=sa_iters)

            if L > best_L:
                best_L = L
                best_classes = classes

        return best_classes, best_L

    # Main: Binary search on L, with multiple prime set candidates
    best_classes = []
    best_L = 0

    # Generate candidate prime sets
    def smart_trim(prime_list):
        sorted_asc = sorted(prime_list)
        result = []
        current_sum = Fraction(0)
        for p in sorted_asc:
            if current_sum + Fraction(1, p) <= 1:
                result.append(p)
                current_sum += Fraction(1, p)
        return result

    # Various prime set strategies
    candidate_sets = []

    # Small primes only
    for min_p in [2, 3, 5, 7, 11, 13]:
        for max_p in [1000, 500, 200, 100, 50, 30]:
            s = [p for p in primes if min_p <= p <= max_p]
            trimmed = smart_trim(s)
            if trimmed:
                candidate_sets.append(trimmed)

    # Include specific small primes
    for extra in [[2], [3], [2,3], [5], [2,5], [3,5], [2,3,5]]:
        for min_p in [7, 11, 13, 17]:
            s = extra + [p for p in primes if p >= min_p and p not in extra]
            trimmed = smart_trim(s)
            if trimmed:
                candidate_sets.append(trimmed)

    # Remove duplicates
    unique_sets = []
    seen = set()
    for cs in candidate_sets:
        key = frozenset(cs)
        if key not in seen:
            seen.add(key)
            unique_sets.append(cs)

    # Binary search on L for each candidate set
    L_low = 100
    L_high = 100000

    # First, do a quick pass to find a reasonable L range
    for cs in unique_sets[:5]:  # Quick check on first few
        if time.time() > deadline - 30:
            break
        classes = greedy_assign_by_conflict(cs, L_low)
        L = calc_L(classes, L_low)
        if L > best_L:
            best_L = L
            best_classes = classes

    # Now binary search
    for cs in unique_sets:
        if time.time() > deadline - 20:
            break

        # Quick greedy to get a baseline
        classes = greedy_assign_by_conflict(cs, 5000)
        L_base = calc_L(classes, 5000)

        if L_base > best_L:
            best_L = L_base
            best_classes = classes

        # Try to improve with simulated annealing
        if L_base > 100 and time.time() < deadline - 25:
            classes_sa, L_sa = simulated_annealing(classes, min(L_base + 1000, 20000), max_iter=200)
            if L_sa > best_L:
                best_L = L_sa
                best_classes = classes_sa

    # Final: try to extend the best solution found
    if best_classes and best_L > 100:
        # Try to extend by adding more primes or adjusting residues
        primes_used = set(p for p, r in best_classes)
        available_primes = [p for p in primes if p not in primes_used]

        # Try adding one more prime at a time
        for p_new in sorted(available_primes, reverse=True):
            if time.time() > deadline - 10:
                break
            # Check budget
            test_set = list(primes_used) + [p_new]
            if not budget_ok(test_set):
                continue
            # Try all residues for the new prime
            best_r = 0
            best_L_try = best_L
            for r in range(p_new):
                test_classes = [list(c) for c in best_classes] + [[p_new, r]]
                L_try = calc_L(test_classes, min(best_L + p_new * 2, 20000))
                if L_try > best_L_try:
                    best_L_try = L_try
                    best_r = r
            if best_L_try > best_L:
                best_L = best_L_try
                best_classes = [list(c) for c in best_classes] + [[p_new, best_r]]
                primes_used.add(p_new)

        # Final simulated annealing on the extended set
        if time.time() < deadline - 15:
            classes_final, L_final = simulated_annealing(best_classes, min(best_L + 2000, 50000), max_iter=300)
            if L_final > best_L:
                best_L = L_final
                best_classes = classes_final

    # Verify final result
    if best_classes:
        # Verify budget
        s = Fraction(0)
        primes_used = set()
        for p, r in best_classes:
            if p in primes_used:
                return [[2, 1]]  # Fallback
            primes_used.add(p)
            if r < 0 or r >= p:
                return [[2, 1]]
            s += Fraction(1, p)
        if s > 1:
            return [[2, 1]]

        # Verify coverage
        if not check_coverage(best_classes, best_L):
            # Find actual L
            actual_L = calc_L(best_classes, 10000)
            if actual_L > 0:
                best_L = actual_L
            else:
                return [[2, 1]]

    if not best_classes:
        return [[2, 1]]

    return best_classes
