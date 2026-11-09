"""Model-and-optimize approach for integer sum-product extremal sets."""
import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 82.0  # Safety margin before 90s kill
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using vectorized numpy."""
        a = np.array(sorted(A), dtype=np.int64)
        sumset = np.unique(a[:, None] + a[None, :])
        prodset = np.unique(a[:, None] * a[None, :])
        return max(len(sumset), len(prodset))

    def fast_eval(A):
        """Faster evaluation using sorted list approach."""
        a = sorted(A)
        sums = set()
        prods = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                prods.add(a[i] * a[j])
        return max(len(sums), len(prods))

    def eval_with_components(A):
        """Return (score, sums_set, prods_set) for a set A."""
        a = sorted(A)
        sums = set()
        prods = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                prods.add(a[i] * a[j])
        return max(len(sums), len(prods)), sums, prods

    def incremental_eval(A, idx_removed, new_val, sums_base, prods_base):
        """
        Given the sums/prods sets for A with element at idx_removed removed,
        compute the new sums/prods when new_val is added.
        A is sorted, idx_removed is the index of the element to replace.
        Returns (new_score, new_sums, new_prods).
        """
        a = list(A)
        old_val = a[idx_removed]
        # Remove old contributions: pairs (old_val, a[j]) for all j
        # and (a[i], old_val) for all i, plus (old_val, old_val)
        for j in range(len(a)):
            if j == idx_removed:
                continue
            s = old_val + a[j]
            p = old_val * a[j]
            if s in sums_base:
                sums_base.discard(s)
            if p in prods_base:
                prods_base.discard(p)
        # Also remove (old_val, old_val) if it was there
        s_oo = old_val + old_val
        p_oo = old_val * old_val
        sums_base.discard(s_oo)
        prods_base.discard(p_oo)
        
        # Now insert new_val
        a[idx_removed] = new_val
        a.sort()
        
        # Add new contributions: pairs (new_val, a[j]) for all j
        for j in range(len(a)):
            if j == idx_removed:
                continue
            s = new_val + a[j]
            p = new_val * a[j]
            sums_base.add(s)
            prods_base.add(p)
        # Add (new_val, new_val)
        sums_base.add(new_val + new_val)
        prods_base.add(new_val * new_val)
        
        return max(len(sums_base), len(prods_base)), sums_base, prods_base

    def score_and_components(A):
        """Return (score, len(sums), len(prods), sums, prods) for a set A."""
        a = sorted(A)
        sums = set()
        prods = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                prods.add(a[i] * a[j])
        return max(len(sums), len(prods)), len(sums), len(prods), sums, prods

    # --- Strategy 1: Generate diverse initial candidates ---
    candidates = []

    # Sequential
    candidates.append(list(range(1, 21)))

    # Geometric progression ratio 2
    candidates.append([2**i for i in range(20)])

    # Geometric progressions with other ratios
    for ratio in [3, 4, 5, 6, 7, 8, 10]:
        A = [ratio**i for i in range(20)]
        if max(A) <= max_val:
            candidates.append(A)

    # Arithmetic progressions
    for a in range(1, 15):
        for d in range(1, 10):
            A_ap = [a + i * d for i in range(n)]
            if max(A_ap) <= max_val:
                candidates.append(A_ap)

    # Divisor sets of highly composite numbers
    def get_divisors(m):
        divs = set()
        i = 1
        while i * i <= m:
            if m % i == 0:
                divs.add(i)
                divs.add(m // i)
            i += 1
        return sorted(divs)

    highly_composite = [720, 2520, 5040, 10080, 15120, 20160, 27720, 45360, 50400,
                        55440, 83160, 110880, 166320, 221760, 277200, 332640, 498960,
                        554400, 665280, 720720, 1081080, 1441440, 2162160, 2882880,
                        3603600, 4324320, 6486480, 7207200, 10810800, 14414400]

    for N in highly_composite:
        if time.time() > deadline - 10:
            break
        divs = [d for d in get_divisors(N) if d <= max_val]
        if len(divs) < n:
            continue
        candidates.append(divs[:n])
        if len(divs) >= 2 * n:
            candidates.append(divs[-n:])
        if len(divs) >= 40:
            mid = len(divs) // 2
            candidates.append(divs[mid - n // 2:mid + n // 2])
        if len(divs) >= 40:
            candidates.append(divs[:10] + divs[-10:])

    # Multiplicative lattice sets (smooth numbers)
    def gen_smooth_numbers(primes, max_exp, limit):
        vals = set([1])
        for p in primes:
            new_vals = set()
            for v in vals:
                cur = v
                for e in range(max_exp):
                    cur *= p
                    if cur > limit:
                        break
                    new_vals.add(cur)
            vals |= new_vals
        return sorted([v for v in vals if v <= limit])

    # Various smooth number sets
    smooth_23 = gen_smooth_numbers([2, 3], 20, max_val)
    if len(smooth_23) >= n:
        candidates.append(smooth_23[:n])
        if len(smooth_23) >= 40:
            step = len(smooth_23) // n
            candidates.append(smooth_23[::step][:n])

    smooth_235 = gen_smooth_numbers([2, 3, 5], 15, max_val)
    if len(smooth_235) >= n:
        candidates.append(smooth_235[:n])
        if len(smooth_235) >= 40:
            step = len(smooth_235) // n
            candidates.append(smooth_235[::step][:n])

    smooth_2357 = gen_smooth_numbers([2, 3, 5, 7], 12, max_val)
    if len(smooth_2357) >= n:
        candidates.append(smooth_2357[:n])

    # Numbers with many divisors (highly composite-like)
    div_counts = [0] * (max_val + 1)
    for i in range(1, int(max_val**0.5) + 1):
        for j in range(i, max_val + 1, i):
            div_counts[j] += 1
            if i != j // i:
                div_counts[j] += 1

    top_div = sorted(range(1, max_val + 1), key=lambda x: div_counts[x], reverse=True)
    candidates.append(top_div[:n])
    if len(top_div) >= 40:
        candidates.append(top_div[:10] + top_div[10:20])
        candidates.append(top_div[:5] + top_div[5:10] + top_div[15:20])

    # Mix of small and large numbers
    small = list(range(1, 11))
    large = [9000 + i for i in range(10)]
    if all(x <= max_val for x in large):
        candidates.append(small + large)

    # Powers and near-powers
    powers = []
    for base in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
        v = 1
        while v <= max_val:
            powers.append(v)
            v *= base
    powers = sorted(set(powers))
    if len(powers) >= n:
        candidates.append(powers[:n])
        if len(powers) >= 40:
            step = len(powers) // n
            candidates.append(powers[::step][:n])

    # Evaluate all candidates
    valid_candidates = []
    for A in candidates:
        if time.time() > deadline - 10:
            break
        if len(set(A)) != n:
            continue
        if max(A) > max_val or min(A) < 1:
            continue
        A_sorted = sorted(A)
        score = fast_eval(A_sorted)
        if score < best_score:
            best_score = score
            best_set = A_sorted[:]
        if score < 300:  # Keep only good candidates for local search
            valid_candidates.append((A_sorted, score))

    # --- Local search from best and other good starts ---
    random.seed(42)

    # Build list of starting points
    starts = [(best_set, best_score)]
    for A_s, sc in valid_candidates:
        if A_s != best_set:
            starts.append((A_s, sc))
    starts = starts[:15]

    for start_set, start_score in starts:
        if time.time() > deadline - 2:
            break

        current = start_set[:]
        current_score = start_score
        # Precompute sums and prods for incremental evaluation
        _, current_sums, current_prods = eval_with_components(current)

        # Local search with hill climbing using incremental eval
        no_improve = 0
        max_no_improve = 500

        while no_improve < max_no_improve and time.time() < deadline - 2:
            improved = False
            move_type = random.random()
            i = random.randint(0, n - 1)

            if move_type < 0.35:
                # Replace one element with random value
                new_val = random.randint(1, max_val)
                if new_val in current:
                    no_improve += 1
                    continue
            elif move_type < 0.55:
                # Replace with a value near an existing element
                base = current[i]
                delta = random.randint(-10, 10)
                new_val = base + delta
                if new_val < 1 or new_val > max_val or new_val in current:
                    no_improve += 1
                    continue
            elif move_type < 0.70:
                # Replace with a multiple of an existing element
                factor = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
                new_val = current[i] * factor
                if new_val > max_val or new_val in current:
                    no_improve += 1
                    continue
            elif move_type < 0.85:
                # Replace with a divisor of an existing element
                val = current[i]
                divs = []
                for d in range(1, int(val**0.5) + 1):
                    if val % d == 0:
                        divs.append(d)
                        if d != val // d:
                            divs.append(val // d)
                if not divs:
                    no_improve += 1
                    continue
                new_val = random.choice(divs)
                if new_val in current:
                    no_improve += 1
                    continue
            else:
                # Replace with a value that is a product of two existing elements
                j = random.randint(0, n - 1)
                if i == j:
                    no_improve += 1
                    continue
                new_val = current[i] * current[j]
                if new_val > max_val or new_val in current:
                    no_improve += 1
                    continue

            # Use incremental evaluation
            # Make copies of sums and prods for testing
            sums_test = set(current_sums)
            prods_test = set(current_prods)
            new_score, new_sums, new_prods = incremental_eval(current, i, new_val, sums_test, prods_test)

            if new_score < current_score:
                current[i] = new_val
                current.sort()
                current_score = new_score
                current_sums = new_sums
                current_prods = new_prods
                improved = True
                no_improve = 0

            if improved:
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]

            if not improved:
                no_improve += 1

    # --- Simulated annealing from the best found so far ---
    if time.time() < deadline - 5 and best_set is not None:
        current = best_set[:]
        current_score = best_score
        _, current_sums, current_prods = eval_with_components(current)
        temp = 5.0
        cooling = 0.998

        for iteration in range(30000):
            if time.time() > deadline - 2:
                break
            temp *= cooling

            i = random.randint(0, n - 1)
            new_val = random.randint(1, max_val)
            if new_val in current:
                continue

            # Use incremental evaluation
            sums_test = set(current_sums)
            prods_test = set(current_prods)
            new_score, new_sums, new_prods = incremental_eval(current, i, new_val, sums_test, prods_test)
            diff = new_score - current_score
            if diff < 0 or random.random() < np.exp(-diff / max(temp, 0.01)):
                current[i] = new_val
                current.sort()
                current_score = new_score
                current_sums = new_sums
                current_prods = new_prods
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]

    # --- Final refinement: try all single-element replacements from a restricted set ---
    if time.time() < deadline - 3 and best_set is not None:
        current = best_set[:]
        current_score = best_score
        _, current_sums, current_prods = eval_with_components(current)
        # Try replacing each element with values from a structured set
        structured_vals = set()
        for x in range(1, 100):
            structured_vals.add(x)
        for x in range(1, 100):
            for m in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                if x * m <= max_val:
                    structured_vals.add(x * m)
        for v in range(1, max_val + 1, 100):
            structured_vals.add(v)
        structured_vals = sorted(structured_vals)

        for i in range(n):
            if time.time() > deadline - 1:
                break
            old_val = current[i]
            for new_val in structured_vals:
                if new_val == old_val or new_val in current:
                    continue
                # Use incremental evaluation
                sums_test = set(current_sums)
                prods_test = set(current_prods)
                new_score, new_sums, new_prods = incremental_eval(current, i, new_val, sums_test, prods_test)
                if new_score < current_score:
                    current[i] = new_val
                    current.sort()
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current[:]
                    break  # Restart inner loop after improvement

    return best_set
# EVOLVE-BLOCK-END
