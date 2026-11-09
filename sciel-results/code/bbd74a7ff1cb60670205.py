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

    def eval_state(A):
        """Return (score, sums_set, prods_set) for a sorted list A."""
        a = A
        sums = set()
        prods = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                prods.add(a[i] * a[j])
        return max(len(sums), len(prods)), sums, prods

    def delta_eval(A_sorted, sums, prods, old_idx, old_val, new_val):
        """Incremental evaluation: replace A_sorted[old_idx] with new_val.
        Returns (new_score, new_sums, new_prods, new_A_sorted) or None if invalid."""
        n_local = len(A_sorted)
        # Check if new_val already in set (excluding old_val position)
        for k in range(n_local):
            if k != old_idx and A_sorted[k] == new_val:
                return None
        # Build new sorted list
        new_A = A_sorted[:old_idx] + A_sorted[old_idx+1:]
        # Insert new_val in sorted position
        lo, hi = 0, len(new_A)
        while lo < hi:
            mid = (lo + hi) // 2
            if new_A[mid] < new_val:
                lo = mid + 1
            else:
                hi = mid
        new_A.insert(lo, new_val)
        new_idx = lo
        # Compute sums/prods for pairs involving new_val (with all elements including itself)
        new_sums = set()
        new_prods = set()
        for j in range(n_local):
            new_sums.add(new_val + new_A[j])
            new_prods.add(new_val * new_A[j])
        # Remove contributions of old_val: for each j, old_val + A[j] and old_val * A[j]
        # But we need to be careful: when j == old_idx, the pair is (old_val, old_val)
        # For j != old_idx, the pair (old_val, A[j]) appears once in the original sets
        # We remove them, but they might still be produced by other pairs, so we use reference counting
        # Simpler approach: rebuild from scratch but only for the 39 pairs + keep the rest
        # Actually, let's just do a full recompute of the 39 pairs and merge with existing minus old pairs
        
        # Better: compute all pairs not involving old_val, then add pairs involving new_val
        # Pairs not involving old_idx: for i < j, i != old_idx, j != old_idx
        remaining_sums = set()
        remaining_prods = set()
        for i in range(n_local):
            if i == old_idx:
                continue
            for j in range(i + 1, n_local):
                if j == old_idx:
                    continue
                remaining_sums.add(A_sorted[i] + A_sorted[j])
                remaining_prods.add(A_sorted[i] * A_sorted[j])
        # Add pairs involving new_val
        final_sums = remaining_sums | new_sums
        final_prods = remaining_prods | new_prods
        score = max(len(final_sums), len(final_prods))
        return score, final_sums, final_prods, new_A

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

    # Precompute smooth number candidates for structured moves
    smooth_pool = set()
    for a in range(14):
        v = 2 ** a
        for b in range(12):
            v2 = v * (3 ** b)
            for c in range(10):
                v3 = v2 * (5 ** c)
                if v3 <= max_val:
                    smooth_pool.add(v3)
    smooth_pool = sorted(smooth_pool)

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
        current_score, current_sums, current_prods = eval_state(current)

        # Local search with hill climbing using incremental eval
        no_improve = 0
        max_no_improve = 2000

        while no_improve < max_no_improve and time.time() < deadline - 2:
            improved = False
            move_type = random.random()

            if move_type < 0.35:
                # Replace one element with random value
                i = random.randint(0, n - 1)
                new_val = random.randint(1, max_val)
                if new_val in current:
                    no_improve += 1
                    continue
                old_val = current[i]
                result = delta_eval(current, current_sums, current_prods, i, old_val, new_val)
                if result is None:
                    no_improve += 1
                    continue
                new_score, new_sums, new_prods, new_A = result
                if new_score < current_score:
                    current = new_A
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    improved = True
                    no_improve = 0

            elif move_type < 0.55:
                # Replace with a value near an existing element
                i = random.randint(0, n - 1)
                base = current[i]
                delta = random.randint(-20, 20)
                new_val = base + delta
                if new_val < 1 or new_val > max_val or new_val in current:
                    no_improve += 1
                    continue
                old_val = current[i]
                result = delta_eval(current, current_sums, current_prods, i, old_val, new_val)
                if result is None:
                    no_improve += 1
                    continue
                new_score, new_sums, new_prods, new_A = result
                if new_score < current_score:
                    current = new_A
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    improved = True
                    no_improve = 0

            elif move_type < 0.70:
                # Replace with a multiple of an existing element
                i = random.randint(0, n - 1)
                factor = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
                new_val = current[i] * factor
                if new_val > max_val or new_val in current:
                    no_improve += 1
                    continue
                old_val = current[i]
                result = delta_eval(current, current_sums, current_prods, i, old_val, new_val)
                if result is None:
                    no_improve += 1
                    continue
                new_score, new_sums, new_prods, new_A = result
                if new_score < current_score:
                    current = new_A
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    improved = True
                    no_improve = 0

            elif move_type < 0.85:
                # Replace with a divisor of an existing element
                i = random.randint(0, n - 1)
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
                old_val = current[i]
                result = delta_eval(current, current_sums, current_prods, i, old_val, new_val)
                if result is None:
                    no_improve += 1
                    continue
                new_score, new_sums, new_prods, new_A = result
                if new_score < current_score:
                    current = new_A
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    improved = True
                    no_improve = 0

            elif move_type < 0.92:
                # Replace with a value that is a product of two existing elements
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    no_improve += 1
                    continue
                new_val = current[i] * current[j]
                if new_val > max_val or new_val in current:
                    no_improve += 1
                    continue
                old_val = current[i]
                result = delta_eval(current, current_sums, current_prods, i, old_val, new_val)
                if result is None:
                    no_improve += 1
                    continue
                new_score, new_sums, new_prods, new_A = result
                if new_score < current_score:
                    current = new_A
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    improved = True
                    no_improve = 0
            else:
                # Replace with a random smooth number (multiplicative lattice move)
                i = random.randint(0, n - 1)
                new_val = random.choice(smooth_pool)
                if new_val in current:
                    no_improve += 1
                    continue
                old_val = current[i]
                result = delta_eval(current, current_sums, current_prods, i, old_val, new_val)
                if result is None:
                    no_improve += 1
                    continue
                new_score, new_sums, new_prods, new_A = result
                if new_score < current_score:
                    current = new_A
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
        current_score, current_sums, current_prods = eval_state(current)
        temp = 8.0
        cooling = 0.9993

        for iteration in range(80000):
            if time.time() > deadline - 2:
                break
            temp *= cooling

            # Mix random and smooth-number moves in SA
            if random.random() < 0.4:
                new_val = random.choice(smooth_pool)
            else:
                new_val = random.randint(1, max_val)

            i = random.randint(0, n - 1)
            if new_val in current:
                continue
            old_val = current[i]
            result = delta_eval(current, current_sums, current_prods, i, old_val, new_val)
            if result is None:
                continue
            new_score, new_sums, new_prods, new_A = result
            diff = new_score - current_score
            if diff < 0 or random.random() < np.exp(-diff / max(temp, 0.01)):
                current = new_A
                current_score = new_score
                current_sums = new_sums
                current_prods = new_prods
                if current_score < best_score:
                    best_score = current_score
                    best_set = current[:]

    # --- Final refinement: exhaustive single-element replacement from structured candidates ---
    if time.time() < deadline - 3 and best_set is not None:
        current = best_set[:]
        current_score, current_sums, current_prods = eval_state(current)
        
        # Build a rich candidate pool for replacement values
        structured_vals = set()
        # All small integers
        for x in range(1, 200):
            structured_vals.add(x)
        # Multiples
        for x in range(1, 200):
            for m in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]:
                if x * m <= max_val:
                    structured_vals.add(x * m)
        # Smooth numbers (2,3,5,7-smooth)
        for a in range(10):
            v = 2 ** a
            for b in range(10):
                v2 = v * (3 ** b)
                for c in range(8):
                    v3 = v2 * (5 ** c)
                    for d in range(6):
                        v4 = v3 * (7 ** d)
                        if v4 <= max_val:
                            structured_vals.add(v4)
        # Products of pairs from current set
        for i in range(n):
            for j in range(n):
                p = current[i] * current[j]
                if p <= max_val:
                    structured_vals.add(p)
        # Sums of pairs from current set
        for i in range(n):
            for j in range(i, n):
                s = current[i] + current[j]
                if s <= max_val:
                    structured_vals.add(s)
        # Near current values
        for v in current:
            for d in range(-50, 51):
                nv = v + d
                if 1 <= nv <= max_val:
                    structured_vals.add(nv)
        
        structured_vals = sorted(structured_vals)
        current_set = set(current)
        
        # Iteratively try improvements (restart after each improvement)
        improved_any = True
        while improved_any and time.time() < deadline - 1:
            improved_any = False
            for i in range(n):
                if time.time() > deadline - 1:
                    break
                old_val = current[i]
                for new_val in structured_vals:
                    if new_val == old_val or new_val in current_set:
                        continue
                    result = delta_eval(current, current_sums, current_prods, i, old_val, new_val)
                    if result is None:
                        continue
                    new_score, new_sums, new_prods, new_A = result
                    if new_score < current_score:
                        current = new_A
                        current_score = new_score
                        current_sums = new_sums
                        current_prods = new_prods
                        current_set = set(current)
                        if current_score < best_score:
                            best_score = current_score
                            best_set = current[:]
                        improved_any = True
                        break  # Restart inner loop after improvement

    # --- Second polish: try all pairs of structured values for each position ---
    if time.time() < deadline - 1.5 and best_set is not None:
        current = best_set[:]
        current_score, current_sums, current_prods = eval_state(current)
        current_set = set(current)
        
        # Focus on values near existing elements and smooth numbers
        for i in range(n):
            if time.time() > deadline - 0.5:
                break
            old_val = current[i]
            # Try values in a tight neighborhood
            for new_val in range(max(1, old_val - 30), min(max_val + 1, old_val + 31)):
                if new_val == old_val or new_val in current_set:
                    continue
                result = delta_eval(current, current_sums, current_prods, i, old_val, new_val)
                if result is None:
                    continue
                new_score, new_sums, new_prods, new_A = result
                if new_score < current_score:
                    current = new_A
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    current_set = set(current)
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current[:]

    # --- Final polish: pair swaps between positions ---
    if time.time() < deadline - 1.0 and best_set is not None:
        current = best_set[:]
        current_score, current_sums, current_prods = eval_state(current)
        current_set = set(current)

        # Try swapping each pair of elements with smooth-number replacements
        for i in range(n):
            if time.time() > deadline - 0.3:
                break
            for j in range(i + 1, n):
                if time.time() > deadline - 0.3:
                    break
                # Try replacing both i and j simultaneously with smooth numbers
                for nv_i in smooth_pool[:50]:
                    if nv_i in current_set or nv_i == current[j]:
                        continue
                    for nv_j in smooth_pool[:50]:
                        if nv_j in current_set or nv_j == current[i] or nv_j == nv_i:
                            continue
                        # Build new set
                        new_set = set(current)
                        new_set.discard(current[i])
                        new_set.discard(current[j])
                        new_set.add(nv_i)
                        new_set.add(nv_j)
                        if len(new_set) != n:
                            continue
                        new_A = sorted(new_set)
                        sc = fast_eval(new_A)
                        if sc < current_score:
                            current = new_A
                            current_score = sc
                            current_set = set(current)
                            current_score, current_sums, current_prods = eval_state(current)
                            if current_score < best_score:
                                best_score = current_score
                                best_set = current[:]
                            break
                    if time.time() > deadline - 0.3:
                        break
                if time.time() > deadline - 0.3:
                    break

    return best_set
# EVOLVE-BLOCK-END
