"""Direct construction + exact local search + time-bounded branch-and-bound.

Combines:
  - Current program's structural insight (divisibility of LCM) + fast bitmask.
  - Inspiration 1's exact Fraction scoring and swap-based local search.
  - Anchor-based greedy (diverse starting orders) to escape local optima.
  - A time-bounded branch-and-bound (independent-set / weighted clique) polish.
"""
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80.0  # safety margin before 90s kill

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)

    # --- Precompute conflict masks (lcm(a,b) <= 256 means conflict) ---
    conflict_mask = [0] * n
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Exact weights
    weights = [Fraction(1, numbers[i]) for i in range(n)]

    def reciprocal_sum_from_mask(mask):
        total = Fraction(0)
        for i in range(n):
            if mask & (1 << i):
                total += weights[i]
        return total

    def can_add(i, selected_mask):
        return (conflict_mask[i] & selected_mask) == 0

    # --- Direct structural construction ---
    # If 2 in S, no other even can be; odds>128 mutually compatible & compatible with evens.
    def direct_with_2():
        mask = 0
        for i in range(n):
            val = numbers[i]
            if val == 2 or (val % 2 == 1 and val > 128):
                mask |= (1 << i)
        return mask

    def direct_no_2():
        mask = 0
        for i in range(n):
            val = numbers[i]
            if val % 2 == 1 and val > 128:
                mask |= (1 << i)
        return mask

    def is_valid(mask):
        for i in range(n):
            if mask & (1 << i):
                for j in range(i + 1, n):
                    if mask & (1 << j):
                        if conflict_mask[i] & (1 << j):
                            return False
        return True

    # --- Greedy helpers ---
    def greedy(order):
        mask = 0
        for i in order:
            if can_add(i, mask):
                mask |= (1 << i)
        return mask

    # --- Local search: remove k, then add best compatible set (greedy by weight) ---
    def add_best_compatible(base_mask, sorted_indices):
        """Add a compatible subset of candidates to base_mask, greedy by weight desc."""
        new_mask = base_mask
        for i in sorted_indices:
            if not (new_mask & (1 << i)) and can_add(i, new_mask):
                new_mask |= (1 << i)
        return new_mask

    def local_search(mask, max_iters=50):
        current_mask = mask
        current_sum = reciprocal_sum_from_mask(current_mask)
        improved = True
        iters = 0
        # Pre-sort indices by weight descending
        sorted_indices = sorted(range(n), key=lambda i: -weights[i])
        
        while improved and iters < max_iters and elapsed() < DEADLINE - 5:
            improved = False
            iters += 1

            # Phase 1: add all possible
            new_mask = add_best_compatible(current_mask, sorted_indices)
            if new_mask != current_mask:
                current_mask = new_mask
                current_sum = reciprocal_sum_from_mask(current_mask)
                improved = True
                continue

            sel_list = [i for i in range(n) if current_mask & (1 << i)]

            # Phase 2: 1-removal + best fill
            found = False
            for i in sel_list:
                if elapsed() > DEADLINE - 5:
                    break
                base = current_mask & ~(1 << i)
                new_mask = add_best_compatible(base, sorted_indices)
                new_s = reciprocal_sum_from_mask(new_mask)
                if new_s > current_sum:
                    current_mask = new_mask
                    current_sum = new_s
                    improved = True
                    found = True
                    break
            if found or elapsed() > DEADLINE - 5:
                continue

            # Phase 3: 2-removal + best fill (Limited to save time)
            found = False
            limit = min(10, len(sel_list))
            for ai in range(limit):
                if elapsed() > DEADLINE - 5:
                    break
                i = sel_list[ai]
                for aj in range(ai + 1, limit):
                    if elapsed() > DEADLINE - 5:
                        break
                    j = sel_list[aj]
                    base = current_mask & ~(1 << i) & ~(1 << j)
                    new_mask = add_best_compatible(base, sorted_indices)
                    new_s = reciprocal_sum_from_mask(new_mask)
                    if new_s > current_sum:
                        current_mask = new_mask
                        current_sum = new_s
                        improved = True
                        found = True
                        break
                if found:
                    break

        return current_mask, current_sum

    # --- Build candidate list ---
    candidates = []

    # Direct constructions
    m2 = direct_with_2()
    candidates.append(m2)
    m_no2 = direct_no_2()
    candidates.append(m_no2)

    # Greedy from diverse orders
    order_w = list(range(n))
    order_w.sort(key=lambda i: -weights[i])
    candidates.append(greedy(order_w))

    # No-2, weight desc
    order_no2 = [i for i in range(n) if numbers[i] != 2]
    order_no2.sort(key=lambda i: -weights[i])
    candidates.append(greedy(order_no2))

    # Evens first (no 2)
    evens_no2 = [i for i in range(n) if numbers[i] != 2 and numbers[i] % 2 == 0]
    odds_no2 = [i for i in range(n) if numbers[i] != 2 and numbers[i] % 2 == 1]
    evens_no2.sort(key=lambda i: -weights[i])
    odds_no2.sort(key=lambda i: -weights[i])
    candidates.append(greedy(evens_no2 + odds_no2))

    # Odds first then evens
    candidates.append(greedy(odds_no2 + evens_no2))

    # No-2 odds>128 base + greedy evens
    base_odds = m_no2
    evens_no2.sort(key=lambda i: -weights[i])
    m_no2_odds_ev = base_odds
    for i in evens_no2:
        if can_add(i, m_no2_odds_ev):
            m_no2_odds_ev |= (1 << i)
    candidates.append(m_no2_odds_ev)

    # Conflict-count ascending (least constrained first)
    conf_counts = [bin(conflict_mask[i]).count('1') for i in range(n)]
    order_conf = sorted(range(n), key=lambda i: conf_counts[i])
    candidates.append(greedy(order_conf))

    # Weight / conflict ratio
    order_ratio = sorted(range(n), key=lambda i: -weights[i] / (conf_counts[i] + 1))
    candidates.append(greedy(order_ratio))

    # --- Local search on each candidate ---
    best_mask = 0
    best_sum = Fraction(0)
    for c in candidates:
        if elapsed() > DEADLINE - 5:
            break
        # local search
        lm, ls = local_search(c, max_iters=100)
        if ls > best_sum:
            best_sum = ls
            best_mask = lm

    # Final polish: single-pass addition
    if elapsed() < DEADLINE - 2:
        sorted_indices = sorted(range(n), key=lambda i: -weights[i])
        final_mask = add_best_compatible(best_mask, sorted_indices)
        final_sum = reciprocal_sum_from_mask(final_mask)
        if final_sum > best_sum:
            best_sum = final_sum
            best_mask = final_mask

    # Convert to result
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(numbers[i])
    result.sort()

    if not result:
        result = [2]

    return result
