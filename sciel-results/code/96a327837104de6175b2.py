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
    def add_best_compatible(base_mask, candidates):
        """Add a compatible subset of candidates to base_mask, greedy by weight desc."""
        new_mask = base_mask
        # sort candidates by weight desc
        cands = sorted(candidates, key=lambda i: -weights[i])
        for i in cands:
            if not (new_mask & (1 << i)) and can_add(i, new_mask):
                new_mask |= (1 << i)
        return new_mask

    def local_search(mask, max_iters=200):
        current_mask = mask
        current_sum = reciprocal_sum_from_mask(current_mask)
        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE - 3:
            improved = False
            iters += 1

            # Phase 1: add all possible
            new_mask = add_best_compatible(current_mask, list(range(n)))
            if new_mask != current_mask:
                current_mask = new_mask
                current_sum = reciprocal_sum_from_mask(current_mask)
                improved = True
                continue

            sel_list = [i for i in range(n) if current_mask & (1 << i)]

            # Phase 2: 1-removal + best fill
            found = False
            for i in sel_list:
                if elapsed() > DEADLINE - 3:
                    break
                base = current_mask & ~(1 << i)
                new_mask = add_best_compatible(base, list(range(n)))
                new_s = reciprocal_sum_from_mask(new_mask)
                if new_s > current_sum:
                    current_mask = new_mask
                    current_sum = new_s
                    improved = True
                    found = True
                    break
            if found or elapsed() > DEADLINE - 3:
                continue

            # Phase 3: 2-removal + best fill
            found = False
            for ai in range(len(sel_list)):
                if elapsed() > DEADLINE - 3:
                    break
                i = sel_list[ai]
                for aj in range(ai + 1, len(sel_list)):
                    if elapsed() > DEADLINE - 3:
                        break
                    j = sel_list[aj]
                    base = current_mask & ~(1 << i) & ~(1 << j)
                    new_mask = add_best_compatible(base, list(range(n)))
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

    # --- Branch and bound (time-bounded) for weighted independent set ---
    # Independent set in conflict graph = clique in complement.
    # We maximize sum of weights. Use Bron-Kerbosch with pivot on complement,
    # but with a weight bound to prune.
    def branch_and_bound(best_mask, best_sum, time_budget):
        # Complement adjacency: comp[i] has bit j set if NOT conflicting (i != j)
        comp = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j and not (conflict_mask[i] & (1 << j)):
                    comp[i] |= (1 << j)

        # Order vertices by weight desc for better bounds
        order = sorted(range(n), key=lambda i: -weights[i])
        # Map to position
        pos = [0] * n
        for idx, v in enumerate(order):
            pos[v] = idx
        # Rebuild comp in ordered index space
        comp_o = [0] * n
        for idx, v in enumerate(order):
            m = 0
            for idx2, v2 in enumerate(order):
                if idx != idx2 and not (conflict_mask[v] & (1 << v2)):
                    m |= (1 << idx2)
            comp_o[idx] = m

        best = [best_sum]
        best_set = [best_mask]
        # candidate mask in ordered space
        # We'll do a DFS: choose vertices in order, maintaining a clique in comp = independent in conflict
        # Standard Bron-Kerbosch with pivot, tracking weight sum.
        # To bound: upper bound = current_sum + sum of weights of remaining candidates.
        # We precompute suffix sums of weights in order.
        suffix = [Fraction(0)] * (n + 1)
        for idx in range(n - 1, -1, -1):
            suffix[idx] = suffix[idx + 1] + weights[order[idx]]

        # DFS state: P (candidate set in ordered-index space as mask), current sum, current set mask (original)
        # We process in order; at each step we can add a vertex from P that is compatible with current set.
        # Use recursion with memoization-free pruning.
        import sys
        sys.setrecursionlimit(10000)

        # We'll use an iterative deepening-ish approach with a node budget.
        node_budget = [0]
        max_nodes = 2000000  # tune

        def dfs(P_mask, current_sum, current_set):
            if elapsed() > DEADLINE - 1 or node_budget[0] > max_nodes:
                return
            node_budget[0] += 1
            # If P empty, record
            if P_mask == 0:
                if current_sum > best[0]:
                    best[0] = current_sum
                    best_set[0] = current_set
                return
            # Bound: find lowest set bit index in P_mask
            # upper = current_sum + sum of weights of vertices in P_mask
            # Compute quickly: sum over set bits
            ub = current_sum
            pm = P_mask
            while pm:
                lb = pm & (-pm)
                idx = lb.bit_length() - 1
                ub += weights[order[idx]]
                pm ^= lb
            if ub <= best[0]:
                return
            # Pick a pivot: vertex in P with max degree in comp within P (to minimize branching)
            # For simplicity, pick first vertex in P (in order).
            # Actually for independent set we want to branch on a vertex and its neighbors.
            # Let's pick vertex v in P. Branch 1: include v. Branch 2: exclude v.
            # Include v: new P = P ∩ comp[v] (in ordered space), new sum += w[v], new set |= v
            # Exclude v: new P = P \ {v}
            # To reduce branching, pick v with high weight first (greedy include).
            # Find a vertex in P, prefer highest weight (lowest idx).
            pm = P_mask
            idx = pm.bit_length() - 1  # highest set bit = lowest weight in order
            # Instead pick lowest set bit = highest weight
            lb = pm & (-pm)
            idx = lb.bit_length() - 1
            v = order[idx]
            v_bit = 1 << idx
            # Branch include
            new_P = P_mask & comp_o[idx]
            dfs(new_P, current_sum + weights[v], current_set | (1 << v))
            # Branch exclude
            dfs(P_mask & ~v_bit, current_sum, current_set)

        # Initial P = all vertices
        all_P = (1 << n) - 1
        dfs(all_P, Fraction(0), 0)
        return best_set[0], best[0]

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

    # --- Branch and bound polish (time-bounded) ---
    if elapsed() < DEADLINE - 10:
        bm, bs = branch_and_bound(best_mask, best_sum, DEADLINE - 10 - elapsed())
        if bs > best_sum:
            best_sum = bs
            best_mask = bm

    # Final polish: single-pass addition
    if elapsed() < DEADLINE - 2:
        final_mask = add_best_compatible(best_mask, list(range(n)))
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
