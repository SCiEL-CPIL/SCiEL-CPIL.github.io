# EVOLVE-BLOCK-START
import time
import random

def construct():
    base = 51975
    deadline = time.time() + 115
    start = time.time()

    # divisors > 1 of 51975 = 3^3*5^2*7*11
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    divisors.sort()
    n_div = len(divisors)

    # Precompute masks: mask[(m, r)] as int of 51975 bits
    mask_cache = {}
    for m in divisors:
        for r in range(m):
            x = r
            v = 0
            while x < base:
                v |= (1 << x)
                x += m
            mask_cache[(m, r)] = v

    # Index divisors for fast lookup
    idx = {m: i for i, m in enumerate(divisors)}

    def score_from_uncovered(uncovered):
        return uncovered.bit_count()

    def greedy(order):
        uncovered = (1 << base) - 1
        chosen = {}  # m -> r
        for m in order:
            best_r = 0
            best_c = -1
            for r in range(m):
                c = (uncovered & mask_cache[(m, r)]).bit_count()
                if c > best_c:
                    best_c = c
                    best_r = r
            chosen[m] = best_r
            uncovered &= ~mask_cache[(m, best_r)]
        return chosen, score_from_uncovered(uncovered)

    def global_greedy():
        """Dynamically pick the class with maximum marginal coverage at each step."""
        uncovered = (1 << base) - 1
        chosen = {}  # m -> r
        remaining = list(divisors)
        while remaining:
            best_m = None
            best_r = 0
            best_c = -1
            for m in remaining:
                for r in range(m):
                    c = (uncovered & mask_cache[(m, r)]).bit_count()
                    if c > best_c:
                        best_c = c
                        best_r = r
                        best_m = m
            if best_m is None:
                break
            chosen[best_m] = best_r
            uncovered &= ~mask_cache[(best_m, best_r)]
            remaining.remove(best_m)
        return chosen, score_from_uncovered(uncovered)

    def compute_wo(chosen):
        """Precompute wo[i] = uncovered without class i, for all i."""
        full = (1 << base) - 1
        covered = 0
        for m, r in chosen.items():
            covered |= mask_cache[(m, r)]
        uncovered = full & ~covered
        wo = []
        for i in range(n_div):
            mi = divisors[i]
            mi_mask = mask_cache[(mi, chosen[mi])]
            # uncovered without mi = full & ~(covered & ~mi_mask)
            # = full & ~(covered | mi_mask) ... no
            # covered_without_mi = covered & ~mi_mask
            # uncovered_without_mi = full & ~covered_without_mi = full & ~(covered & ~mi_mask)
            # = full & (~covered | mi_mask)
            # But ~covered in Python is infinite bits, so:
            # uncovered_without_mi = (full & ~covered) | (full & mi_mask & ~covered)
            # Actually simpler: uncovered_without_mi = (full & ~(covered & ~mi_mask))
            covered_wo = covered & ~mi_mask
            wo.append(full & ~covered_wo)
        return wo, uncovered

    def local_search_fast(chosen, deadline):
        """Fast local search using precomputed wo masks."""
        n = n_div
        full = (1 << base) - 1

        # Compute initial uncovered
        covered = 0
        for m, r in chosen.items():
            covered |= mask_cache[(m, r)]
        uncovered = full & ~covered
        best_score = uncovered.bit_count()

        if best_score == 0:
            return chosen, 0

        improved = True
        passes = 0
        while improved and passes < 80 and time.time() < deadline:
            improved = False
            passes += 1
            # Precompute wo for all i
            covered = 0
            for m, r in chosen.items():
                covered |= mask_cache[(m, r)]
            for i in range(n):
                if time.time() > deadline:
                    break
                mi = divisors[i]
                mi_mask = mask_cache[(mi, chosen[mi])]
                covered_wo = covered & ~mi_mask
                wo_i = full & ~covered_wo
                # Find best residue for mi given wo_i
                cur_r = chosen[mi]
                best_r = cur_r
                best_c = (wo_i & mask_cache[(mi, cur_r)]).bit_count()
                for nr in range(mi):
                    if nr == cur_r:
                        continue
                    c = (wo_i & mask_cache[(mi, nr)]).bit_count()
                    if c > best_c:
                        best_c = c
                        best_r = nr
                if best_r != cur_r:
                    chosen[mi] = best_r
                    improved = True
                    if best_score == 0:
                        break
            # Recompute score after pass
            if improved:
                covered = 0
                for m, r in chosen.items():
                    covered |= mask_cache[(m, r)]
                uncovered = full & ~covered
                new_score = uncovered.bit_count()
                if new_score < best_score:
                    best_score = new_score
                if best_score == 0:
                    break
        return chosen, best_score

    def two_move_search(chosen, deadline):
        """Try dropping class i and re-adding best available class."""
        n = n_div
        full = (1 << base) - 1
        covered = 0
        for m, r in chosen.items():
            covered |= mask_cache[(m, r)]
        uncovered = full & ~covered
        best_score = uncovered.bit_count()

        for _ in range(100):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                mi = divisors[i]
                mi_mask = mask_cache[(mi, chosen[mi])]
                covered_wo = covered & ~mi_mask
                wo_i = full & ~covered_wo
                # Find best class (m, r) to add given wo_i, m != mi
                best_gain = -1
                best_m = None
                best_r = None
                for m in divisors:
                    if m == mi:
                        continue
                    for r in range(m):
                        c = (wo_i & mask_cache[(m, r)]).bit_count()
                        if c > best_gain:
                            best_gain = c
                            best_m = m
                            best_r = r
                # Also consider keeping mi with different residue
                for r in range(mi):
                    c = (wo_i & mask_cache[(mi, r)]).bit_count()
                    if c > best_gain:
                        best_gain = c
                        best_m = mi
                        best_r = r
                if best_m is None:
                    continue
                if best_m != mi or best_r != chosen[mi]:
                    # Save old state
                    old_mi = mi
                    old_mr = chosen[mi]
                    # Apply
                    del chosen[old_mi]
                    chosen[best_m] = best_r
                    # Recompute
                    covered2 = 0
                    for m, r in chosen.items():
                        covered2 |= mask_cache[(m, r)]
                    uncovered2 = full & ~covered2
                    ns = uncovered2.bit_count()
                    if ns < best_score:
                        best_score = ns
                        improved = True
                        # Update covered for next iteration
                        covered = covered2
                        if best_score == 0:
                            return chosen, 0
                    else:
                        # Revert
                        del chosen[best_m]
                        chosen[old_mi] = old_mr
            if not improved:
                break
        return chosen, best_score

    # --- Main search ---
    best_classes = None
    best_score = base

    # Global greedy (dynamic selection)
    chosen, sc = global_greedy()
    if sc < best_score:
        best_score = sc
        best_classes = dict(chosen)

    # Greedy in increasing order
    chosen, sc = greedy(divisors)
    if sc < best_score:
        best_score = sc
        best_classes = dict(chosen)

    # Greedy in decreasing order
    chosen, sc = greedy(divisors[::-1])
    if sc < best_score:
        best_score = sc
        best_classes = dict(chosen)

    # Random orders with fixed-order greedy
    random.seed(12345)
    t_end_rand = deadline - 50
    tries = 0
    while time.time() < t_end_rand and tries < 300:
        order = divisors[:]
        random.shuffle(order)
        chosen, sc = greedy(order)
        if sc < best_score:
            best_score = sc
            best_classes = dict(chosen)
            if best_score == 0:
                break
        tries += 1

    # Local search polish on best
    if time.time() < deadline - 10 and best_classes is not None:
        chosen, sc = local_search_fast(best_classes, deadline - 10)
        if sc < best_score:
            best_score = sc
            best_classes = chosen

    # Two-move search
    if time.time() < deadline - 5 and best_score > 0 and best_classes is not None:
        chosen, sc = two_move_search(best_classes, deadline - 5)
        if sc < best_score:
            best_score = sc
            best_classes = chosen

    # Second round: more random + local if time
    if time.time() < deadline - 20 and best_score > 0:
        random.seed(99999)
        for _ in range(80):
            if time.time() > deadline - 20:
                break
            order = divisors[:]
            random.shuffle(order)
            chosen, sc = greedy(order)
            if sc <= best_score:
                chosen, sc2 = local_search_fast(chosen, deadline - 20)
                if sc2 < best_score:
                    best_score = sc2
                    best_classes = chosen
                    if best_score == 0:
                        break

    # Final local search
    if time.time() < deadline and best_score > 0 and best_classes is not None:
        chosen, sc = local_search_fast(best_classes, deadline)
        if sc < best_score:
            best_score = sc
            best_classes = chosen

    if best_classes is None:
        best_classes = {m: 0 for m in divisors}

    return [[m, best_classes[m]] for m in divisors]
# EVOLVE-BLOCK-END
