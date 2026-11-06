"""Rewritten program with stronger search for pair-sum divisibility avoidance."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0  # safety margin before 90s kill

    N = 256
    odds = [x for x in range(1, N + 1) if x % 2 == 1]
    evens = [x for x in range(1, N + 1) if x % 2 == 0]
    n_odds = len(odds)
    n_evens = len(evens)

    # Precompute conflicts
    # e_conf_odds[e_idx] = bitmask of odds conflicting with even e
    e_conf_odds = []
    # e_conf_evens[e_idx] = bitmask of evens (with higher idx) conflicting
    e_conf_evens = []

    for i, e in enumerate(evens):
        oc = 0
        for j, o in enumerate(odds):
            if (e * o) % (e + o) == 0:
                oc |= (1 << j)
        e_conf_odds.append(oc)

        ec = 0
        for k in range(i + 1, n_evens):
            e2 = evens[k]
            if (e * e2) % (e + e2) == 0:
                ec |= (1 << k)
        e_conf_evens.append(ec)

    # Build symmetric even-even conflict: for each even, which evens conflict
    # Using bitmasks over even indices
    even_conf = [0] * n_evens  # even_conf[i] has bit j set if even i conflicts with even j
    for i in range(n_evens):
        even_conf[i] = e_conf_evens[i]
        # Also need conflicts with lower-indexed evens
        for j in range(i):
            e1, e2 = evens[i], evens[j]
            if (e1 * e2) % (e1 + e2) == 0:
                even_conf[i] |= (1 << j)

    def time_ok():
        return time.time() < DEADLINE

    def greedy_select(order):
        """Greedy: process evens in order, add if compatible with current set."""
        selected_odds = (1 << n_odds) - 1  # all odds selected
        selected_evens = 0
        for e_idx in order:
            if selected_evens & (1 << e_idx):
                continue
            # Check conflict with odds
            if e_conf_odds[e_idx] & selected_odds:
                continue
            # Check conflict with selected evens
            if even_conf[e_idx] & selected_evens:
                continue
            selected_evens |= (1 << e_idx)
        return selected_odds, selected_evens

    def set_size(odds_mask, evens_mask):
        return bin(odds_mask).count('1') + bin(evens_mask).count('1')

    # Try multiple greedy orderings
    best_size = 0
    best_odds_mask = (1 << n_odds) - 1
    best_evens_mask = 0

    orders = [
        sorted(range(n_evens), key=lambda i: (bin(e_conf_odds[i]).count('1'), bin(even_conf[i]).count('1'), i)),
        sorted(range(n_evens), key=lambda i: (bin(even_conf[i]).count('1'), bin(e_conf_odds[i]).count('1'), i)),
        sorted(range(n_evens), key=lambda i: evens[i]),
        sorted(range(n_evens), key=lambda i: -evens[i]),
        sorted(range(n_evens), key=lambda i: evens[i].bit_count(), reverse=True),
        sorted(range(n_evens), key=lambda i: evens[i].bit_count()),
        sorted(range(n_evens), key=lambda i: bin(e_conf_odds[i]).count('1'), reverse=True),
        sorted(range(n_evens), key=lambda i: bin(e_conf_odds[i]).count('1')),
    ]

    for order in orders:
        if not time_ok():
            break
        om, em = greedy_select(order)
        sz = set_size(om, em)
        if sz > best_size:
            best_size = sz
            best_odds_mask = om
            best_evens_mask = em

    # Randomized greedy restarts
    rng = random.Random(12345)
    for _ in range(300):
        if not time_ok():
            break
        # Create a biased random order
        order = list(range(n_evens))
        # Partially shuffle with bias toward low-conflict elements
        for i in range(len(order) - 1, 0, -1):
            if rng.random() < 0.4:
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        om, em = greedy_select(order)
        sz = set_size(om, em)
        if sz > best_size:
            best_size = sz
            best_odds_mask = om
            best_evens_mask = em

    # Local search: try to improve by swapping evens
    def local_search(odds_mask, evens_mask):
        cur_odds = odds_mask
        cur_evens = evens_mask
        cur_size = set_size(cur_odds, cur_evens)
        improved = True
        while improved and time_ok():
            improved = False
            # Try adding each unselected even
            for e_idx in range(n_evens):
                if cur_evens & (1 << e_idx):
                    continue
                if e_conf_odds[e_idx] & cur_odds:
                    continue
                if even_conf[e_idx] & cur_evens:
                    continue
                cur_evens |= (1 << e_idx)
                cur_size += 1
                improved = True
                break

            # Try 1-for-1 swap: remove one even, add another
            if not improved and time_ok():
                for rem_idx in range(n_evens):
                    if not (cur_evens & (1 << rem_idx)):
                        continue
                    trial_evens = cur_evens & ~(1 << rem_idx)
                    # Try adding any compatible even
                    for add_idx in range(n_evens):
                        if add_idx == rem_idx:
                            continue
                        if trial_evens & (1 << add_idx):
                            continue
                        if e_conf_odds[add_idx] & cur_odds:
                            continue
                        if even_conf[add_idx] & trial_evens:
                            continue
                        cur_evens = trial_evens | (1 << add_idx)
                        improved = True
                        break
                    if improved:
                        break

            # Try 1-for-2 swap: remove one even, add two
            if not improved and time_ok():
                for rem_idx in range(n_evens):
                    if not (cur_evens & (1 << rem_idx)):
                        continue
                    trial_evens = cur_evens & ~(1 << rem_idx)
                    added = 0
                    for add_idx in range(n_evens):
                        if trial_evens & (1 << add_idx):
                            continue
                        if e_conf_odds[add_idx] & cur_odds:
                            continue
                        if even_conf[add_idx] & trial_evens:
                            continue
                        trial_evens |= (1 << add_idx)
                        added += 1
                        if added >= 2:
                            break
                    if added >= 2:
                        cur_evens = trial_evens
                        cur_size += added - 1
                        improved = True
                        break

            # Try 2-for-3 swap
            if not improved and time_ok():
                evens_list = [i for i in range(n_evens) if cur_evens & (1 << i)]
                if len(evens_list) >= 2:
                    for a in range(len(evens_list)):
                        for b in range(a + 1, len(evens_list)):
                            ra, rb = evens_list[a], evens_list[b]
                            trial_evens = cur_evens & ~(1 << ra) & ~(1 << rb)
                            added = 0
                            for add_idx in range(n_evens):
                                if trial_evens & (1 << add_idx):
                                    continue
                                if e_conf_odds[add_idx] & cur_odds:
                                    continue
                                if even_conf[add_idx] & trial_evens:
                                    continue
                                trial_evens |= (1 << add_idx)
                                added += 1
                                if added >= 3:
                                    break
                            if added >= 3:
                                cur_evens = trial_evens
                                cur_size += added - 2
                                improved = True
                                break
                        if improved:
                            break

        return cur_odds, cur_evens, cur_size

    best_odds_mask, best_evens_mask, best_size = local_search(best_odds_mask, best_evens_mask)

    # More randomized restarts with local search
    for _ in range(100):
        if not time_ok():
            break
        order = list(range(n_evens))
        for i in range(len(order) - 1, 0, -1):
            if rng.random() < 0.5:
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        om, em = greedy_select(order)
        sz = set_size(om, em)
        if sz >= best_size - 2:  # only local search if close to best
            om2, em2, sz2 = local_search(om, em)
            if sz2 > best_size:
                best_size = sz2
                best_odds_mask = om2
                best_evens_mask = em2

    # Final local search
    if time_ok():
        best_odds_mask, best_evens_mask, best_size = local_search(best_odds_mask, best_evens_mask)

    # Build result
    result = []
    for j, o in enumerate(odds):
        if best_odds_mask & (1 << j):
            result.append(o)
    for i, e in enumerate(evens):
        if best_evens_mask & (1 << i):
            result.append(e)
    result.sort()

    # Verify
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                # Invalid, fall back to all odds
                return odds
    return result
# EVOLVE-BLOCK-END
