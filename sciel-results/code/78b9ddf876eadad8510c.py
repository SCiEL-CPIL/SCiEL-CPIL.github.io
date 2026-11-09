"""B2[2] set in 1..1000: maximize size with every sum having <= 2 representations a+b, a<=b."""
import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill

    def elapsed():
        return time.time() - t0

    def sum_counts(chosen):
        """Return dict of sum -> count of representations a+b, a<=b, a,b in chosen."""
        counts = {}
        n = len(chosen)
        for i in range(n):
            a = chosen[i]
            for j in range(i, n):
                b = chosen[j]
                s = a + b
                counts[s] = counts.get(s, 0) + 1
        return counts

    def insertion_sums(x, chosen):
        """New sums added by inserting x: x+y for y in chosen, plus 2x."""
        return [x + y for y in chosen] + [2 * x]

    def is_valid_insertion(x, chosen, counts):
        """Check if inserting x keeps every sum <= 2."""
        for s in insertion_sums(x, chosen):
            if counts.get(s, 0) + 1 > 2:
                return False
        return True

    def apply_insertion(x, chosen, counts):
        """Insert x, update counts. Assumes valid."""
        chosen.append(x)
        for s in insertion_sums(x, chosen[:-1]):
            counts[s] = counts.get(s, 0) + 1
        return chosen, counts

    def remove_element(x, chosen, counts):
        """Remove x, update counts."""
        chosen.remove(x)
        # subtract sums x+y for y in chosen (now without x), plus 2x
        for y in chosen:
            s = x + y
            counts[s] -= 1
            if counts[s] == 0:
                del counts[s]
        s = 2 * x
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
        return chosen, counts

    # ---- Phase 1: probabilistic greedy ----
    random.seed(42)
    chosen = []
    counts = {}
    for x in range(1, 1001):
        if elapsed() > DEADLINE - 60:
            break
        new_sums = insertion_sums(x, chosen)
        # slack: number of new sums that would have count < 2 after insertion
        slack = 0
        valid = True
        for s in new_sums:
            c = counts.get(s, 0) + 1
            if c > 2:
                valid = False
                break
            if c < 2:
                slack += 1
        if not valid:
            continue
        # accept probability proportional to slack (more slack = better balance)
        # base acceptance with a temperature-like factor
        p = min(1.0, slack / 3.0 + 0.5)
        if random.random() < p:
            apply_insertion(x, chosen, counts)

    # ---- Phase 2: local search — add-only + swaps (fast) ----
    def best_add(chosen, counts):
        """Find an unchosen x that can be added; return (x, slack) for best, or None."""
        best_x = None
        best_slack = -1
        chosen_set_local = chosen_set  # avoid closure lookup
        for x in range(1, 1001):
            if x in chosen_set_local:
                continue
            valid = True
            slack = 0
            for y in chosen:
                s = x + y
                c = counts.get(s, 0) + 1
                if c > 2:
                    valid = False
                    break
                if c < 2:
                    slack += 1
            if not valid:
                continue
            s2 = 2 * x
            c2 = counts.get(s2, 0) + 1
            if c2 > 2:
                continue
            if c2 < 2:
                slack += 1
            if slack > best_slack:
                best_slack = slack
                best_x = x
                if slack == 0:
                    break
        if best_x is None:
            return None
        return best_x, best_slack

    chosen_set = set(chosen)

    def try_swap_remove_add(chosen, counts, chosen_set, a):
        """Try to find x not in chosen such that removing a and adding x is valid.
        Returns (x, slack) for the best such x, or None."""
        best_x = None
        best_slack = -1
        for x in range(1, 1001):
            if x in chosen_set:
                continue
            valid = True
            slack = 0
            for y in chosen:
                if y == a:
                    continue
                s = x + y
                # count of s after removing a: subtract a's contribution to s
                # a contributes to s if (s - a) is in chosen and s - a != a, or 2a == s
                a_contrib = 0
                if s - a in chosen_set and s - a != a:
                    a_contrib = 1
                elif 2 * a == s:
                    a_contrib = 1
                c = counts.get(s, 0) - a_contrib + 1
                if c > 2:
                    valid = False
                    break
                if c < 2:
                    slack += 1
            if not valid:
                continue
            s2 = 2 * x
            a_contrib2 = 0
            if s2 - a in chosen_set and s2 - a != a:
                a_contrib2 = 1
            elif 2 * a == s2:
                a_contrib2 = 1
            c2 = counts.get(s2, 0) - a_contrib2 + 1
            if c2 > 2:
                continue
            if c2 < 2:
                slack += 1
            if slack > best_slack:
                best_slack = slack
                best_x = x
                if slack == 0:
                    break
        if best_x is None:
            return None
        return best_x, best_slack

    improved = True
    while improved and elapsed() < DEADLINE - 20:
        improved = False
        res = best_add(chosen, counts)
        if res is not None:
            x, _ = res
            apply_insertion(x, chosen, counts)
            chosen_set.add(x)
            improved = True
            continue
        # No add possible; try swaps
        if chosen and elapsed() < DEADLINE - 20:
            # Try a few random removals, look for a valid add
            for _ in range(20):
                if elapsed() > DEADLINE - 20:
                    break
                a = random.choice(chosen)
                swap_res = try_swap_remove_add(chosen, counts, chosen_set, a)
                if swap_res is not None:
                    x, _ = swap_res
                    chosen, counts = remove_element(a, chosen, counts)
                    chosen_set.remove(a)
                    apply_insertion(x, chosen, counts)
                    chosen_set.add(x)
                    improved = True
                    break

    # ---- Phase 3: final polish — exhaustive add pass ----
    while elapsed() < DEADLINE - 5:
        res = best_add(chosen, counts)
        if res is None:
            break
        x, _ = res
        apply_insertion(x, chosen, counts)
        chosen_set.add(x)

    # ---- Phase 4: simulated-annealing-style perturb to escape local minima ----
    if elapsed() < DEADLINE - 15 and chosen:
        T = 5.0  # initial "temperature"
        best_size = len(chosen)
        best_chosen = list(chosen)
        best_counts = dict(counts)
        best_chosen_set = set(chosen)
        no_improve = 0
        while elapsed() < DEADLINE - 5 and T > 0.1:
            # Try a perturbation: remove a random element, add a random valid one
            a = random.choice(chosen)
            swap_res = try_swap_remove_add(chosen, counts, chosen_set, a)
            if swap_res is not None:
                x, _ = swap_res
                chosen, counts = remove_element(a, chosen, counts)
                chosen_set.remove(a)
                apply_insertion(x, chosen, counts)
                chosen_set.add(x)
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = list(chosen)
                    best_counts = dict(counts)
                    best_chosen_set = set(chosen)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            # Try an add
            res = best_add(chosen, counts)
            if res is not None:
                x, _ = res
                apply_insertion(x, chosen, counts)
                chosen_set.add(x)
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = list(chosen)
                    best_counts = dict(counts)
                    best_chosen_set = set(chosen)
                    no_improve = 0
                else:
                    no_improve += 1
            # Cool down
            T *= 0.97
            if no_improve > 30:
                # Reset to best and cool faster
                chosen = best_chosen
                counts = best_counts
                chosen_set = best_chosen_set
                no_improve = 0
                T *= 0.85
        # Restore best
        if len(best_chosen) > len(chosen):
            chosen = best_chosen
            counts = best_counts
            chosen_set = best_chosen_set

    # ---- Phase 5: final add pass after SA ----
    while elapsed() < DEADLINE - 2:
        res = best_add(chosen, counts)
        if res is None:
            break
        x, _ = res
        apply_insertion(x, chosen, counts)
        chosen_set.add(x)

    return chosen
