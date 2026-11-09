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

    # ---- Phase 2: local search — add-only + swaps ----
    def best_add(chosen, counts):
        """Find an unchosen x that can be added; return (x, slack) for best, or None."""
        best_x = None
        best_slack = -1
        chosen_set_local = chosen_set  # avoid closure lookup
        for x in range(1, 1001):
            if x in chosen_set_local:
                continue
            new_sums = insertion_sums(x, chosen)
            slack = 0
            valid = True
            for s in new_sums:
                c = counts.get(s, 0) + 1
                if c > 2:
                    valid = False
                    break
            if not valid:
                continue
            for s in new_sums:
                if counts.get(s, 0) + 1 < 2:
                    slack += 1
            if slack > best_slack:
                best_slack = slack
                best_x = x
                # Early exit: if all new sums reach count 2, this is optimal
                if slack == 0:
                    break
        if best_x is None:
            return None
        return best_x, best_slack

    chosen_set = set(chosen)
    improved = True
    while improved and elapsed() < DEADLINE - 30:
        improved = False
        res = best_add(chosen, counts)
        if res is not None:
            x, _ = res
            apply_insertion(x, chosen, counts)
            chosen_set.add(x)
            improved = True
            continue
        # No add possible; try swaps to enable future adds
        # Try removing a random element and see if an add becomes possible
        if chosen:
            # try fewer random removals but with faster check
            for _ in range(10):
                if elapsed() > DEADLINE - 30:
                    break
                a = random.choice(chosen)
                # Quick check: remove a, then see if any x can be added
                # First, compute what sums would be freed by removing a
                freed_sums = [a + y for y in chosen if y != a] + [2*a]
                # For each freed sum, check if it currently has count 2 (blocking)
                # If so, removing a might unblock some x
                # Try to find an x that was blocked only by a
                found_swap = False
                for x in range(1, 1001):
                    if x in chosen_set:
                        continue
                    # Check if x can be added after removing a
                    valid = True
                    slack = 0
                    for y in chosen:
                        if y == a:
                            continue
                        s = x + y
                        c = counts.get(s, 0) - 1 + 1  # subtract a's contribution, add x's
                        # More carefully: counts currently includes a's sums
                        # After removing a, sum s=x+y has count = counts[s] - (1 if a+y==s else 0)
                        # This is getting complex; just do the full check
                        pass
                    # Full check for this x with a removed
                    new_sums_x = [x + y for y in chosen if y != a] + [2*x]
                    valid = True
                    slack = 0
                    for s in new_sums_x:
                        # count of s after removing a and adding x
                        # original count of s, minus 1 if a contributes to s, plus 1 for x's contribution
                        a_contrib = 1 if (s - a in chosen_set and s - a != a) or (2*a == s) else 0
                        # Actually simpler: a contributes to sum s if there exists y in chosen with a+y=s, i.e., y=s-a in chosen
                        # And 2a=s
                        if s - a in chosen_set and s - a != a:
                            a_contrib = 1
                        elif 2*a == s:
                            a_contrib = 1
                        else:
                            a_contrib = 0
                        c_after_remove = counts.get(s, 0) - a_contrib
                        c_final = c_after_remove + 1
                        if c_final > 2:
                            valid = False
                            break
                        if c_final < 2:
                            slack += 1
                    if valid and slack >= 0:
                        # Do the swap
                        chosen2, counts2 = remove_element(a, list(chosen), dict(counts))
                        chosen3, counts3 = apply_insertion(x, list(chosen2), dict(counts2))
                        chosen, counts = chosen3, counts3
                        chosen_set = set(chosen)
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break

    # ---- Phase 3: final polish — exhaustive add pass ----
    while elapsed() < DEADLINE - 5:
        res = best_add(chosen, counts)
        if res is None:
            break
        x, _ = res
        apply_insertion(x, chosen, counts)
        chosen_set.add(x)

    # ---- Phase 4: one more round of swaps if time permits ----
    if elapsed() < DEADLINE - 10 and chosen:
        for _ in range(5):
            if elapsed() > DEADLINE - 10:
                break
            res = best_add(chosen, counts)
            if res is not None:
                x, _ = res
                apply_insertion(x, chosen, counts)
                chosen_set.add(x)
                continue
            # Try a few quick swaps
            for _ in range(5):
                if elapsed() > DEADLINE - 10:
                    break
                a = random.choice(chosen)
                found_swap = False
                for x in range(1, 1001):
                    if x in chosen_set:
                        continue
                    new_sums_x = [x + y for y in chosen if y != a] + [2*x]
                    valid = True
                    for s in new_sums_x:
                        if s - a in chosen_set and s - a != a:
                            a_contrib = 1
                        elif 2*a == s:
                            a_contrib = 1
                        else:
                            a_contrib = 0
                        c_final = counts.get(s, 0) - a_contrib + 1
                        if c_final > 2:
                            valid = False
                            break
                    if valid:
                        chosen2, counts2 = remove_element(a, list(chosen), dict(counts))
                        chosen3, counts3 = apply_insertion(x, list(chosen2), dict(counts2))
                        chosen, counts = chosen3, counts3
                        chosen_set = set(chosen)
                        found_swap = True
                        break
                if found_swap:
                    break

    return chosen
