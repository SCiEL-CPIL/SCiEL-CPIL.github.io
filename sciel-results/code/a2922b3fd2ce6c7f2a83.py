"""Rewritten program for the suite task; strong algorithm with deadline guard."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    t0 = time.time()
    DEADLINE = 75.0          # wall-clock safety margin (evaluator kills ~90s)
    MAXV = 512
    TARGET = 48
    rng = random.Random(1234567)

    # Precompute all candidate values 1..512
    universe = list(range(1, MAXV + 1))

    def is_valid(arr):
        """Check no element is the exact mean of any subset (>=2) of OTHER elements.
        Uses bitset subset-sum DP. Returns True if valid."""
        n = len(arr)
        if n < 2:
            return True
        maxv = max(arr)
        # For each element a, build DP over the others and check k*a reachability
        for i in range(n):
            a = arr[i]
            # reachable sums using the other elements
            # bitset as python int; bit s set iff sum s reachable
            bits = 1  # sum 0
            for j in range(n):
                if j == i:
                    continue
                v = arr[j]
                bits |= (bits << v)
            # check if any k*a (k>=2, k*a <= max_sum) is reachable
            max_sum = sum(arr) - a
            k = 2
            while k * a <= max_sum:
                if (bits >> (k * a)) & 1:
                    return False
                k += 1
        return True

    def greedy(order):
        """Greedy add in given order; return valid set."""
        chosen = []
        for v in order:
            cand = chosen + [v]
            if is_valid(cand):
                chosen.append(v)
        return chosen

    # Start with a decent baseline: ascending greedy
    base = greedy(universe)
    best = base
    best_len = len(base)

    # Try multiple randomized orders to find a larger valid set
    attempt = 0
    while time.time() < DEADLINE:
        attempt += 1
        # Build a randomized order biased toward values that "fit"
        # Use a shuffled copy
        order = universe[:]
        rng.shuffle(order)
        cand = greedy(order)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            if best_len >= TARGET:
                break
        # Also try greedy starting from best then shuffling remaining
        if len(best) < TARGET:
            remaining = [v for v in universe if v not in best]
            rng.shuffle(remaining)
            cand2 = list(best)
            for v in remaining:
                c = cand2 + [v]
                if is_valid(c):
                    cand2.append(v)
            if len(cand2) > best_len:
                best = cand2
                best_len = len(cand2)
                if best_len >= TARGET:
                    break

    # Local search: try to swap out a problematic element for a new one
    if len(best) < TARGET and time.time() < DEADLINE:
        best = list(best)
        best_set = set(best)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            # find an element whose removal allows adding a new element
            # try each position to drop, then greedily add the best new element
            for idx in range(len(best)):
                trial = [x for i, x in enumerate(best) if i != idx]
                trial_set = set(trial)
                # find a new element (not in trial) that makes it valid and larger
                for v in universe:
                    if v in trial_set:
                        continue
                    cand = trial + [v]
                    if is_valid(cand):
                        # ensure we actually gain: len same (swap) but maybe better for future
                        # only accept if it leads to same size but we can continue; accept swap
                        best = cand
                        best_set = set(best)
                        improved = True
                        break
                if improved:
                    break
            # try pure addition if possible
            for v in universe:
                if v in best_set:
                    continue
                cand = best + [v]
                if is_valid(cand):
                    best = cand
                    best_set = set(best)
                    improved = True
                    break

    # Final safety: ensure we return a valid set (re-verify)
    if not is_valid(best):
        # fallback to a known-valid small set
        best = list(range(1, 13))
    return best
# EVOLVE-BLOCK-END
