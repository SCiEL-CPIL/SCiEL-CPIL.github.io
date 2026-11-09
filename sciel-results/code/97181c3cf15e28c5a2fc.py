"""Rewritten program for the suite task; strong algorithm with deadline guard."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    t0 = time.time()
    DEADLINE = 85.0          # wall-clock safety margin (evaluator kills ~90s)
    MAXV = 512
    TARGET = 48
    rng = random.Random(42)

    universe = list(range(1, MAXV + 1))

    def is_valid(arr):
        """Check no element is the exact mean of any subset (>=2) of OTHER elements."""
        n = len(arr)
        if n < 2:
            return True
        for i in range(n):
            a = arr[i]
            bits = 1  # bitset: bit s set iff sum s reachable from other elements
            for j in range(n):
                if j != i:
                    bits |= (bits << arr[j])
            max_sum = sum(arr) - a
            # Check if k*a is reachable for any k >= 2
            for k in range(2, max_sum // a + 1):
                if (bits >> (k * a)) & 1:
                    return False
        return True

    def is_valid_fast(arr, new_elem):
        """Check if adding new_elem to arr keeps validity. Only check constraints involving new_elem."""
        n = len(arr)
        # Check 1: new_elem is not the mean of any subset of arr (size >= 2)
        a = new_elem
        bits = 1
        for v in arr:
            bits |= (bits << v)
        max_sum = sum(arr)
        for k in range(2, max_sum // a + 1):
            if (bits >> (k * a)) & 1:
                return False
        # Check 2: no existing element is the mean of a subset that includes new_elem
        for i in range(n):
            b = arr[i]
            # subsets of (arr without arr[i]) that may or may not include new_elem
            # We need: for any subset S of (arr \ {arr[i]}) with new_elem in S or not,
            # but S must have size >= 2 and not include arr[i]
            # Equivalent: check subsets of (arr \ {arr[i]}) that include new_elem
            # and subsets of (arr \ {arr[i]}) that don't include new_elem (already checked in is_valid(arr))
            # So we only need to check subsets that include new_elem
            others_without_i = [v for v in arr if v != b]
            # Build bitset for others_without_i
            bits2 = 1
            for v in others_without_i:
                bits2 |= (bits2 << v)
            # Now check: is there a subset of others_without_i with sum = k*b - new_elem for k>=2?
            # where the subset has size >= 1 (so total subset size >= 2 including new_elem)
            # Actually: subset T of others_without_i, total subset = T ∪ {new_elem}, size = |T|+1 >= 2 => |T| >= 1
            # mean condition: sum(T) + new_elem = (|T|+1) * b
            # => sum(T) = (|T|+1)*b - new_elem
            # Let k = |T|+1 >= 2, so sum(T) = k*b - new_elem, |T| = k-1 >= 1
            # We need to check if there exists T subset of others_without_i with sum(T) = k*b - new_elem and |T| = k-1
            # This is harder with just sum. But we can check: for each k >= 2,
            # does there exist a subset of others_without_i of size (k-1) with sum (k*b - new_elem)?
            # For efficiency, just check if (k*b - new_elem) is reachable as a subset sum of others_without_i
            # (the size constraint is implicit in the original problem but we're being conservative)
            max_sum2 = sum(others_without_i)
            for k in range(2, max_sum2 // b + 2):
                target = k * b - new_elem
                if target < 0:
                    continue
                if target > max_sum2:
                    break
                if (bits2 >> target) & 1:
                    return False
        return True

    def greedy(order, start=None):
        """Greedy add in given order; return valid set."""
        if start is None:
            chosen = []
        else:
            chosen = list(start)
        chosen_set = set(chosen)
        for v in order:
            if v in chosen_set:
                continue
            if is_valid_fast(chosen, v):
                chosen.append(v)
                chosen_set.add(v)
        return chosen

    # Strategy 1: Start with powers of 2 (known to be sparse)
    # Powers of 2 up to 512: 1,2,4,8,...,256 (9 elements)
    p2 = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best = greedy(universe, start=p2)
    best_len = len(best)

    # Strategy 2: Try different starting points
    for start_val in [1, 10, 50, 100, 200, 300, 400, 500]:
        if time.time() > DEADLINE:
            break
        start = [start_val]
        cand = greedy(universe, start=start)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            if best_len >= TARGET:
                break

    # Strategy 3: Randomized greedy from empty
    for _ in range(200):
        if time.time() > DEADLINE:
            break
        order = universe[:]
        rng.shuffle(order)
        cand = greedy(order)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            if best_len >= TARGET:
                break

    # Strategy 4: Local search - try to add more elements
    if len(best) < TARGET and time.time() < DEADLINE:
        best_set = set(best)
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            remaining = [v for v in universe if v not in best_set]
            rng.shuffle(remaining)
            for v in remaining:
                if is_valid_fast(best, v):
                    best.append(v)
                    best_set.add(v)
                    improved = True
                    break
            # Try swaps
            if not improved:
                for idx in range(len(best)):
                    if time.time() > DEADLINE:
                        break
                    trial = best[:idx] + best[idx+1:]
                    trial_set = set(trial)
                    for v in remaining:
                        if v in trial_set:
                            continue
                        if is_valid_fast(trial, v):
                            best = trial + [v]
                            best_set = set(best)
                            improved = True
                            break
                    if improved:
                        break

    # Final verification
    if not is_valid(best):
        # Fallback
        best = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    return best
# EVOLVE-BLOCK-END
