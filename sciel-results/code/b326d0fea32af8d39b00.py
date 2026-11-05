"""Local-search optimizer: pairwise-coprime set maximizing sum 1/(512-a)."""
import time
import random
from math import gcd

N = 512
DEADLINE = 140.0  # safety margin under the 150s wall-clock kill


def _score(nums):
    s = 0.0
    for a in nums:
        s += 1.0 / (N - a)
    return s


def _initial_greedy():
    # Descending greedy: pick the largest still-coprime numbers.
    chosen = []
    for a in range(511, 0, -1):
        if all(gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
    return chosen


def _swap_gain(removed, added, cur_score):
    delta = 0.0
    for a in removed:
        delta -= 1.0 / (N - a)
    for a in added:
        delta += 1.0 / (N - a)
    return cur_score + delta


def _try_swap(nums, removed, added, cur_score):
    # removed/added are tuples of indices/values; verify pairwise coprime.
    rem = list(removed)
    add = list(added)
    if not rem or not add:
        return False
    # check added are distinct and not already present
    present = set(nums)
    vals_add = []
    for a in add:
        if a in present:
            return False
        vals_add.append(a)
    vals_rem = [nums[i] for i in rem]
    if len(set(vals_add)) != len(vals_add):
        return False
    # build candidate set
    cand = [x for i, x in enumerate(nums) if i not in set(rem)]
    cand += vals_add
    # verify pairwise coprime
    for i in range(len(cand)):
        ai = cand[i]
        for j in range(i + 1, len(cand)):
            if gcd(ai, cand[j]) != 1:
                return False
    new_score = cur_score
    for a in vals_rem:
        new_score -= 1.0 / (N - a)
    for a in vals_add:
        new_score += 1.0 / (N - a)
    if new_score > cur_score + 1e-12:
        nums[:] = cand
        return True
    return False


def _best_added(cand, present, nums):
    # find best a not in present, coprime to all nums, maximizing 1/(512-a)
    best_a = None
    best_v = 0.0
    for a in cand:
        if a in present:
            continue
        ok = True
        for b in nums:
            if gcd(a, b) != 1:
                ok = False
                break
        if ok:
            v = 1.0 / (N - a)
            if v > best_v:
                best_v = v
                best_a = a
    return best_a


def construct():
    start = time.time()
    random.seed(12345)

    nums = _initial_greedy()
    cur_score = _score(nums)
    best_nums = list(nums)
    best_score = cur_score

    # candidate "added" values: large numbers give large 1/(512-a)
    big_vals = list(range(511, 0, -1))

    # --- Phase 1: single adds (grow set) ---
    improved = True
    while improved and (time.time() - start) < DEADLINE:
        improved = False
        present = set(nums)
        best_a = _best_added(big_vals, present, nums)
        if best_a is not None:
            nums.append(best_a)
            cur_score += 1.0 / (N - best_a)
            improved = True
            if cur_score > best_score:
                best_score = cur_score
                best_nums = list(nums)

    # --- Phase 2: 2-for-1 swap moves (merge small primes into large composites) ---
    # repeatedly: pick a small member to remove, try to add up to 2 large values.
    improved = True
    while improved and (time.time() - start) < DEADLINE:
        improved = False
        # try removing each member (smallest contribution first)
        idx_order = sorted(range(len(nums)), key=lambda i: 1.0 / (N - nums[i]))
        for ri in idx_order:
            if (time.time() - start) > DEADLINE:
                break
            rem_val = nums[ri]
            # candidate new set without rem_val
            base = [nums[i] for i in range(len(nums)) if i != ri]
            present = set(base)
            base_score = cur_score - 1.0 / (N - rem_val)
            # try adding 1
            best_a = _best_added(big_vals, present, base)
            if best_a is not None:
                cand = base + [best_a]
                ok = True
                for i in range(len(cand)):
                    for j in range(i + 1, len(cand)):
                        if gcd(cand[i], cand[j]) != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    ns = base_score + 1.0 / (N - best_a)
                    if ns > cur_score + 1e-12:
                        nums = cand
                        cur_score = ns
                        if cur_score > best_score:
                            best_score = cur_score
                            best_nums = list(nums)
                        improved = True
                        break
            # try adding 2
            # greedily find two large coprime values
            c1 = _best_added(big_vals, present, base)
            if c1 is not None:
                base2 = base + [c1]
                present2 = present | {c1}
                c2 = _best_added(big_vals, present2, base2)
                if c2 is not None:
                    cand = base2 + [c2]
                    ok = True
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            if gcd(cand[i], cand[j]) != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        ns = base_score + 1.0 / (N - c1) + 1.0 / (N - c2)
                        if ns > cur_score + 1e-12:
                            nums = cand
                            cur_score = ns
                            if cur_score > best_score:
                                best_score = cur_score
                                best_nums = list(nums)
                            improved = True
                            break

    # --- Phase 3: randomized restarts with swaps ---
    while (time.time() - start) < DEADLINE:
        random.shuffle(nums)
        s = _score(nums)
        if s > best_score:
            best_score = s
            best_nums = list(nums)
        # local improvement on this arrangement
        improved = True
        while improved and (time.time() - start) < DEADLINE:
            improved = False
            idx_order = sorted(range(len(nums)), key=lambda i: 1.0 / (N - nums[i]))
            for ri in idx_order:
                if (time.time() - start) > DEADLINE:
                    break
                rem_val = nums[ri]
                base = [nums[i] for i in range(len(nums)) if i != ri]
                present = set(base)
                base_score = s - 1.0 / (N - rem_val)
                best_a = _best_added(big_vals, present, base)
                if best_a is not None:
                    cand = base + [best_a]
                    ok = True
                    for i in range(len(cand)):
                        for j in range(i + 1, len(cand)):
                            if gcd(cand[i], cand[j]) != 1:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        ns = base_score + 1.0 / (N - best_a)
                        if ns > s + 1e-12:
                            nums = cand
                            s = ns
                            if s > best_score:
                                best_score = s
                                best_nums = list(nums)
                            improved = True
                            break

    # final safety: ensure validity
    valid = []
    for a in best_nums:
        if all(gcd(a, b) == 1 for b in valid):
            valid.append(a)
    return valid
