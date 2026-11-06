"""Constraint-propagation + fast local-search solver for Sidon set extension
containing {1, 2, 5, 11} in [1, 284]."""
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before the 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    N = 284
    MAXS = 568  # 2 * 284

    # ---- fast sum maintenance ------------------------------------------------
    # sum_count[s] = number of ordered pairs (a,b) in the current set with a+b==s.
    # A set is Sidon iff every sum value is used at most once.
    sum_count = [0] * (MAXS + 1)

    def add_elem(x):
        for y in current_set:
            s = x + y
            if s <= MAXS and sum_count[s] == 0:
                sum_count[s] = 1
            else:
                return False
        if sum_count[2 * x] == 0:
            sum_count[2 * x] = 1
        else:
            return False
        current_set.add(x)
        return True

    def remove_elem(x):
        current_set.discard(x)
        for y in current_set:
            s = x + y
            if s <= MAXS:
                sum_count[s] -= 1
        sum_count[2 * x] -= 1

    def count_conflicts(x):
        """Number of existing set elements y with x+y already used (plus 2x)."""
        c = 0
        for y in current_set:
            if sum_count[x + y] > 0:
                c += 1
        if sum_count[2 * x] > 0:
            c += 1
        return c

    # ---- global best ----------------------------------------------------------
    best_set = set()
    best_size = 0

    def update_best():
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

    def init_state():
        nonlocal current_set
        for s in range(MAXS + 1):
            sum_count[s] = 0
        current_set = set()
        for x in prescribed:
            add_elem(x)  # always valid

    current_set = set()
    init_state()
    update_best()

    rng = random.Random(12345)

    # ---- Strategy 1: randomized greedy restarts ------------------------------
    remaining = [x for x in range(1, 285) if x not in prescribed_set]

    def greedy_random_order(order):
        init_state()
        for x in order:
            if x in current_set:
                continue
            if add_elem(x):
                update_best()

    # deterministic extremes
    greedy_random_order(sorted(remaining))
    greedy_random_order(sorted(remaining, reverse=True))

    # randomized restarts
    t0 = time.time()
    trial = 0
    while time.time() - t0 < 25.0 and trial < 4000:
        order = remaining[:]
        rng.shuffle(order)
        greedy_random_order(order)
        trial += 1

    # ---- Strategy 2: local search (1-opt + 2-opt) ----------------------------
    def local_search(start_set, time_limit):
        init_state()
        for x in start_set:
            add_elem(x)
        update_best()
        start = time.time()
        all_cands = list(range(1, 285))
        removable = [x for x in current_set if x not in prescribed_set]

        for _ in range(2000000):
            if time.time() - start > time_limit or time.time() > deadline - 5:
                break

            # 1-opt add
            for _a in range(8):
                x = rng.choice(all_cands)
                if x not in current_set and add_elem(x):
                    update_best()
                    break

            # 1-opt swap
            if removable:
                xr = rng.choice(removable)
                remove_elem(xr)
                removable = [x for x in current_set if x not in prescribed_set]
                for _a in range(6):
                    xa = rng.choice(all_cands)
                    if xa not in current_set and add_elem(xa):
                        break
                remove_elem(xr)  # restore

            # 2-opt swap (remove two, add two)
            if len(removable) >= 2:
                pair = rng.sample(removable, 2)
                x1, x2 = pair
                remove_elem(x1)
                remove_elem(x2)
                removable = [x for x in current_set if x not in prescribed_set]
                for _a in range(6):
                    xa = rng.choice(all_cands)
                    if xa not in current_set and add_elem(xa):
                        break
                for _a in range(6):
                    xb = rng.choice(all_cands)
                    if xb not in current_set and add_elem(xb):
                        break
                remove_elem(x1)
                remove_elem(x2)  # restore

    local_search(best_set, 35.0)

    # ---- Strategy 3: backtracking CSP with forward checking -------------------
    def backtrack(time_limit):
        init_state()
        start = time.time()
        stack = []
        remaining_cands = [x for x in range(1, 285) if x not in prescribed_set]

        def rec():
            nonlocal remaining_cands
            if time.time() - start > time_limit or time.time() > deadline - 5:
                return
            update_best()

            # forward check: is there any addable candidate?
            valid = [x for x in remaining_cands
                     if x not in current_set and sum_count[2 * x] == 0]
            # (full per-candidate check is done at add time; quick filter here)
            if not valid:
                return
            # heuristic: prefer candidates with fewest conflicts
            valid.sort(key=lambda x: count_conflicts(x))
            # randomize a small window to diversify
            window = valid[:12]
            rng.shuffle(window)

            for x in window:
                if add_elem(x):
                    remaining_cands.append(x)
                    stack.append(x)
                    rec()
                    stack.pop()
                    remaining_cands.pop()
                    remove_elem(x)
                    if time.time() - start > time_limit or time.time() > deadline - 5:
                        return

        rec()

    backtrack(40.0)

    # ---- Strategy 4: final polish from the best set --------------------------
    local_search(best_set, 25.0)

    # greedy fill from best
    init_state()
    for x in best_set:
        add_elem(x)
    for x in range(1, 285):
        if x not in current_set and add_elem(x):
            update_best()

    # ---- return & verify ------------------------------------------------------
    result = sorted(best_set)
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for y in result[i:]:
            s = x + y
            if s in sums:
                valid = False
                break
            sums.add(s)
        if not valid:
            break
    if not valid or not prescribed_set.issubset(set(result)):
        return prescribed
    return result


if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
    sums = set()
    valid = True
    for i, x in enumerate(result):
        for y in result[i:]:
            s = x + y
            if s in sums:
                valid = False
                break
            sums.add(s)
    print(f"Valid: {valid}")
