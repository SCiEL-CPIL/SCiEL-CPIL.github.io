import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max possible d: 1 + 2*255 = 511 <= 512

    def time_ok():
        return time.time() < deadline

    # Bitmask approach: bit i represents element (i+1) in the set
    # So element value v corresponds to bit (v-1)

    def score_mask(mask):
        """Count distinct d > 0 such that a, a+d, a+2d all in mask."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    # Fast delta: when replacing old_val with new_val, only check d's that involve old_val or new_val
    def delta_replace(mask, old_val, new_val):
        """Return (delta, new_mask) when replacing old_val with new_val."""
        new_mask = (mask & ~(1 << (old_val - 1))) | (1 << (new_val - 1))

        # Find d's that could be affected
        affected_ds = set()
        # d's involving old_val
        for d in range(1, MAX_D + 1):
            # old_val as first: old_val, old_val+d, old_val+2d
            if old_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            # old_val as middle: old_val-d, old_val, old_val+d
            if old_val - d >= 1 and old_val + d <= MAX_VAL:
                affected_ds.add(d)
            # old_val as last: old_val-2d, old_val-d, old_val
            if old_val - 2 * d >= 1:
                affected_ds.add(d)

        # d's involving new_val
        for d in range(1, MAX_D + 1):
            if new_val + 2 * d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - d >= 1 and new_val + d <= MAX_VAL:
                affected_ds.add(d)
            if new_val - 2 * d >= 1:
                affected_ds.add(d)

        delta = 0
        for d in affected_ds:
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1

        return delta, new_mask

    # Greedy construction
    def greedy_construct():
        mask = 0
        s = 0
        s_set = set()

        for _ in range(N):
            best_new = None
            best_s = s

            # Try all possible new elements
            candidates = []
            for v in range(1, MAX_VAL + 1):
                if v in s_set:
                    continue
                candidates.append(v)

            # Sample a subset for speed if too many
            if len(candidates) > 300:
                random.shuffle(candidates)
                candidates = candidates[:300]

            for v in candidates:
                new_mask = mask | (1 << (v - 1))
                new_s = score_mask(new_mask)
                if new_s > best_s:
                    best_s = new_s
                    best_new = v

            if best_new is None:
                best_new = random.choice(candidates) if candidates else 1
                new_mask = mask | (1 << (best_new - 1))
                best_s = score_mask(new_mask)

            mask = new_mask
            s = best_s
            s_set.add(best_new)

        return mask, s, mask_to_list(mask)

    # Local search with targeted moves
    def local_search(mask, s, s_set, max_rounds=100):
        lst = mask_to_list(mask)

        for round_idx in range(max_rounds):
            if not time_ok():
                break

            improved = False
            for _ in range(len(lst)):
                i = random.randint(0, N - 1)
                old_val = lst[i]

                best_new = old_val
                best_delta = 0
                best_new_mask = mask

                # Strategy 1: Random replacements
                for _ in range(10):
                    new_val = random.randint(1, MAX_VAL)
                    if new_val == old_val or new_val in s_set:
                        continue

                    delta, new_mask = delta_replace(mask, old_val, new_val)
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new_val
                        best_new_mask = new_mask

                # Strategy 2: Targeted moves - seek values completing 3-APs
                # Try to add a middle term: for pairs (a, c) in S, try (a+c)//2
                for a in lst:
                    for c in lst:
                        if a == c:
                            continue
                        mid = (a + c) // 2
                        if a + c % 2 != 0:
                            continue
                        if 1 <= mid <= MAX_VAL and mid != old_val and mid not in s_set:
                            delta, new_mask = delta_replace(mask, old_val, mid)
                            if delta > best_delta:
                                best_delta = delta
                                best_new = mid
                                best_new_mask = new_mask

                # Try to add endpoints: for pairs (a, b) in S, try a-b or a+b
                for a in lst:
                    for b in lst:
                        if a == b:
                            continue
                        # Try a + (a-b) = 2a - b
                        end1 = 2 * a - b
                        if 1 <= end1 <= MAX_VAL and end1 != old_val and end1 not in s_set:
                            delta, new_mask = delta_replace(mask, old_val, end1)
                            if delta > best_delta:
                                best_delta = delta
                                best_new = end1
                                best_new_mask = new_mask
                        # Try b + (b-a) = 2b - a
                        end2 = 2 * b - a
                        if 1 <= end2 <= MAX_VAL and end2 != old_val and end2 not in s_set:
                            delta, new_mask = delta_replace(mask, old_val, end2)
                            if delta > best_delta:
                                best_delta = delta
                                best_new = end2
                                best_new_mask = new_mask

                if best_new != old_val:
                    s_set.discard(old_val)
                    s_set.add(best_new)
                    lst[i] = best_new
                    mask = best_new_mask
                    s += best_delta
                    improved = True

            if not improved:
                break

        return mask, s, s_set, lst

    # Final polish: try to add any remaining beneficial single moves
    def final_polish(mask, s, s_set):
        lst = mask_to_list(mask)
        improved = True
        while improved and time_ok():
            improved = False
            for i in range(len(lst)):
                old_val = lst[i]
                best_new = old_val
                best_delta = 0
                best_new_mask = mask

                # Try all possible replacements
                for new_val in range(1, MAX_VAL + 1):
                    if new_val == old_val or new_val in s_set:
                        continue
                    delta, new_mask = delta_replace(mask, old_val, new_val)
                    if delta > best_delta:
                        best_delta = delta
                        best_new = new_val
                        best_new_mask = new_mask

                if best_new != old_val:
                    s_set.discard(old_val)
                    s_set.add(best_new)
                    lst[i] = best_new
                    mask = best_new_mask
                    s += best_delta
                    improved = True

        return mask, s, s_set, lst

    # Main: try multiple strategies
    random.seed(42)

    best_mask = 0
    best_s = 0
    best_list = []

    # Strategy 1: Greedy
    if time_ok():
        mask, s, lst = greedy_construct()
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]

    # Strategy 2: Interval start + local search
    if time_ok():
        mask = 0
        for i in range(1, 33):
            mask |= (1 << (i - 1))
        s = score_mask(mask)
        s_set = set(range(1, 33))

        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = mask_to_list(mask)

        mask, s, s_set, lst = local_search(mask, s, s_set, max_rounds=50)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]

    # Strategy 3: AP start + local search
    for step in [8, 12, 16, 20]:
        if not time_ok():
            break
        mask = 0
        lst = [1 + i * step for i in range(N)]
        if max(lst) > MAX_VAL:
            continue
        for x in lst:
            mask |= (1 << (x - 1))
        s = score_mask(mask)
        s_set = set(lst)

        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]

        mask, s, s_set, lst = local_search(mask, s, s_set, max_rounds=30)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]

    # Strategy 4: Random start + local search
    for _ in range(5):
        if not time_ok():
            break
        lst = random.sample(range(1, MAX_VAL + 1), N)
        mask = list_to_mask(lst)
        s = score_mask(mask)
        s_set = set(lst)

        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]

        mask, s, s_set, lst = local_search(mask, s, s_set, max_rounds=30)
        if s > best_s:
            best_mask = mask
            best_s = s
            best_list = lst[:]

    # Final polish on the best solution
    if time_ok() and best_list:
        best_s_set = set(best_list)
        best_mask, best_s, best_s_set, best_list = final_polish(best_mask, best_s, best_s_set)

    return sorted(best_list)
