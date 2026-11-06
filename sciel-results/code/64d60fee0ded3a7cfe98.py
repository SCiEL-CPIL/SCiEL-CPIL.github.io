import time
import random

N = 256
TOTAL_DEADLINE = 56.5


def _validate(sel):
    # Check distinct unordered sums (a<=b) and membership in [1,N].
    if not sel:
        return False
    seen = set()
    for i in range(len(sel)):
        a = sel[i]
        if a < 1 or a > N:
            return False
        if (a + a) in seen:
            return False
        seen.add(a + a)
        for j in range(i + 1, len(sel)):
            b = sel[j]
            s = a + b
            if s in seen:
                return False
            seen.add(s)
    return True


def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_DEADLINE
    rng = random.Random(20240517)

    def get_mask(sel):
        m = 0
        for i in range(len(sel)):
            a = sel[i]
            m |= 1 << (2 * a)
            for j in range(i + 1, len(sel)):
                m |= 1 << (a + sel[j])
        return m

    def can_add(x, mask, sel):
        if (mask >> (2 * x)) & 1:
            return False
        for a in sel:
            if (mask >> (x + a)) & 1:
                return False
        return True

    def is_maximal(sel, sel_set, mask):
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            if can_add(y, mask, sel):
                return False
        return True

    def deficit(sel, sel_set, mask):
        # Count of missing elements that are addable (0 => maximal).
        d = 0
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            if can_add(y, mask, sel):
                d += 1
        return d

    def saturate(sel, sel_set, mask, time_budget=0.6):
        # Greedily add any addable element until none remain (=> maximal).
        t_end = time.time() + time_budget
        changed = True
        while changed and time.time() < t_end:
            changed = False
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                if can_add(x, mask, sel):
                    for a in sel:
                        mask |= 1 << (x + a)
                    mask |= 1 << (2 * x)
                    sel.append(x)
                    sel_set.add(x)
                    changed = True
        return sel, sel_set, mask

    def greedy_build(seed):
        r = random.Random(seed)
        sel = []
        sel_set = set()
        mask = 0
        remaining = list(range(1, N + 1))
        r.shuffle(remaining)
        while remaining and time.time() < deadline:
            best_x = -1
            best_score = -1.0
            if len(remaining) < 30:
                cands = remaining
            else:
                cands = r.sample(remaining, 30)
            for x in cands:
                if x in sel_set:
                    continue
                if not can_add(x, mask, sel):
                    continue
                block = 0
                for y in remaining:
                    if y == x:
                        continue
                    if (mask >> (y + x)) & 1:
                        block += 1
                if block > best_score:
                    best_score = block
                    best_x = x
            if best_x == -1:
                for x in remaining:
                    if x not in sel_set and can_add(x, mask, sel):
                        best_x = x
                        break
            if best_x == -1:
                break
            for a in sel:
                mask |= 1 << (best_x + a)
            mask |= 1 << (2 * best_x)
            sel.append(best_x)
            sel_set.add(best_x)
            remaining.remove(best_x)
        sel, sel_set, mask = saturate(sel, sel_set, mask)
        return sel, sel_set, mask

    def try_replace(sel, sel_set, mask, rem_set, rem_mask, y):
        # Try adding y to rem (mask after removing two elements); return new
        # saturated list if it is valid+maximal and strictly smaller than sel.
        if y in sel_set or y in rem_set:
            return None
        if not can_add(y, rem_mask, rem_set):
            return None
        new_mask = rem_mask
        for a in rem_set:
            new_mask |= 1 << (y + a)
        new_mask |= 1 << (2 * y)
        new_sel = list(rem_set)
        new_sel.append(y)
        new_sel_set = set(new_sel)
        new_sel, new_sel_set, new_mask = saturate(new_sel, new_sel_set, new_mask)
        if len(new_sel) < len(sel):
            return new_sel
        return None

    def local_search_reduce(sel, time_budget=4.0):
        t_start = time.time()
        t_end = t_start + time_budget
        cur = list(sel)
        cur_set = set(cur)
        cur_mask = get_mask(cur)
        while time.time() < t_end:
            improved = False
            elems = list(cur)
            # 2-exchange: drop two, replace with one (size -1)
            for i in range(len(elems)):
                if time.time() > t_end:
                    break
                x1 = elems[i]
                for j in range(i + 1, len(elems)):
                    if time.time() > t_end:
                        break
                    x2 = elems[j]
                    rem = [e for e in cur if e != x1 and e != x2]
                    rem_set = set(rem)
                    rem_mask = get_mask(rem)
                    found = None
                    for y in range(1, N + 1):
                        if found:
                            break
                        found = try_replace(cur, cur_set, cur_mask, rem_set, rem_mask, y)
                    if found is not None:
                        cur = found
                        cur_set = set(cur)
                        cur_mask = get_mask(cur)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # 1-exchange: drop one, replace with one (size 0), must stay maximal
            found = None
            for i in range(len(cur)):
                if found:
                    break
                x = cur[i]
                rem = [e for e in cur if e != x]
                rem_set = set(rem)
                rem_mask = get_mask(rem)
                for y in range(1, N + 1):
                    if y in cur_set or y in rem_set:
                        continue
                    if not can_add(y, rem_mask, rem_set):
                        continue
                    new_mask = rem_mask
                    for a in rem_set:
                        new_mask |= 1 << (y + a)
                    new_mask |= 1 << (2 * y)
                    new_sel = list(rem_set)
                    new_sel.append(y)
                    new_sel_set = set(new_sel)
                    new_sel, new_sel_set, new_mask = saturate(
                        new_sel, new_sel_set, new_mask, time_budget=0.3)
                    if len(new_sel) == len(cur) and is_maximal(new_sel, new_sel_set, new_mask):
                        found = new_sel
                        break
                if found:
                    break
            if found is not None:
                cur = found
                cur_set = set(cur)
                cur_mask = get_mask(cur)
                improved = True
            if not improved:
                break
        return cur

    # Phase 1: multiple greedy seeds
    best_sel = None
    best_size = 10 ** 9
    num_trials = 40
    for i in range(num_trials):
        if time.time() > deadline - 12:
            break
        sel, sel_set, mask = greedy_build(i)
        if len(sel) < best_size:
            best_size = len(sel)
            best_sel = list(sel)
    if best_sel is None:
        best_sel, _, _ = greedy_build(0)

    # Phase 2: intensive reduction
    current = local_search_reduce(list(best_sel), time_budget=min(20.0, max(2.0, deadline - time.time() - 2.0)))

    # Phase 3: repeated short reductions to escape local minima
    while time.time() < deadline - 1.5:
        new_current = local_search_reduce(list(current), time_budget=2.0)
        if len(new_current) < len(current):
            current = new_current
        else:
            # random perturbation then reduce
            if len(current) > 2 and time.time() < deadline - 1.0:
                perturbed = list(current)
                k = min(len(perturbed), rng.randint(1, 3))
                idxs = rng.sample(range(len(perturbed)), k)
                for ix in sorted(idxs, reverse=True):
                    perturbed.pop(ix)
                pset = set(perturbed)
                pmask = get_mask(perturbed)
                perturbed, pset, pmask = saturate(perturbed, pset, pmask, time_budget=0.4)
                reduced = local_search_reduce(list(perturbed), time_budget=1.5)
                if len(reduced) < len(current):
                    current = reduced

    if not _validate(current):
        # Fallback to a guaranteed-valid maximal set
        sel, sel_set, mask = greedy_build(0)
        current = list(sel)

    return current


if __name__ == "__main__":
    s = construct()
    print(len(s), _validate(s))
