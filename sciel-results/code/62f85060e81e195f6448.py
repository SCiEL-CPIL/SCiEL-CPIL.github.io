import time
import itertools

def construct():
    deadline = time.time() + 115.0
    target_mask = (1 << 30) - 1  # bits 0..29 for diffs 1..30

    # Precompute mask for each difference d in 1..30
    diff_mask = [0] * 31
    for d in range(1, 31):
        diff_mask[d] = 1 << (d - 1)

    def is_sidon_valid(S, x, used_diffs):
        """Check if x can be added to S without violating Sidon property."""
        for s in S:
            d = x - s
            if d in used_diffs:
                return False
        return True

    def get_new_coverage(S, x, covered_mask):
        """Get the new differences covered by adding x."""
        new_mask = 0
        for s in S:
            d = x - s
            if 1 <= d <= 30:
                new_mask |= diff_mask[d]
        return new_mask & ~covered_mask

    def search(D, time_limit):
        """Find a Sidon set in [0, D] containing 0 with all diffs 1..30."""
        start_time = time.time()
        best_result = [None]

        def dfs(S, used_diffs, covered_mask):
            if time.time() - start_time > time_limit:
                return

            if (covered_mask & target_mask) == target_mask:
                if best_result[0] is None:
                    best_result[0] = list(S)
                return

            cur_max = S[-1] if S else 0
            if cur_max >= D:
                return

            # Pruning: check if we can possibly cover all remaining diffs
            missing = target_mask & ~covered_mask
            num_missing = bin(missing).count('1')
            n = len(S)
            # Max new elements we can add: at most D - cur_max, but also limited by 32 - n
            max_add = min(D - cur_max, 32 - n)
            # Each new element x with existing S creates |S| new diffs
            # Plus diffs between new elements
            potential = max_add * n + max_add * (max_add - 1) // 2
            if potential < num_missing:
                return

            # Generate and sort candidates
            cands = []
            for x in range(cur_max + 1, D + 1):
                if not is_sidon_valid(S, x, used_diffs):
                    continue
                new_cov = get_new_coverage(S, x, covered_mask)
                cnt = bin(new_cov).count('1')
                cands.append((cnt, x))

            # Sort by coverage descending, then x ascending
            cands.sort(key=lambda t: (-t[0], t[1]))

            # Limit candidates to avoid explosion
            if len(cands) > 30:
                cands = cands[:30]

            for _, x in cands:
                new_used = set(used_diffs)
                new_covered = covered_mask
                for s in S:
                    d = x - s
                    new_used.add(d)
                    if 1 <= d <= 30:
                        new_covered |= diff_mask[d]

                S.append(x)
                dfs(S, new_used, new_covered)
                S.pop()

                if best_result[0] is not None:
                    return
                if time.time() - start_time > time_limit:
                    return

        dfs([0], set(), 0)
        return best_result[0]

    # Try D from lower bound upward
    # Lower bound: need at least enough elements to cover 30 diffs
    # With k elements, max diffs = k*(k-1)/2. Need >= 30, so k >= 9
    # But also D must be at least 30 (since diff 30 must exist)
    for D in range(30, 120):
        if time.time() > deadline:
            break
        time_left = deadline - time.time()
        # Allocate time based on remaining budget
        time_per_D = min(5.0, time_left * 0.7)
        if time_per_D < 1.0:
            break
        res = search(D, time_per_D)
        if res is not None:
            return res

    # Fallback: greedy construction
    S = [0]
    used = set()
    target = set(range(1, 31))
    while target and time.time() < deadline - 5:
        best_x = None
        best_cover = -1
        for x in range(max(S) + 1, max(S) + 500):
            ok = True
            cover = 0
            for s in S:
                d = x - s
                if d in used:
                    ok = False
                    break
                if d in target:
                    cover += 1
            if ok and cover > best_cover:
                best_cover = cover
                best_x = x
        if best_x is None:
            break
        S.append(best_x)
        for s in S[:-1]:
            used.add(best_x - s)
        target -= used

    return S
