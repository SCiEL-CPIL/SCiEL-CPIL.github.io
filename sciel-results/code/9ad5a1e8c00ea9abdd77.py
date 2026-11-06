import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin under 90s kill
    n = 512
    target = 48
    max_sum = target * n  # 24576

    def add_to_dp(dp, x, m):
        # dp has length m+1 (0..m); returns new dp of length m+2 (0..m+1) for set ∪ {x}
        new_dp = dp + [0]
        for k in range(m + 1, 0, -1):
            new_dp[k] |= (new_dp[k - 1] << x)
        return new_dp

    def is_valid_candidate(C, dp, dp_without, x):
        m = len(C)
        # Check A: x is mean of a k-subset of C, k>=2
        for k in range(2, m + 1):
            target_sum = k * x
            if target_sum > m * n:
                continue
            if (dp[k] >> target_sum) & 1:
                return False
        
        # Check B: existing y becomes mean of a subset of C U {x} containing x
        for y in C:
            # dp_without[y] represents C \ {y}, size m-1, list length m (0..m-1)
            dpy = dp_without[y]
            for j in range(1, m):  # j = size of U subset of C\{y}
                target_sum = (j + 1) * y - x
                if target_sum < 0:
                    continue
                if target_sum > (m - 1) * n:
                    continue
                if (dpy[j] >> target_sum) & 1:
                    return False
        return True

    def full_validate(S):
        # Returns True if S is valid
        m = len(S)
        if m == 0:
            return True
        # Build dp for S
        dp = [0] * (m + 1)
        dp[0] = 1
        for val in S:
            for k in range(m, 0, -1):
                dp[k] |= (dp[k - 1] << val)
        # For each x in S, check no subset of S\{x} of size k>=2 sums to k*x
        for x in S:
            # Build dp for S \ {x}
            dp2 = [0] * (m)  # S\{x} has m-1 elements, indices 0..m-1
            dp2[0] = 1
            for val in S:
                if val == x:
                    continue
                for k in range(m - 1, 0, -1):
                    dp2[k] |= (dp2[k - 1] << val)
            for k in range(2, m):
                target_sum = k * x
                if target_sum > (m - 1) * n:
                    continue
                if (dp2[k] >> target_sum) & 1:
                    return False
        return True

    def trim_to_valid(S):
        # Remove elements until valid
        S = list(S)
        while S and not full_validate(S):
            # remove the "most problematic" element: the one that is a mean
            removed = False
            for x in S:
                # check if x is a mean of a subset of S\{x}
                m = len(S)
                dp2 = [0] * (m)
                dp2[0] = 1
                for val in S:
                    if val == x:
                        continue
                    for k in range(m - 1, 0, -1):
                        dp2[k] |= (dp2[k - 1] << val)
                bad = False
                for k in range(2, m):
                    target_sum = k * x
                    if target_sum > (m - 1) * n:
                        continue
                    if (dp2[k] >> target_sum) & 1:
                        bad = True
                        break
                if bad:
                    S.remove(x)
                    removed = True
                    break
            if not removed:
                # no single element is a mean; should not happen if full_validate failed
                # remove last element as fallback
                S.pop()
        return S

    best = []
    rng = random.Random(12345)
    max_trials = 20000

    for trial in range(max_trials):
        if time.time() > deadline - 3.0:
            break
        if len(best) >= target:
            break

        order = list(range(1, n + 1))
        rng.shuffle(order)

        C = []
        dp = [1]  # dp[0] = 1
        dp_without = {}

        for x in order:
            if len(C) >= target:
                break
            if is_valid_candidate(C, dp, dp_without, x):
                m = len(C)
                # add x
                # dp_without[x] = dp of C (before adding x)
                dp_without[x] = list(dp)
                # update dp_without[y] for y in C to include x
                for y in C:
                    dpy = dp_without[y] + [0]
                    for k in range(m, 0, -1):
                        dpy[k] |= (dpy[k - 1] << x)
                    dp_without[y] = dpy
                # update dp
                dp = add_to_dp(dp, x, m)
                C.append(x)

        if len(C) > len(best):
            best = list(C)
            if len(best) >= target:
                break

    # Final safety: validate and trim
    if best and not full_validate(best):
        best = trim_to_valid(best)

    return best
