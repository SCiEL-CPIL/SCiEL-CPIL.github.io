import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030  # lcm of all divisors

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s wall-clock kill

    best_result = None
    best_span = float('inf')

    # Try window sizes from minimum (9) upward
    for W in range(9, 250):
        if time.time() > deadline:
            break

        # For each starting position L in one LCM period
        # The window is [L, L+W]
        # We need to assign each divisor d to a distinct multiple in [L, L+W]
        # A multiple of d in [L, L+W] has the form L + r where r is in [0, W] and (L+r) % d == 0
        # i.e., r ≡ (-L) mod d

        for L in range(LCM):
            if time.time() > deadline:
                break

            # Compute candidate offsets for each divisor
            candidates = []
            valid = True
            for d in divisors:
                r0 = (-L) % d
                if r0 > W:
                    valid = False
                    break
                cands = []
                r = r0
                while r <= W:
                    cands.append(r)
                    r += d
                if not cands:
                    valid = False
                    break
                candidates.append(cands)

            if not valid:
                continue

            # MRV ordering: most constrained (fewest candidates) first
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))

            # If any divisor has 0 candidates, skip (already handled)
            # Quick Hall's condition check on the most constrained few
            # For the k most constrained variables, union of candidates must have size >= k
            for k in range(1, min(n, 4) + 1):
                union = set()
                for idx in range(k):
                    i = indices[idx]
                    union.update(candidates[i])
                if len(union) < k:
                    valid = False
                    break

            if not valid:
                continue

            # Backtracking search
            used = [False] * (W + 1)
            used_count = 0
            choices = [-1] * n

            def backtrack(idx):
                nonlocal used_count
                if idx == n:
                    return True

                # Pruning: remaining variables need (n - idx) distinct slots
                remaining_slots = (W + 1) - used_count
                if remaining_slots < (n - idx):
                    return False

                i = indices[idx]
                for r in candidates[i]:
                    if not used[r]:
                        used[r] = True
                        used_count += 1
                        choices[i] = r
                        if backtrack(idx + 1):
                            return True
                        used[r] = False
                        used_count -= 1
                        choices[i] = -1
                return False

            if backtrack(0):
                result = []
                for i in range(n):
                    result.append(L + choices[i])
                # Verify: distinct, positive, <= 1000000, divisible
                if any(x <= 0 or x > 1000000 for x in result):
                    continue
                if len(set(result)) != n:
                    continue
                for i, x in enumerate(result):
                    if x % divisors[i] != 0:
                        break
                else:
                    span = max(result) - min(result)
                    if span < best_span:
                        best_span = span
                        best_result = [int(x) for x in result]
                    # Found a valid solution with this W, no need to continue
                    # since we're iterating W in increasing order
                    return [int(x) for x in result]

    # Fallback
    if best_result is not None:
        return best_result
    return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
