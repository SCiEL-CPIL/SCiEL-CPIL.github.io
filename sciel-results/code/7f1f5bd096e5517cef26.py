import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030

    start_time = time.time()
    deadline = start_time + 80  # Leave margin for 90s wall-clock kill

    # Start from minimum possible span
    for W in range(9, 200):
        if time.time() > deadline:
            break

        for L in range(LCM):
            if time.time() > deadline:
                break

            # Quick filter
            valid = True
            for d in divisors:
                if (-L) % d > W:
                    valid = False
                    break
            if not valid:
                continue

            # Generate candidates
            candidates = []
            for d in divisors:
                r0 = (-L) % d
                cands = []
                r = r0
                while r <= W:
                    cands.append(r)
                    r += d
                candidates.append(cands)

            # MRV ordering
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))

            used = [False] * (W + 1)
            choices = [-1] * n

            # Precompute for each position in the MRV-ordered list,
            # which original divisor index it corresponds to
            def backtrack(idx):
                if idx == n:
                    return True

                # Check remaining capacity: need n-idx more positions,
                # but only (W+1) - used_count total slots
                # Actually, we need to ensure enough unused slots remain
                # Count unused slots from idx onward
                remaining_slots = (W + 1) - sum(used)
                if remaining_slots < (n - idx):
                    return False

                i = indices[idx]
                # Try candidates in order of least constraining first
                # (those that leave more options for subsequent variables)
                # Simple heuristic: try smaller offsets first
                for r in candidates[i]:
                    if not used[r]:
                        used[r] = True
                        choices[i] = r
                        if backtrack(idx + 1):
                            return True
                        used[r] = False
                        choices[i] = -1
                return False

            if backtrack(0):
                result = []
                for i in range(n):
                    result.append(L + choices[i])
                return [int(x) for x in result]

    # Fallback
    return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
