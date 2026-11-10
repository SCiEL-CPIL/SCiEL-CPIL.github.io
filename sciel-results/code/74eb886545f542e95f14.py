import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    LCM = 30030  # 2*3*5*7*11*13

    start_time = time.time()
    deadline = start_time + 82  # Safety margin for 90s wall-clock kill

    # Minimum possible span is 9 (10 distinct integers need at least 10 slots)
    for W in range(9, 200):
        if time.time() > deadline:
            break

        # Quick feasibility: window [L, L+W] has W+1 slots, need n distinct.
        # W+1 >= n is guaranteed since W >= 9 >= 10? No, W+1 >= 10 requires W>=9. OK.
        # Also need each divisor to have at least one multiple in the window.

        for L in range(LCM):
            if time.time() > deadline:
                break

            # For each divisor d, the multiples in [L, L+W] are at relative
            # offsets r where (L + r) % d == 0, i.e. r ≡ (-L) mod d, 0 <= r <= W
            candidates = []
            valid = True
            total_slots_needed = 0
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
                total_slots_needed += len(cands)

            if not valid:
                continue

            # MRV ordering: most constrained (fewest candidates) first
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))

            # Additional pruning: check Hall's condition on the most constrained
            # variables. For the k most constrained, the union of their candidate
            # sets must have size >= k.
            # (Cheap check: just verify the first few)
            # Skip for speed; backtracking will handle it.

            used = [False] * (W + 1)
            used_count = 0
            choices = [-1] * n

            # Iterative backtracking with explicit stack to avoid recursion depth issues
            # and to allow tighter time checking
            # Actually, recursion depth is only 10, so recursion is fine.
            # Use recursion but with a boolean array for O(1) membership.

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
                # L is in [0, LCM-1], choices[i] in [0, W], so result[i] in [0, LCM-1+W]
                # Minimum L=0, choices can be 0 -> result=0, not positive!
                # Need result > 0. If L=0 and any choice is 0, that's invalid.
                # But wait: if L=0, then r=0 means value=0 which is not positive.
                # So we need to ensure all values are positive.
                # Actually, if L=0 and r=0, value=0. We need value >= 1.
                # Let's check: if any result[i] <= 0, skip.
                if any(x <= 0 for x in result):
                    continue
                # All good
                return [int(x) for x in result]

    # Fallback: return the divisors themselves (distinct, positive, <=1000000, each divisible by itself)
    return [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
