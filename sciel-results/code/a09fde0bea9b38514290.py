import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0

    best_val = float('inf')
    best_assign = None

    def get_candidates(L, W, d):
        start_k = (L + d - 1) // d
        end_k = (L + W) // d
        if end_k < start_k:
            return []
        return list(range(start_k * d, (end_k + 1) * d, d))

    def backtrack(idx, used, assign, L, W):
        nonlocal best_val, best_assign
        if idx == n:
            val = max(assign) - min(assign)
            if val < best_val:
                best_val = val
                best_assign = assign[:]
            return True
        d = divisors[idx]
        cands = get_candidates(L, W, d)
        for c in cands:
            if c not in used:
                used.add(c)
                assign.append(c)
                if backtrack(idx + 1, used, assign, L, W):
                    return True
                assign.pop()
                used.remove(c)
        return False

    # Search W from 16 upward (inspirations found 16)
    for W in range(16, 26):
        if time.time() > deadline:
            break
        found = False
        # Scan L; the pattern is periodic with LCM, but scan a reasonable range
        # For W=16, each divisor has 1-2 multiples in the window
        for L in range(1, min(MAX_V - W + 1, 200000)):
            if time.time() > deadline:
                break
            # Quick feasibility: each divisor must have at least one multiple in [L, L+W]
            feasible = True
            for d in divisors:
                start_k = (L + d - 1) // d
                end_k = (L + W) // d
                if end_k < start_k:
                    feasible = False
                    break
            if not feasible:
                continue
            used = set()
            assign = []
            if backtrack(0, used, assign, L, W):
                found = True
                break
        if found:
            break

    if best_assign is None:
        return divisors[:]

    return [int(x) for x in best_assign]
