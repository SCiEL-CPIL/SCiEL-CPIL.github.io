# EVOLVE-BLOCK-START
import time
import random

N = 256
# Precompute all nonconstant 4-APs (a, a+d, a+2d, a+3d) within [1, N]
FOUR_AP = []
for a in range(1, N - 2):
    for d in range(1, (N - a) // 3 + 1):
        FOUR_AP.append((a, a + d, a + 2 * d, a + 3 * d))
# For each element, list of 4-AP indices that contain it (for fast feasibility)
FOUR_AP_OF = [[] for _ in range(N + 1)]
for idx, (a, b, c, d) in enumerate(FOUR_AP):
    for x in (a, b, c, d):
        FOUR_AP_OF[x].append(idx)

def count_3aps(mask):
    """Count distinct 3-APs (a, a+d, a+2d) with d>0 contained in mask (bit i = element i)."""
    cnt = 0
    for a in range(1, N - 1):
        if not (mask >> a) & 1:
            continue
        # max d such that a+2d <= N
        max_d = (N - a) // 2
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            if (mask >> b) & 1 and (mask >> c) & 1:
                cnt += 1
    return cnt

def is_4ap_free(mask):
    for (a, b, c, d) in FOUR_AP:
        if (mask & ((1 << a) | (1 << b) | (1 << c) | (1 << d))) == ((1 << a) | (1 << b) | (1 << c) | (1 << d)):
            return False
    return True

def build_skeleton():
    """Deterministic digit-based 4-AP-free skeleton: keep x if its base-2 digits
    contain no '11' pattern AND base-3 digits contain no '11' pattern.
    This is a classic Cantor-like construction that avoids long APs."""
    mask = 0
    for x in range(1, N + 1):
        # base 2 no adjacent 1s
        ok2 = True
        v = x
        prev = 0
        while v:
            bit = v & 1
            if bit and prev:
                ok2 = False
                break
            prev = bit
            v >>= 1
        if not ok2:
            continue
        # base 3 no adjacent 1s
        ok3 = True
        v = x
        prev = 0
        while v:
            digit = v % 3
            if digit == 1 and prev == 1:
                ok3 = False
                break
            prev = digit
            v //= 3
        if ok3:
            mask |= (1 << x)
    return mask

def density_complete(mask, deadline):
    """Greedy add elements that don't create a 4-AP, prioritizing those that
    create the most new 3-APs. Uses fast conflict check via FOUR_AP_OF."""
    # current set as boolean list for speed
    in_set = [(mask >> i) & 1 for i in range(N + 1)]
    # precompute for each candidate x: which 4-APs would be completed if x added
    # (i.e., 4-APs where x is the missing element and other 3 are present)
    def add_gain(x):
        """Number of new 3-APs created by adding x (assuming no 4-AP created)."""
        gain = 0
        # x as first term: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if in_set[b] and in_set[c]:
                gain += 1
        # x as second term: (x-d, x, x+d)
        for d in range(1, x):
            a = x - d
            c = x + d
            if in_set[a] and in_set[c]:
                gain += 1
        # x as third term: (x-2d, x-d, x)
        for d in range(1, (x // 2) + 1):
            a = x - 2 * d
            b = x - d
            if in_set[a] and in_set[b]:
                gain += 1
        return gain

    def creates_4ap(x):
        """Check if adding x creates any 4-AP."""
        for ap_idx in FOUR_AP_OF[x]:
            a, b, c, d = FOUR_AP[ap_idx]
            others = [y for y in (a, b, c, d) if y != x]
            if all(in_set[y] for y in others):
                return True
        return False

    # Candidates not in set
    candidates = [x for x in range(1, N + 1) if not in_set[x]]
    # Sort by potential gain (static estimate), then add greedily
    # Re-evaluate dynamically: at each step pick best feasible candidate
    while time.time() < deadline:
        best_x = None
        best_gain = 0
        for x in candidates:
            if creates_4ap(x):
                continue
            g = add_gain(x)
            if g > best_gain:
                best_gain = g
                best_x = x
        if best_x is None:
            break
        in_set[best_x] = 1
        mask |= (1 << best_x)
        candidates.remove(best_x)
        if best_gain == 0:
            # No more gain; stop
            break
    return mask

def local_search(mask, deadline):
    """Fast local search: try single-element removals and additions,
    accepting if 3-AP count increases and 4-AP-free is maintained."""
    in_set = [(mask >> i) & 1 for i in range(N + 1)]
    current_score = count_3aps(mask)

    def creates_4ap(x):
        for ap_idx in FOUR_AP_OF[x]:
            a, b, c, d = FOUR_AP[ap_idx]
            others = [y for y in (a, b, c, d) if y != x]
            if all(in_set[y] for y in others):
                return True
        return False

    def add_gain(x):
        gain = 0
        for d in range(1, (N - x) // 2 + 1):
            if in_set[x + d] and in_set[x + 2 * d]:
                gain += 1
        for d in range(1, x):
            if in_set[x - d] and in_set[x + d]:
                gain += 1
        for d in range(1, (x // 2) + 1):
            if in_set[x - 2 * d] and in_set[x - d]:
                gain += 1
        return gain

    def remove_loss(x):
        """Number of 3-APs destroyed by removing x."""
        loss = 0
        for d in range(1, (N - x) // 2 + 1):
            if in_set[x + d] and in_set[x + 2 * d]:
                loss += 1
        for d in range(1, x):
            if in_set[x - d] and in_set[x + d]:
                loss += 1
        for d in range(1, (x // 2) + 1):
            if in_set[x - 2 * d] and in_set[x - d]:
                loss += 1
        return loss

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try additions
        for x in range(1, N + 1):
            if in_set[x]:
                continue
            if creates_4ap(x):
                continue
            g = add_gain(x)
            if g > 0:
                in_set[x] = 1
                mask |= (1 << x)
                current_score += g
                improved = True
        # Try removals that allow better re-addition (skip for simplicity; 
        # single-pass additions are usually sufficient after good skeleton)
        if improved:
            continue
        # If no additions, try one swap: remove x, then see if we can add y with net gain
        for x in range(1, N + 1):
            if not in_set[x]:
                continue
            if time.time() >= deadline:
                break
            loss = remove_loss(x)
            if loss == 0:
                continue  # removing gives no benefit to try
            in_set[x] = 0
            mask &= ~(1 << x)
            best_y = None
            best_net = 0
            for y in range(1, N + 1):
                if in_set[y]:
                    continue
                if creates_4ap(y):
                    continue
                g = add_gain(y)
                net = g - loss
                if net > best_net:
                    best_net = net
                    best_y = y
            if best_y is not None and best_net > 0:
                in_set[best_y] = 1
                mask |= (1 << best_y)
                current_score += best_net
                improved = True
            else:
                # revert
                in_set[x] = 1
                mask |= (1 << x)
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin

    # Step 1: Deterministic skeleton
    mask = build_skeleton()

    # Step 2: Density-complete
    mask = density_complete(mask, deadline)

    # Step 3: Local search refinement
    mask = local_search(mask, deadline)

    # Ensure nonempty
    if mask == 0:
        mask = 1 << 1

    return [i for i in range(1, N + 1) if (mask >> i) & 1]
# EVOLVE-BLOCK-END
