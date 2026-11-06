import random
import time

def _check_ap(colors, a, d, n):
    """Check if 4-term AP starting at a with diff d violates constraint.
    Positions are 1-indexed in the problem, but colors is 0-indexed.
    a, a+d, a+2d, a+3d must be in 1..n."""
    if a + 3 * d > n:
        return True  # out of range, no violation
    c0 = colors[a - 1]
    c1 = colors[a + d - 1]
    c2 = colors[a + 2 * d - 1]
    c3 = colors[a + 3 * d - 1]
    return len({c0, c1, c2, c3}) >= 3

def _is_valid(colors, n=128):
    """Check if all 4-term APs in 1..n use at least 3 colors."""
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            if not _check_ap(colors, a, d, n):
                return False
    return True

def _count_violations(colors, n=128):
    """Count number of violating 4-term APs."""
    count = 0
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            if not _check_ap(colors, a, d, n):
                count += 1
    return count

def _greedy_construct(n, K, rng, deadline):
    """Greedy: assign colors one by one, trying to minimize violations."""
    colors = [-1] * n
    # Try multiple random orderings
    for attempt in range(200):
        if time.time() > deadline:
            return None
        colors = [-1] * n
        order = list(range(n))
        rng.shuffle(order)
        for idx in order:
            pos = idx + 1  # 1-indexed
            best_color = -1
            best_violations = float('inf')
            for c in range(K):
                colors[idx] = c
                # Count violations involving this position
                v = 0
                # As first term: pos, pos+d, pos+2d, pos+3d
                for d in range(1, (n - pos) // 3 + 1):
                    if not _check_ap(colors, pos, d, n):
                        v += 1
                # As second term: pos-d, pos, pos+d, pos+2d
                for d in range(1, (pos - 1) // 2 + 1):
                    if pos - d >= 1 and not _check_ap(colors, pos - d, d, n):
                        v += 1
                # As third term: pos-2d, pos-d, pos, pos+d
                for d in range(1, (pos - 1) // 2 + 1):
                    if pos - 2 * d >= 1 and not _check_ap(colors, pos - 2 * d, d, n):
                        v += 1
                # As fourth term: pos-3d, pos-2d, pos-d, pos
                for d in range(1, (pos - 1) // 3 + 1):
                    if pos - 3 * d >= 1 and not _check_ap(colors, pos - 3 * d, d, n):
                        v += 1
                if v < best_violations:
                    best_violations = v
                    best_color = c
                if v == 0:
                    break
            colors[idx] = best_color
        if _is_valid(colors, n):
            return colors
    return None

def _local_search(colors, n, K, rng, deadline):
    """Local search: try to reduce number of colors used."""
    colors = list(colors)
    used = set(colors)
    for _ in range(5000):
        if time.time() > deadline:
            break
        # Try to recolor a position to a color already in use
        idx = rng.randrange(n)
        old_c = colors[idx]
        # Try each used color
        for c in sorted(used):
            if c == old_c:
                continue
            colors[idx] = c
            if _is_valid(colors, n):
                used.discard(old_c)
                if len(used) < K:
                    return colors
                break
        else:
            colors[idx] = old_c
    return colors

def construct():
    n = 128
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    rng = random.Random(42)
    
    # Try increasing number of colors
    for K in range(3, 10):
        if time.time() > deadline:
            break
        # Try greedy construction
        result = _greedy_construct(n, K, rng, deadline)
        if result is not None:
            # Try to reduce colors further via local search
            result = _local_search(result, n, K, rng, deadline)
            return result
    
    # Fallback: deterministic greedy (like seed but with small palette)
    colors = []
    for x in range(1, 129):
        placed = False
        for color in range(3):
            ok = True
            # Check all APs ending at x
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a >= 1:
                    c0 = colors[a - 1]
                    c1 = colors[a + d - 1]
                    c2 = colors[a + 2 * d - 1]
                    if len({c0, c1, c2, color}) < 3:
                        ok = False
                        break
            if ok:
                colors.append(color)
                placed = True
                break
        if not placed:
            # Try color 3, 4, ...
            for color in range(3, 128):
                ok = True
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a >= 1:
                        c0 = colors[a - 1]
                        c1 = colors[a + d - 1]
                        c2 = colors[a + 2 * d - 1]
                        if len({c0, c1, c2, color}) < 3:
                            ok = False
                            break
                if ok:
                    colors.append(color)
                    placed = True
                    break
        if not placed:
            colors.append(0)  # shouldn't happen
    
    return colors
