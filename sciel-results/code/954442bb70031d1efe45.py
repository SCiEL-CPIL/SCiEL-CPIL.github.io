"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start = time.time()
    deadline = start + 118.0

    # 30 required differences -> bit 0..29 represent d = 1..30
    REQUIRED = tuple(range(1, 31))
    FULL_MASK = (1 << 30) - 1

    def elapsed_ok():
        return time.time() < deadline

    # ---------- Bitset helpers ----------
    def mask_of_diffs(diffs):
        m = 0
        for d in diffs:
            if 1 <= d <= 30:
                m |= (1 << (d - 1))
        return m

    # ---------- Known / heuristic candidates ----------
    def greedy_cover():
        """Greedy: add the smallest x that is Sidon-valid and covers the most missing diffs."""
        a = [0]
        used = set()
        for _ in range(40):
            missing = [d for d in REQUIRED if d not in used]
            if not missing:
                break
            best_x = None
            best_cov = -1
            for x in range(1, 2000):
                new_diffs = []
                ok = True
                for v in a:
                    d = x - v
                    if d <= 0:
                        continue
                    if d in used:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue
                cov = sum(1 for d in new_diffs if d in missing)
                if cov > best_cov:
                    best_cov = cov
                    best_x = x
                    if best_cov == len(missing):
                        break
            if best_x is None or best_cov <= 0:
                # force add a far element to keep progress
                x = max(a) + 31
                while True:
                    ok = True
                    for v in a:
                        d = x - v
                        if d in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                a.append(x)
                for v in a[:-1]:
                    used.add(x - v)
                break
            a.append(best_x)
            for v in a[:-1]:
                used.add(best_x - v)
        return a

    def is_valid(a):
        """Return (is_sidon, covers_all, diffs_set)."""
        diffs = set()
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                d = a[j] - a[i]
                if d in diffs:
                    return False, False, diffs
                diffs.add(d)
        covers = all(d in diffs for d in REQUIRED)
        return True, covers, diffs

    def local_search(a, deadline):
        """Refinement: shrink the largest element, then any element, to smaller values."""
        a = sorted(a)
        best = list(a)
        best_diam = a[-1]
        while elapsed_ok() and time.time() < deadline:
            improved = False
            n = len(a)
            # Try to reduce the largest element first
            last = n - 1
            old = a[last]
            for nv in range(1, old):
                if nv in a:
                    continue
                trial = a[:last] + [nv]
                trial.sort()
                ok, covers, _ = is_valid(trial)
                if ok and covers:
                    a = trial
                    best = list(a)
                    best_diam = a[-1]
                    improved = True
                    break
            # Try to reduce any other element
            if not improved:
                for i in range(n - 1, 0, -1):
                    if not elapsed_ok():
                        break
                    old = a[i]
                    for nv in range(1, old):
                        if nv in a:
                            continue
                        trial = a[:i] + [nv] + a[i + 1:]
                        trial.sort()
                        ok, covers, _ = is_valid(trial)
                        if ok and covers:
                            a = trial
                            best = list(a)
                            best_diam = a[-1]
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        return best, best_diam

    # ---------- Fast backtracking (structural core) ----------
    def backtrack_minimal(max_diam, deadline):
        """Search in increasing x; return first valid set with max <= max_diam."""
        elements = [0]
        used = set()
        covered = 0  # bitmask of covered required diffs

        def rec():
            if not elapsed_ok():
                return None
            if covered == FULL_MASK:
                return list(elements)
            # prune: cannot beat current best diameter
            last = elements[-1]
            upper = min(max_diam, 120)
            for x in range(last + 1, upper + 1):
                if not elapsed_ok():
                    return None
                ok = True
                new_cov = 0
                for v in elements:
                    d = x - v
                    if d in used:
                        ok = False
                        break
                    if 1 <= d <= 30:
                        new_cov |= (1 << (d - 1))
                if not ok:
                    continue
                elements.append(x)
                for v in elements[:-1]:
                    used.add(x - v)
                result = rec()
                if result is not None:
                    return result
                elements.pop()
                for v in elements:
                    used.discard(x - v)
            return None

        return rec()

    # ---------- Driver ----------
    best = None
    best_diam = float('inf')

    # Phase 1: greedy gives an initial valid set (fast)
    g = greedy_cover()
    ok, covers, _ = is_valid(g)
    if ok and covers:
        best = g
        best_diam = g[-1]

    # Phase 2: backtracking in increasing diameter order
    lo = 1
    hi = (best_diam if best is not None else 120) - 1
    while elapsed_ok() and lo <= hi:
        mid = (lo + hi) // 2
        r = backtrack_minimal(mid, deadline)
        if r is not None:
            ok, covers, _ = is_valid(r)
            if ok and covers:
                best = r
                best_diam = r[-1]
                hi = mid - 1
            else:
                hi = mid - 1
        else:
            lo = mid + 1

    # Phase 3: local search refinement
    if best is not None and elapsed_ok():
        improved, improved_diam = local_search(best, deadline - 2.0)
        if improved_diam < best_diam:
            best = improved
            best_diam = improved_diam

    # Fallback: ensure we return something valid
    if best is None:
        best = greedy_cover()
        while True:
            ok, covers, _ = is_valid(best)
            if ok and covers:
                break
            x = max(best) + 31
            while True:
                ok2 = True
                for v in best:
                    if (x - v) in {a - b for i, a in enumerate(best) for b in best[:i]}:
                        ok2 = False
                        break
                if ok2:
                    break
                x += 1
            best.append(x)

    return best
# EVOLVE-BLOCK-END
