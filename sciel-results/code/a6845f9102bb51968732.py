"""Sidon sets containing 0 with differences 1..30 – branch-and-bound + local search."""
# EVOLVE-BLOCK-START
import time
import random
import math

_DEADLINE = 150.0
_MARGIN = 5.0

# ---------------------------------------------------------------------------
# Fast incremental Sidon tracker
# ---------------------------------------------------------------------------
class _Tracker:
    __slots__ = ("arr", "diff_count", "cov")

    def __init__(self, arr):
        self.arr = sorted(arr)
        self.diff_count = {}
        self.cov = 0
        for i in range(len(self.arr)):
            ai = self.arr[i]
            for j in range(i + 1, len(self.arr)):
                d = self.arr[j] - ai
                c = self.diff_count.get(d, 0) + 1
                self.diff_count[d] = c
                if c == 1 and 1 <= d <= 30:
                    self.cov += 1

    def is_valid(self):
        if self.cov < 30:
            return False
        return all(c == 1 for c in self.diff_count.values())

    def add(self, x):
        a = self.arr
        n = len(a)
        i = 0
        while i < n and a[i] < x:
            i += 1
        a.insert(i, x)
        dc = self.diff_count
        for j in range(i):
            d = x - a[j]
            c = dc.get(d, 0) + 1
            dc[d] = c
            if c == 1 and 1 <= d <= 30:
                self.cov += 1
        for j in range(i + 1, len(a)):
            d = a[j] - x
            c = dc.get(d, 0) + 1
            dc[d] = c
            if c == 1 and 1 <= d <= 30:
                self.cov += 1

    def remove(self, x):
        a = self.arr
        n = len(a)
        i = 0
        while i < n and a[i] != x:
            i += 1
        if i >= n:
            return
        a.pop(i)
        dc = self.diff_count
        for j in range(i):
            d = x - a[j]
            c = dc[d] - 1
            if c == 0:
                del dc[d]
                if 1 <= d <= 30:
                    self.cov -= 1
            else:
                dc[d] = c
        for j in range(i, len(a)):
            d = a[j] - x
            c = dc[d] - 1
            if c == 0:
                del dc[d]
                if 1 <= d <= 30:
                    self.cov -= 1
            else:
                dc[d] = c

    def max(self):
        return self.arr[-1]


def _valid_list(s):
    if 0 not in s or len(s) < 2 or len(s) > 32:
        return False
    if max(s) > 10**6:
        return False
    t = _Tracker(s)
    return t.is_valid()


# ---------------------------------------------------------------------------
# Greedy constructive: build a valid Sidon set covering 1..30 within diameter cap
# ---------------------------------------------------------------------------
def _greedy_construct(cap, rng, time_limit):
    arr = [0]
    dc = {}
    cov = 0
    need = set(range(1, 31))

    def add(x):
        nonlocal cov
        n = len(arr)
        i = 0
        while i < n and arr[i] < x:
            i += 1
        arr.insert(i, x)
        for j in range(i):
            d = x - arr[j]
            c = dc.get(d, 0) + 1
            dc[d] = c
            if c == 1 and d in need:
                cov += 1
                need.discard(d)
        for j in range(i + 1, len(arr)):
            d = arr[j] - x
            c = dc.get(d, 0) + 1
            dc[d] = c
            if c == 1 and d in need:
                cov += 1
                need.discard(d)

    def remove(x):
        nonlocal cov
        n = len(arr)
        i = 0
        while i < n and arr[i] != x:
            i += 1
        if i >= n:
            return
        arr.pop(i)
        for j in range(i):
            d = x - arr[j]
            c = dc[d] - 1
            if c == 0:
                del dc[d]
                if d in range(1, 31):
                    cov -= 1
                    need.add(d)
            else:
                dc[d] = c
        for j in range(i, len(a)):
            d = a[j] - x
            c = dc[d] - 1
            if c == 0:
                del dc[d]
                if d in range(1, 31):
                    cov -= 1
                    need.add(d)
            else:
                dc[d] = c

    def sidon_ok():
        return all(c == 1 for c in dc.values())

    for _ in range(500):
        if not need:
            break
        if len(arr) > 31:
            break
        if time.time() > time_limit:
            return None
        best_x, best_gain = None, 0
        cand = []
        for x in range(1, cap + 1):
            if x in arr:
                continue
            ok = True
            gain = 0
            for v in arr:
                d = abs(x - v)
                if d in dc and dc[d] >= 1:
                    ok = False
                    break
                if d in need:
                    gain += 1
            if not ok:
                continue
            cand.append((gain, x))
        cand.sort(reverse=True)
        for gain, x in cand[:30]:
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is not None and best_gain > 0:
            add(best_x)
        else:
            break
    if need:
        return None
    return sorted(arr)


# ---------------------------------------------------------------------------
# Local shrink: replace elements with smaller values to reduce diameter
# ---------------------------------------------------------------------------
def _local_shrink(arr, rng, deadline):
    t = _Tracker(arr)
    improved = True
    while improved and time.time() < deadline - _MARGIN:
        improved = False
        D = t.max()
        for i in range(len(t.arr) - 1, 0, -1):
            x = t.arr[i]
            for _ in range(40):
                v = rng.randint(1, D - 1)
                if v in t.arr:
                    continue
                t.remove(x)
                t.add(v)
                if t.is_valid():
                    improved = True
                else:
                    t.remove(v)
                    t.add(x)
                if improved:
                    break
    return list(t.arr)


# ---------------------------------------------------------------------------
# Try removing elements to reduce size
# ---------------------------------------------------------------------------
def _try_remove_elements(arr, deadline):
    t = _Tracker(arr)
    changed = True
    while changed and time.time() < deadline - _MARGIN:
        changed = False
        for i in range(len(t.arr) - 1, 0, -1):
            if t.arr[i] == 0:
                continue
            x = t.arr[i]
            t.remove(x)
            if t.is_valid():
                changed = True
            else:
                t.add(x)
    return list(t.arr)


# ---------------------------------------------------------------------------
# Simulated annealing refinement
# ---------------------------------------------------------------------------
def _sa_refine(arr, rng, deadline, T0=40.0, cooling=0.997, max_iter=3000):
    t = _Tracker(arr)
    best = list(t.arr)
    best_max = t.max()
    cur_max = best_max
    T = T0
    for _ in range(max_iter):
        if time.time() > deadline - _MARGIN:
            break
        i = rng.randint(1, len(t.arr) - 1)
        old = t.arr[i]
        if rng.random() < 0.6:
            delta = rng.randint(-20, 20)
            new = old + delta
        else:
            new = rng.randint(1, t.max())
        if new <= 0 or new in t.arr:
            continue
        t.remove(old)
        t.add(new)
        if t.is_valid():
            nm = t.max()
            if nm < cur_max:
                cur_max = nm
                if nm < best_max:
                    best_max = nm
                    best = list(t.arr)
        else:
            if rng.random() < math.exp(-1.0 / max(T, 0.1)):
                cur_max = t.max()
            else:
                t.remove(new)
                t.add(old)
        T *= cooling
    return best


# ---------------------------------------------------------------------------
# Branch-and-bound search for minimum diameter Sidon set
# ---------------------------------------------------------------------------
def _branch_and_bound(max_diameter, rng, deadline):
    """
    Search for a Sidon set containing 0 with all differences 1..30
    and max element <= max_diameter.
    
    Uses recursive backtracking with pruning.
    """
    best = None
    best_size = 33
    
    def search(current_set, current_dc, current_cov, current_need, depth):
        nonlocal best, best_size
        if time.time() > deadline - _MARGIN:
            return
        if not current_need:
            # Found a complete solution
            s = sorted(current_set)
            if len(s) < best_size:
                best_size = len(s)
                best = list(s)
            return
        if len(current_set) >= 32:
            return
        if depth > 32:
            return
        
        # Try adding each possible element
        # Sort candidates by how many new differences they cover
        candidates = []
        for x in range(1, max_diameter + 1):
            if x in current_set:
                continue
            # Check if x can be added without violating Sidon
            ok = True
            gain = 0
            for v in current_set:
                d = abs(x - v)
                if d in current_dc and current_dc[d] >= 1:
                    ok = False
                    break
                if d in current_need:
                    gain += 1
            if not ok:
                continue
            candidates.append((gain, x))
        
        # Sort by gain descending
        candidates.sort(reverse=True)
        
        # Limit branching factor
        for gain, x in candidates[:15]:
            if gain == 0:
                break
            # Add x
            new_set = list(current_set) + [x]
            new_dc = dict(current_dc)
            new_cov = current_cov
            new_need = set(current_need)
            
            for v in current_set:
                d = abs(x - v)
                c = new_dc.get(d, 0) + 1
                new_dc[d] = c
                if c == 1 and d in new_need:
                    new_cov += 1
                    new_need.discard(d)
            
            search(new_set, new_dc, new_cov, new_need, depth + 1)
            
            if best is not None and len(best) <= len(current_set):
                # No point continuing if we can't beat current best
                return
    
    # Start search
    search([0], {}, 0, set(range(1, 31)), 0)
    return best


# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(42)

    best = None
    best_max = 10**6

    def consider(s):
        nonlocal best, best_max
        if s is None or not _valid_list(s):
            return
        m = max(s)
        if m < best_max:
            best = list(s)
            best_max = m

    # Phase 1: Greedy constructive with various caps
    for cap in [35, 40, 45, 50, 55, 60, 70, 80, 100, 130, 170, 250]:
        if time.time() > deadline - _MARGIN:
            break
        for _ in range(3):
            if time.time() > deadline - _MARGIN:
                break
            s = _greedy_construct(cap, rng, deadline - _MARGIN)
            consider(s)

    # Phase 2: Branch-and-bound for smaller diameters
    for max_d in range(35, 60):
        if time.time() > deadline - _MARGIN:
            break
        s = _branch_and_bound(max_d, rng, deadline)
        consider(s)

    # Phase 3: SA refinement on best found
    if best is not None:
        if time.time() < deadline - _MARGIN:
            consider(_sa_refine(list(best), rng, deadline, T0=50.0, cooling=0.995, max_iter=4000))
        if time.time() < deadline - _MARGIN:
            consider(_try_remove_elements(list(best), deadline))
        if time.time() < deadline - _MARGIN:
            consider(_local_shrink(list(best), rng, deadline))
        if time.time() < deadline - _MARGIN:
            consider(_sa_refine(list(best), rng, deadline, T0=30.0, cooling=0.997, max_iter=2000))
        if time.time() < deadline - _MARGIN:
            consider(_try_remove_elements(list(best), deadline))

    if best is None:
        # Fallback: guaranteed-valid large construction
        a = [0]
        def diffs(ss):
            return [y - x for i, x in enumerate(ss) for y in ss[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            placed = False
            for x in sorted(set(v + dd for v in a for dd in (d, -d) if v + dd >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    placed = True
                    break
            if not placed:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        return a

    return best
# EVOLVE-BLOCK-END
