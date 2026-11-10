"""Rewritten program: narrow admissible tuple via smooth surrogate + local search."""
import time
import math
import random

N = 5000
DEADLINE = 320.0  # safety margin before 330s kill

# ---- primes ----
def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# primes up to N (covers any tuple length up to 5001)
_PRIMES_N = _primes(N)
# primes up to 200 (enough: admissibility only needs primes <= len(tuple) <= 5001,
# but for size ~2000-3000 we need primes up to that; use full set up to N)
_PRIMES_FULL = _primes(N)

# For the optimization we only need primes up to a cap K (max tuple size we expect).
# Larger primes p > N/2 can only remove 1 element (floor(5000/p)+1 = 1 or 2).
# We'll include primes up to ~ 2000 to be safe.
K_CAP = 2000
_PRIMES_OPT = [p for p in _PRIMES_FULL if p <= K_CAP]

# Precompute, for each prime p and residue r, the count of elements in [0,N] with that residue
# and the actual membership bitset (as a set of ints) for fast union.
def _build_class_data():
    data = {}
    for p in _PRIMES_OPT:
        counts = [0] * p
        for v in range(N + 1):
            counts[v % p] += 1
        # membership sets per residue
        sets = [set() for _ in range(p)]
        for v in range(N + 1):
            sets[v % p].add(v)
        data[p] = (counts, sets)
    return data

_CLASS_DATA = _build_class_data()

def _kept_count(selection):
    """selection: dict prime->residue. Returns number of elements NOT in any removed class."""
    removed = set()
    for p, r in selection.items():
        removed |= _CLASS_DATA[p][1][r]
    return (N + 1) - len(removed)

def _admissible(points):
    k = len(points)
    for p in _PRIMES_FULL:
        if p > k:
            break
        seen = set()
        for v in points:
            seen.add(v % p)
        if len(seen) == p:
            return False
    return True

def _construct_from_selection(selection):
    removed = set()
    for p, r in selection.items():
        removed |= _CLASS_DATA[p][1][r]
    points = [v for v in range(N + 1) if v not in removed]
    # ensure admissibility: for primes up to len(points), if all classes present, remove least-populated
    changed = True
    while changed:
        changed = False
        k = len(points)
        for p in _PRIMES_FULL:
            if p > k:
                break
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                smallest = min(classes.values(), key=len)
                points = [v for v in points if v not in smallest]
                changed = True
                break
    return points

def _soft_min(x, temperature):
    """Smooth surrogate for min over a list of numbers (negative log-sum-exp)."""
    m = min(x)
    if temperature <= 0:
        return m
    s = 0.0
    for a in x:
        s += math.exp((m - a) / temperature)
    return m - temperature * math.log(s)

def _soft_max(x, temperature):
    m = max(x)
    if temperature <= 0:
        return m
    s = 0.0
    for a in x:
        s += math.exp((a - m) / temperature)
    return m + temperature * math.log(s)

def construct():
    start = time.time()
    random.seed(12345)

    # ---- greedy initialization: for each prime pick the residue class that removes the
    #      most currently-unremoved elements (greedy set cover to minimize union). ----
    def greedy_init():
        selection = {}
        # work with removed set
        removed = set()
        # order primes by size of best class (largest primes have small classes; process
        # small primes first since they remove many)
        primes_sorted = sorted(_PRIMES_OPT, key=lambda p: min(_CLASS_DATA[p][0]), reverse=True)
        for p in primes_sorted:
            counts, sets = _CLASS_DATA[p]
            best_r = -1
            best_gain = float('inf')
            for r in range(p):
                gain = len(sets[r] - removed)
                if gain < best_gain:
                    best_gain = gain
                    best_r = r
            selection[p] = best_r
            removed |= sets[best_r]
        return selection

    best_sel = greedy_init()
    best_points = _construct_from_selection(best_sel)
    best_score = len(best_points)

    # ---- local search / simulated-annealing over residue choices using smooth surrogate ----
    # Objective to maximize: kept count. We use soft-min of per-prime "coverage cost".
    # Simpler: hill-climb / SA on kept_count directly with random single-prime reassignment.
    def evaluate(sel):
        return _kept_count(sel)

    cur_sel = dict(best_sel)
    cur_kept = evaluate(cur_sel)

    # temperature schedule
    T0 = 5.0
    T_end = 0.01
    iters = 0
    improved = 0
    while time.time() < DEADLINE:
        iters += 1
        # pick a random prime to reassign
        p = random.choice(_PRIMES_OPT)
        counts, sets = _CLASS_DATA[p]
        old_r = cur_sel[p]
        new_r = random.randrange(p)
        if new_r == old_r:
            continue
        cur_sel[p] = new_r
        new_kept = evaluate(cur_sel)
        delta = new_kept - cur_kept
        if delta >= 0 or random.random() < math.exp(delta / max(T0, 1e-9)):
            cur_kept = new_kept
            if new_kept > best_score:
                best_score = new_kept
                best_sel = dict(cur_sel)
                best_points = _construct_from_selection(best_sel)
                improved += 1
        else:
            cur_sel[p] = old_r
        # cool down
        T0 = max(T_end, T0 * 0.9999)
        if time.time() > DEADLINE - 10:
            break

    # ---- final polish: deterministic local search (best single-prime move) ----
    def polish(points):
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            k = len(points)
            for p in _PRIMES_FULL:
                if p > k:
                    break
                classes = {}
                for v in points:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    # must remove one class; choose the one whose removal keeps most
                    # and is consistent; pick least-populated
                    smallest = min(classes.values(), key=len)
                    new_points = [v for v in points if v not in smallest]
                    points = new_points
                    improved = True
        return points

    best_points = polish(best_points)
    best_score = len(best_points)

    # ---- multi-start: try several greedy orderings / random initial selections ----
    for trial in range(20):
        if time.time() > DEADLINE - 15:
            break
        # random-ish init: for each prime pick residue weighted by class size (prefer large removals)
        sel = {}
        for p in _PRIMES_OPT:
            counts, sets = _CLASS_DATA[p]
            # pick residue with largest count (removes most), with random tie-break
            maxc = max(counts)
            candidates = [r for r in range(p) if counts[r] == maxc]
            sel[p] = random.choice(candidates)
        pts = _construct_from_selection(sel)
        pts = polish(pts)
        if len(pts) > best_score:
            best_score = len(pts)
            best_points = pts

    if time.time() > DEADLINE - 3:
        return best_points

    return best_points

if __name__ == "__main__":
    r = construct()
    print(len(r))
