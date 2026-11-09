import random
import time

N = 16
NP = N * N  # 256
START = 0.0
DEADLINE = 310.0  # safety margin before 330s kill

# Precompute conflict matrix: conflict[i] is a bitmask of points j such that
# {i, j} participates in an isosceles triangle (i.e., there exists a grid point
# k != i, j with a duplicate distance). Points in conflict[i] cannot coexist.
CONFLICT = [0] * NP
for i in range(NP):
    xi, yi = divmod(i, N)
    dm = {}
    for j in range(NP):
        if j == i:
            continue
        xj, yj = divmod(j, N)
        d = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
        dm.setdefault(d, []).append(j)
    mask = 0
    for lst in dm.values():
        if len(lst) >= 2:
            for j in lst:
                mask |= (1 << j)
    CONFLICT[i] = mask


def _valid(s):
    """s is a bitmask; returns True if no two points in s are mutually conflicting."""
    for i in range(NP):
        if (s >> i) & 1:
            if s & CONFLICT[i]:
                return False
    return True


def _maximal(s):
    """s is maximal iff no outside point can be added (all outside points conflict)."""
    for j in range(NP):
        if not ((s >> j) & 1) and (s & CONFLICT[j]) == 0:
            return False
    return True


def _build_greedy(rng):
    s = 0
    for _ in range(NP):
        # find all addable points
        addable = [j for j in range(NP)
                   if not ((s >> j) & 1) and (s & CONFLICT[j]) == 0]
        if not addable:
            break
        j = rng.choice(addable)
        s |= (1 << j)
    # Ensure maximal (greedy until no addable point guarantees maximality)
    return s


def _anneal(s, rng, deadline):
    s = int(s)
    cur = s
    cur_size = bin(cur).count("1")
    best = cur
    best_size = cur_size
    T = 0.5
    Tmin = 1e-5
    t0 = time.time()
    step = 0
    while time.time() < deadline:
        step += 1
        # Randomly pick an action: insertion or removal
        if rng.random() < 0.5:
            # Try inserting a random outside point (if it keeps validity)
            # Build list of outside points occasionally (cheap enough)
            # Pick random j
            j = rng.randrange(NP)
            if (cur >> j) & 1:
                continue
            if cur & CONFLICT[j]:
                continue
            new = cur | (1 << j)
            new_size = cur_size + 1
            # validity guaranteed; maximality may be lost
            # accept if improves best or with probability
            if new_size < best_size and _maximal(new):
                best = new
                best_size = new_size
            # Accept insertions only rarely (they grow the set); use them to
            # escape local maxima of the removal process.
            if rng.random() < 0.05:
                cur = new
                cur_size = new_size
        else:
            # Try removing a random point
            # collect set points
            pts = [i for i in range(NP) if (cur >> i) & 1]
            if not pts:
                break
            i = rng.choice(pts)
            new = cur & ~(1 << i)
            new_size = cur_size - 1
            if not _valid(new):
                continue  # shouldn't happen since removing preserves validity
            if _maximal(new):
                # forced removal that keeps maximality: always accept
                cur = new
                cur_size = new_size
                if new_size < best_size:
                    best = new
                    best_size = new_size
            else:
                # removal breaks maximality: accept with annealing prob to escape
                if rng.random() < T:
                    cur = new
                    cur_size = new_size
        # cool down
        T = max(Tmin, T * 0.999)
    return best, best_size


def construct():
    global START
    if START == 0.0:
        START = time.time()
    deadline = time.time() + (DEADLINE - (time.time() - START))
    deadline = min(deadline, time.time() + 310)
    if deadline - time.time() < 5:
        deadline = time.time() + 310

    rng = random.Random(12345)
    best = None
    best_size = NP + 1

    # Initial solution
    s0 = _build_greedy(rng)
    if _maximal(s0):
        sz = bin(s0).count("1")
        best = s0
        best_size = sz

    # Run annealing for the budget
    while time.time() < deadline - 2:
        s, sz = _anneal(s0, rng, deadline - 1)
        if sz < best_size:
            best_size = sz
            best = s
            s0 = s
        # Occasionally re-randomize the starting point
        if rng.random() < 0.1:
            s0 = _build_greedy(rng)

    if best is None:
        # Fallback: pure greedy
        s = 0
        for j in range(NP):
            if (s & CONFLICT[j]) == 0:
                s |= (1 << j)
        best = s

    # Convert bitmask to list of (x, y)
    result = []
    for i in range(NP):
        if (best >> i) & 1:
            x, y = divmod(i, N)
            result.append((x, y))
    return result
