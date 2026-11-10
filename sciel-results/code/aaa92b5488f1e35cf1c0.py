# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def nikodym_set() -> np.ndarray:
    """
    Build a (small) Nikodym set in F_49^2 by maximizing its complement C via a
    randomized greedy set-cover heuristic.

    Field model: F_49 = F_7[alpha], alpha^2 = 3.  A point is ((xa,xb),(ya,yb)).

    A point x may belong to C iff EVERY line through x (direction (1,m) for m in
    F_49, plus the vertical (0,1)) contains at least one other point of C.
    Equivalently, define need[x] = the number of lines through x that are still
    'open' (i.e. contain no other point of C).  x is admissible in C exactly when
    need[x] == 0.  We grow C greedily: repeatedly add the admissible point that
    covers the most still-admissible points, with random tie-breaking and a
    randomized restart, to escape poor local structure.  The Nikodym set is
    N = F_49^2 \ C.
    """
    p = 7
    q = p * p
    K = 3  # alpha^2 = 3 (quadratic non-residue mod 7)

    def idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute all q+1 directions (as (dx_a, dx_b, dy_a, dy_b)).
    dirs = []
    for ma in range(p):
        for mb in range(p):
            dirs.append((1, 0, ma, mb))  # (1, m)
    dirs.append((0, 1, 0, 0))            # vertical (0, 1)
    nd = len(dirs)

    # Line membership: for each direction d and each point x on the line,
    # the other q-1 points on that line.
    line_members = []  # line_members[d][x] -> list of q-1 indices
    for d in range(nd):
        dx_a, dx_b, dy_a, dy_b = dirs[d]
        for x in range(q * q):
            xa = (x // (p * p * p)) % p
            xb = (x // (p * p)) % p
            ya = (x // p) % p
            yb = x % p
            others = []
            for t in range(1, q):
                nx_a = (xa + t * dx_a) % p
                nx_b = (xb + t * dx_b) % p
                ny_a = (ya + t * dy_a) % p
                ny_b = (yb + t * dy_b) % p
                others.append(idx(nx_a, nx_b, ny_a, ny_b))
            line_members.append(others)
    line_members = np.array(line_members, dtype=np.int64)  # (nd, q*q, q-1)

    # Reverse incidence: for each point y, the list of (x, d) such that y is the
    # 'other' member of line (d, x).  i.e. x is on the same line as y in dir d.
    # When y enters C, need[x] -= 1 for every such (x, d).
    rev = [[] for _ in range(q * q)]
    for d in range(nd):
        for x in range(q * q):
            for y in line_members[d][x]:
                rev[int(y)].append((int(x), d))

    start = time.time()
    deadline = 560.0
    best_complement = -1
    best_set = None

    rng = random.Random(20240711)

    def run_greedy(seed):
        nonlocal best_complement, best_set
        r = random.Random(seed)
        need = np.full(q * q, nd, dtype=np.int64)
        inC = np.zeros(q * q, dtype=bool)
        complement = []
        while True:
            # admissible = need == 0 and not inC
            adm = (need == 0) & (~inC)
            if not np.any(adm):
                break
            adm_idx = np.nonzero(adm)[0]
            # For each candidate y, coverage = number of admissible points x
            # (that are not y) such that y is the 'other' on line (d, x).
            # We can compute coverage by, for each y, summing over its rev list
            # of (x, d) where x is admissible and x != y.
            best_y = -1
            best_cov = -1
            # sample-based to keep it fast: if many candidates, sample a subset
            if len(adm_idx) > 64:
                sample = rng.sample(list(adm_idx), 64)
            else:
                sample = list(adm_idx)
            for y in sample:
                cov = 0
                for (x, d) in rev[y]:
                    if x == y:
                        continue
                    if inC[x]:
                        continue
                    if need[x] == 0:
                        cov += 1
                if cov > best_cov:
                    best_cov = cov
                    best_y = y
            if best_y < 0:
                break
            # commit
            inC[best_y] = True
            complement.append(best_y)
            for (x, d) in rev[best_y]:
                if not inC[x]:
                    need[x] -= 1
        return complement

    # First, a deterministic greedy pass.
    comp = run_greedy(0)
    if len(comp) > best_complement:
        best_complement = len(comp)
        best_set = comp

    # Randomized restarts until deadline.
    s = 1
    while time.time() < deadline:
        comp = run_greedy(s)
        if len(comp) > best_complement:
            best_complement = len(comp)
            best_set = comp
        s += 1

    if best_set is None or best_complement <= 0:
        # Fallback: a guaranteed-valid (small) construction.
        # Use the unital-based exclusion so the result is at least valid.
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
        pts = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * K) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            continue
                        pts.append([[xa, xb], [ya, yb]])
        return np.array(pts, dtype=np.int64)

    # Build N = all points minus complement.
    comp_set = set(int(c) for c in best_set)
    pts = []
    for x in range(q * q):
        if x in comp_set:
            continue
        xa = (x // (p * p * p)) % p
        xb = (x // (p * p)) % p
        ya = (x // p) % p
        yb = x % p
        pts.append([[xa, xb], [ya, yb]])

    return np.array(pts, dtype=np.int64)


# EVOLVE-BLOCK-END
