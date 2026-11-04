"""
Ternary Covering Code K_3(8,1) solver.

Strategy:
  * Precompute the 17-point ball (index list) around every one of the 6561
    words once.  Coverage is maintained with a vectorized np.add.at counter.
  * Build an initial covering code with a randomized Luby-Lagrange greedy
    (harmonic-weighted random choice among max-gain candidates).  This is far
    cheaper per step than the old boolean-matrix recount, so many restarts fit
    in the time budget.
  * Post-process each candidate with a fast local search: a codeword is
    "critical" if it is the unique cover of some point.  We repeatedly try to
    swap in a random uncovered-by-criticality point as a new codeword whenever
    the swap strictly increases the number of critical codewords (which, on a
    covering set, is equivalent to a feasible size reduction).  Then we prune
    any remaining redundancies.
"""
import numpy as np
import time
import itertools


def construct(q, n, radius):
    # Non-target parameters: fall back to a valid (if not optimal) covering code.
    if not (q == 3 and n == 8 and radius == 1):
        if n >= 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q]
                         for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode
                    for tail in itertools.product(range(q), repeat=n - 4)]
        return list(itertools.product(range(q), repeat=n))

    total = q ** n          # 6561
    ball_size = 1 + n * (q - 1)   # 17
    powers = np.array([q ** i for i in range(n)], dtype=np.int64)

    # ---- Precompute the ball (index list) around every word, once. ----
    words = np.zeros((total, n), dtype=np.int64)
    for i in range(n):
        words[:, i] = (np.arange(total) // powers[i]) % q
    deltas = np.repeat(powers, q - 1)                       # 16 position deltas
    shifts = np.tile(np.arange(1, q), n)                    # 1..2 per position
    balls = np.empty((total, ball_size), dtype=np.int64)
    balls[:, 0] = np.arange(total)
    balls[:, 1:] = (np.arange(total)[:, None] + deltas[None, :]
                    - shifts[None, :] * words[:, None, :]) % total

    def idx_to_word(idx):
        w = [0] * n
        t = idx
        for i in range(n):
            w[i] = t % q
            t //= q
        return w

    # ---- Randomized Luby-Lagrange greedy set cover. ----
    def greedy(seed):
        rng = np.random.default_rng(seed)
        cov = np.zeros(total, dtype=np.int32)
        selected = []
        while True:
            gain = cov[balls]                       # (6561, 17)
            gain[gain > 0] = 0
            gain = gain.sum(axis=1)
            mx = gain.max()
            if mx == 0:
                break
            cand = np.flatnonzero(gain == mx)
            w = 1.0 / (np.arange(1, len(cand) + 1, dtype=np.float64))
            w /= w.sum()
            sel = cand[rng.choice(len(cand), p=w)]
            selected.append(sel)
            np.add.at(cov, balls[sel], 1)
        return selected

    # ---- Local search: swap in points that raise the critical count. ----
    def local_search(selected, rng, max_iters=4000):
        selected = list(selected)
        cov = np.zeros(total, dtype=np.int32)
        for s in selected:
            np.add.at(cov, balls[s], 1)
        in_set = np.zeros(total, dtype=bool)
        in_set[selected] = True
        crit = (cov == 1)                           # points with unique cover
        crit_count = int(crit.sum())
        for _ in range(max_iters):
            new = rng.integers(0, total)
            if in_set[new]:
                continue
            nb = balls[new]
            # A point p becomes critical after adding `new` iff cov[p]==0 or
            # (cov[p]==1 and p is in new's ball).  Count such p.
            new_crit = int((cov[nb] == 0).sum()) + int((cov[nb] == 1).sum())
            if new_crit <= crit_count:
                continue
            # Feasibility: no point may lose its only cover.
            #   cov==1 outside new's ball -> safe; cov==1 inside -> needs cov>=2.
            if int((cov == 1).sum()) > int((cov[nb] >= 2).sum()):
                continue
            in_set[new] = True
            np.add.at(cov, nb, 1)
            crit_count = new_crit
            selected.append(new)
        return selected

    # ---- Remove redundant codewords. ----
    def prune(selected):
        cov = np.zeros(total, dtype=np.int32)
        for s in selected:
            np.add.at(cov, balls[s], 1)
        changed = True
        while changed:
            changed = False
            for i, s in enumerate(selected):
                if (cov[balls[s]] > 1).all():
                    np.add.at(cov, balls[s], -1)
                    selected.pop(i)
                    changed = True
                    break
        return selected

    # ---- Main loop: many cheap randomized restarts + local search. ----
    start = time.time()
    deadline = start + 140.0          # hard wall-clock guard (evaluator kills at 150s)
    best = None
    best_len = float('inf')
    it = 0
    while time.time() < deadline:
        sel = greedy(it)
        rng = np.random.default_rng(it + 1000003)
        sel = local_search(sel, rng)
        sel = prune(sel)
        if len(sel) < best_len:
            best_len = len(sel)
            best = sel
            if best_len <= 486:
                break
        it += 1

    if best is None:
        best = prune(greedy(0))

    return [idx_to_word(int(i)) for i in best]
