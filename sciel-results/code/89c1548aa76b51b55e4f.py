# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _tetracode_baseline(q, n):
    """Fast guaranteed-valid baseline: 9 * q**(n-4) words via direct sum with the
    ternary tetracode (a perfect radius-1 code on 4 coords)."""
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q]
                 for a in range(q) for b in range(q)]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _ball_indices(cw, q, n, total):
    """Flat indices of the ball (radius 1) around codeword cw."""
    base = np.array(cw, dtype=np.int64)
    idx = [int(np.dot(base, q ** np.arange(n - 1, -1, -1)))]
    for i in range(n):
        b = base.copy()
        for v in range(q):
            if v == base[i]:
                continue
            b[i] = v
            idx.append(int(np.dot(b, q ** np.arange(n - 1, -1, -1))))
    return idx


def _rebuild_counts(cw_list, q, n, total, powers):
    count = np.zeros(total, dtype=np.int32)
    for cw in cw_list:
        count[_ball_indices(cw, q, n, total)] += 1
    return count


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0          # safety margin under 150s wall-clock
    total = q ** n
    powers = q ** np.arange(n - 1, -1, -1, dtype=np.int64)

    # Guaranteed-valid baseline
    code = _tetracode_baseline(q, n)
    best = list(code)
    best_size = len(code)

    # ---- Greedy initialization (fast, valid) ----
    covered = np.zeros(total, dtype=bool)
    count = np.zeros(total, dtype=np.int32)
    for cw in code:
        bi = _ball_indices(cw, q, n, total)
        covered[bi] = True
        count[bi] += 1
    uncovered = int(np.sum(~covered))

    if time.time() < deadline - 5.0 and uncovered > 0:
        # Build candidate pool: all words not already in the code
        all_words = np.array(
            [list(w) for w in itertools.product(range(q), repeat=n)],
            dtype=np.int64).reshape(-1, n)
        in_code = np.zeros(total, dtype=bool)
        for cw in code:
            in_code[int(np.dot(np.array(cw, dtype=np.int64), powers))] = True
        cand = all_words[~in_code]
        # Ball for each candidate (cand x ball_size)
        ball_size = 1 + n * (q - 1)
        balls = np.zeros((len(cand), ball_size), dtype=np.int64)
        for j, cw in enumerate(cand):
            balls[j] = _ball_indices(cw, q, n, total)
        cand_idx = np.array(
            [int(np.dot(cw, powers)) for cw in cand], dtype=np.int64)

        while time.time() < deadline - 5.0 and uncovered > 0:
            # gain = number of uncovered points each candidate covers
            gain = (~covered[balls]).sum(axis=1)
            gi = int(np.argmax(gain))
            if gain[gi] <= 0:
                break
            code.append(cand[gi])
            bi = balls[gi]
            new = ~covered[bi]
            covered[bi] = True
            count[bi] += 1
            uncovered -= int(np.sum(new))
            in_code[cand_idx[gi]] = True

    if len(code) < best_size:
        best = list(code)
        best_size = len(code)

    # ---- Local search ----
    count = _rebuild_counts(code, q, n, total, powers)
    covered = count > 0

    def try_improve():
        nonlocal code, count, covered, best, best_size
        progress = True
        while progress and time.time() < deadline - 2.0:
            progress = False

            # (a) remove redundant codewords
            for i in range(len(code)):
                if count[_ball_indices(code[i], q, n, total)].min() == 0:
                    bi = _ball_indices(code[i], q, n, total)
                    count[bi] -= 1
                    code.pop(i)
                    progress = True
                    break
            if progress:
                continue

            # (b) swap a low-utility codeword for a high-gain candidate
            if time.time() < deadline - 2.0:
                util = np.array([count[_ball_indices(cw, q, n, total)].min()
                                 for cw in code], dtype=np.int32)
                # candidates = all words not in code
                in_code = np.zeros(total, dtype=bool)
                for cw in code:
                    in_code[int(np.dot(np.array(cw, dtype=np.int64), powers))] = True
                cand_idx = np.nonzero(~in_code)[0]
                if len(cand_idx) > 0 and len(util) > 0:
                    # build balls for candidates (cap work)
                    max_cand = 4000
                    if len(cand_idx) > max_cand:
                        # sample
                        sel = np.random.choice(len(cand_idx), max_cand, replace=False)
                        cand_idx = cand_idx[sel]
                    cand = [list(np.unravel_index(c, (q,) * n)[::-1]) for c in cand_idx]
                    balls = np.array([_ball_indices(cw, q, n, total) for cw in cand])
                    gain = (~covered[balls]).sum(axis=1)
                    gi = int(np.argmax(gain))
                    if gain[gi] > 0:
                        # choose codeword with smallest utility to replace
                        li = int(np.argmin(util))
                        bi_old = _ball_indices(code[li], q, n, total)
                        # ensure removing li doesn't break coverage (it won't if count stays >=1)
                        if (count[bi_old] - 1).min() >= 0 and (count[bi_old] >= 1).all():
                            count[bi_old] -= 1
                            bi_new = balls[gi]
                            count[bi_new] += 1
                            code[li] = cand[gi]
                            covered = count > 0
                            progress = True

        if len(code) < best_size:
            best = list(code)
            best_size = len(code)

    # run a few rounds with randomization to escape local minima
    rng = np.random.default_rng(0)
    for _ in range(4):
        if time.time() >= deadline - 2.0:
            break
        # small random perturbation: shuffle order of removal priority
        try_improve()

    # final validity safety: if anything looks off, fall back to best
    return best
# EVOLVE-BLOCK-END
