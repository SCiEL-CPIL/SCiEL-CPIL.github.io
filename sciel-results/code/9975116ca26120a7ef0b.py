import time
import random
import itertools
import numpy as np

_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _ball_indices(cw, n, q, radius, inc):
    """List of space indices within Hamming distance <= radius of cw."""
    cw = list(cw)
    base = 0
    for i in range(n):
        base = base * q + cw[i]
    if radius == 0:
        return [base]
    if radius == 1:
        out = [base]
        for i in range(n):
            v = cw[i]
            for d in range(q):
                if d != v:
                    out.append(base + (d - v) * inc[i])
        return out
    # general small radius
    out = [base]
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for changes in itertools.product(range(q), repeat=r):
                idx = base
                for pos, ch in zip(positions, changes):
                    idx += (ch - cw[pos]) * inc[pos]
                out.append(idx)
    return out


def _tetracode_solution(q, n, radius):
    if n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    return [h + list(tail) for h in _TETRA
            for tail in itertools.product(range(q), repeat=n - 4)]


def _is_cover(counts):
    return bool(np.all(counts > 0))


def _greedy_build(q, n, radius, inc, deadline, rng):
    total = q ** n
    counts = np.zeros(total, dtype=np.int32)
    chosen = []
    balls = []
    all_idx = np.arange(total, dtype=np.int64)
    # random starting seed so the build is not fully deterministic
    for _ in range(3):
        cw = [rng.randrange(q) for _ in range(n)]
        for idx in _ball_indices(cw, n, q, radius, inc):
            counts[idx] += 1
        chosen.append(cw)
        balls.append(_ball_indices(cw, n, q, radius, inc))
    while not _is_cover(counts) and time.time() < deadline:
        # gain of a candidate = number of currently-UNcovered points it covers
        gain = np.bincount(all_idx, weights=(counts == 0).astype(np.int64))
        best = int(np.argmax(gain))
        if best <= 0:
            break
        cw = [0] * n
        x = best
        for i in range(n - 1, -1, -1):
            cw[i] = x % q
            x //= q
        for idx in _ball_indices(cw, n, q, radius, inc):
            counts[idx] += 1
        chosen.append(cw)
        balls.append(_ball_indices(cw, n, q, radius, inc))
    return chosen, balls, counts


def _prune(chosen, balls, counts, deadline):
    """Repeatedly remove any codeword whose ball is fully covered by the others."""
    order = list(range(len(chosen)))
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in order:
            if i >= len(chosen):
                continue
            b = balls[i]
            # temporarily remove i
            for idx in b:
                counts[idx] -= 1
            ok = bool(np.all(counts > 0))
            for idx in b:
                counts[idx] += 1
            if ok:
                for idx in b:
                    counts[idx] -= 1
                balls.pop(i)
                chosen.pop(i)
                order.pop(i)
                changed = True
                break
    return chosen, balls, counts


def _local_search(chosen, balls, counts, q, n, radius, inc, deadline, rng):
    total = q ** n
    best = len(chosen)
    # precompute candidate pool: a superset of codewords to consider for add/swap
    pool = list(chosen)
    seen = set(tuple(c) for c in pool)
    # add tetracode direct-sum words and some randoms to the pool
    for w in _tetracode_solution(q, n, radius):
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            pool.append(w)
    for _ in range(400):
        w = [rng.randrange(q) for _ in range(n)]
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            pool.append(w)
    pool_balls = [_ball_indices(w, n, q, radius, inc) for w in pool]
    # map from codeword tuple to pool index
    pool_map = {tuple(w): k for k, w in enumerate(pool)}

    no_improve = 0
    max_no_improve = 25000
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        # ---- REMOVE moves (full sweep, apply all safe removals) ----
        i = 0
        while i < len(chosen):
            b = balls[i]
            for idx in b:
                counts[idx] -= 1
            ok = bool(np.all(counts > 0))
            for idx in b:
                counts[idx] += 1
            if ok:
                for idx in b:
                    counts[idx] -= 1
                balls.pop(i)
                chosen.pop(i)
                improved = True
                best = min(best, len(chosen))
            else:
                i += 1
        if improved:
            no_improve = 0
            continue

        # ---- ADD / SWAP moves (randomized best-improvement) ----
        # sample a batch of candidate moves, take the best improving one
        batch = 60
        best_delta = 1  # we want delta < 1 (strict improvement in size)
        best_move = None
        for _ in range(batch):
            k = rng.randrange(len(pool))
            cand = pool[k]
            if tuple(cand) in seen and any(tuple(c) == tuple(cand) for c in chosen):
                continue
            cb = pool_balls[k]
            # remove-move (drop cand) if cand is in chosen
            if any(tuple(c) == tuple(cand) for c in chosen):
                ci = next(j for j, c in enumerate(chosen) if tuple(c) == tuple(cand))
                for idx in cb:
                    counts[idx] -= 1
                # does adding a random other word help? no -> this is just a removal
                # removal only helps if cand redundant (already handled above), skip
                for idx in cb:
                    counts[idx] += 1
                continue
            # ADD move: add cand, measure uncovered it fixes
            add_gain = 0
            for idx in cb:
                if counts[idx] == 0:
                    add_gain += 1
            if add_gain < 1:
                continue
            # SWAP move: add cand then try to remove one existing codeword
            for idx in cb:
                counts[idx] += 1
            # find an existing codeword that becomes redundant
            removed = -1
            for j in range(len(chosen)):
                bj = balls[j]
                for idx in bj:
                    counts[idx] -= 1
                ok = bool(np.all(counts > 0))
                for idx in bj:
                    counts[idx] += 1
                if ok:
                    removed = j
                    break
            # undo add
            for idx in cb:
                counts[idx] -= 1
            if removed >= 0:
                delta = 0  # add one, remove one -> size unchanged; only useful if it enables more removals
                # we only accept swaps that immediately allow removal (delta 0) is not a gain;
                # so require that after swap, a further removal is possible -> net -1
                # Simplify: accept swap if it lets us remove (size same now, but frees structure).
                # To keep strictly improving size, we instead accept only if removed exists AND
                # the net effect (we'll apply add+remove) keeps validity. That keeps size same.
                # So for a size *decrease* we need a pure removal (handled) or add-then-two-removes.
                # Keep it simple: apply the swap (size unchanged) which can unlock later removals.
                if delta < best_delta:
                    best_delta = delta
                    best_move = ('swap', k, removed)
            else:
                # pure ADD never decreases size; skip (we only minimize size)
                pass

        if best_move is not None:
            kind, k, removed = best_move
            cand = pool[k]
            cb = pool_balls[k]
            for idx in cb:
                counts[idx] += 1
            chosen.append(cand)
            balls.append(cb)
            # remove
            bj = balls[removed]
            for idx in bj:
                counts[idx] -= 1
            balls.pop(removed)
            chosen.pop(removed)
            improved = True
            no_improve = 0
        else:
            no_improve += 1
    return chosen, balls, counts


def construct(q, n, radius):
    start = time.time()
    deadline = start + 115.0  # safety margin under 120s (and 150s kill)
    rng = random.Random(12345)
    total = q ** n
    inc = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    try:
        # Phase 1: build a valid cover
        chosen, balls, counts = _greedy_build(q, n, radius, inc, deadline, rng)
        if not _is_cover(counts):
            # fallback: tetracode direct sum is always valid for q=3
            chosen = _tetracode_solution(q, n, radius)
            balls = [_ball_indices(cw, n, q, radius, inc) for cw in chosen]
            counts = np.zeros(total, dtype=np.int32)
            for b in balls:
                for idx in b:
                    counts[idx] += 1

        # Phase 2: prune
        chosen, balls, counts = _prune(chosen, balls, counts, deadline)

        # Phase 3: local search (add/swap/remove)
        chosen, balls, counts = _local_search(chosen, balls, counts, q, n, radius, inc, deadline, rng)

        # Final prune to be safe
        chosen, balls, counts = _prune(chosen, balls, counts, deadline)

        if _is_cover(counts) and len(chosen) > 0:
            # verify distinctness
            if len(set(tuple(c) for c in chosen)) == len(chosen):
                return chosen
    except Exception:
        pass

    # Fallback: guaranteed valid tetracode direct sum
    return _tetracode_solution(q, n, radius)
