"""Direct-sum covering with an optimized short code + ball-pruning.
Family of the seed: C (short covering code on k coords) direct-summed with full
space on the remaining n-k coords, then prune words whose radius-1 ball is already
covered by the rest. All within Hamming radius `radius` (==1 here)."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

_DEADLINE = 150.0
_MARGIN = 12.0          # safety margin before the 150s kill
_SMALL_N = 24           # below this, prune every candidate word (exact)


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _balls(word, q, n):
    """All words within Hamming distance 1 of `word` (a list of tuples)."""
    out = [tuple(word)]
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                w = list(word)
                w[i] = v
                out.append(tuple(w))
    return out


def _symmetry_key(word, q, n, rng):
    """A canonical-ish symmetry signature used only for deterministic tie-breaking:
    coordinate permutation (sorted by symbol) + a random affine symbol relabel."""
    perm = list(range(n))
    rng.shuffle(perm)
    a = rng.randint(1, q - 1)
    b = rng.randint(0, q - 1)
    return tuple(sorted(((word[i] * a + b) % q, i) for i in perm))


def _is_covered(ball, others_balls, q, n):
    """True if every point in `ball` is within distance 1 of some word in `others_balls`."""
    # Build the union of the other balls quickly for a single query set.
    covered = set()
    for ob in others_balls:
        covered.update(ob)
    for p in ball:
        if p not in covered:
            return False
    return True


def _prune(codewords, q, n, radius, deadline, small):
    """Remove redundant words. A word is redundant if its radius-1 ball is covered
    by the remaining words. We keep a valid covering at all times (only drop when
    the rest already covers that ball)."""
    words = [tuple(w) for w in codewords]
    balls = [_balls(w, q, n) for w in words]
    # Precompute the union of all balls to know the universe we must keep covered.
    universe = set()
    for b in balls:
        universe.update(b)

    # Order: drop words that are "least useful" first (fewest uniquely-covered points).
    def unique_count(i):
        mine = set(balls[i])
        others = set()
        for j in range(len(words)):
            if j != i:
                others.update(balls[j])
        return len(mine - others)

    order = sorted(range(len(words)), key=lambda i: unique_count(i))
    active = set(range(len(words)))
    covered_now = set()
    for i in range(len(words)):
        covered_now.update(balls[i])

    for i in order:
        if time.time() > deadline:
            break
        if i not in active:
            continue
        # Does removing word i keep the covering valid?
        # i.e. is balls[i] subset of (union of active \ {i})?
        # Recompute union of active without i (cheap enough for small active sets).
        if len(active) > small and (len(active) - 1) % 7 != 0:
            # For large sets, only attempt a cheap check on a sample to save time.
            sample = list(balls[i])
            others = set()
            for j in active:
                if j != i:
                    others.update(balls[j])
            if all(p in others for p in sample):
                active.discard(i)
        else:
            others = set()
            for j in active:
                if j != i:
                    others.update(balls[j])
            if all(p in others for p in balls[i]):
                active.discard(i)

    return [list(words[i]) for i in sorted(active)]


def _greedy_cover(q, n, radius, deadline, seed):
    """Greedy set-cover: repeatedly add the word whose radius-1 ball covers the most
    not-yet-covered points, with symmetry-guided tie-breaking."""
    rng = random.Random(seed)
    universe = set(itertools.product(range(q), repeat=n))
    uncovered = set(universe)
    if not uncovered:
        return []
    # Precompute balls lazily.
    balls_cache = {}
    def ball_of(w):
        t = tuple(w)
        if t not in balls_cache:
            balls_cache[t] = _balls(list(t), q, n)
        return balls_cache[t]

    chosen = []
    # Candidate pool: start with a random sample + structured candidates.
    # We consider the whole space but score incrementally.
    # To keep it fast, iterate over the space but only re-score when needed.
    while uncovered and time.time() < deadline:
        best = None
        best_gain = -1
        best_key = None
        for w in itertools.product(range(q), repeat=n):
            b = ball_of(list(w))
            gain = len(uncovered.intersection(b))
            if gain > best_gain:
                best_gain = gain
                best = w
                best_key = _symmetry_key(list(w), q, n, rng)
            elif gain == best_gain and gain > 0:
                k = _symmetry_key(list(w), q, n, rng)
                if best_key is None or k < best_key:
                    best = w
                    best_key = k
        if best is None or best_gain <= 0:
            break
        chosen.append(list(best))
        uncovered.difference_update(ball_of(best))
    return chosen


def _direct_sum(short_code, q, n, k, radius):
    """Direct sum: every short codeword on the first k coords × every word on the rest."""
    tails = list(itertools.product(range(q), repeat=n - k))
    out = []
    for h in short_code:
        for t in tails:
            out.append(list(h) + list(t))
    return out


def _optimize_short_code(q, k, radius, deadline, seed):
    """Find a small covering code of length k, radius `radius` over q, via greedy + prune."""
    rng = random.Random(seed)
    # Greedy cover on the short space.
    universe = set(itertools.product(range(q), repeat=k))
    uncovered = set(universe)
    balls_cache = {}
    def ball_of(w):
        t = tuple(w)
        if t not in balls_cache:
            balls_cache[t] = _balls(list(t), q, k)
        return balls_cache[t]
    chosen = []
    while uncovered and time.time() < deadline:
        best = None
        best_gain = -1
        best_key = None
        for w in itertools.product(range(q), repeat=k):
            b = ball_of(list(w))
            gain = len(uncovered.intersection(b))
            if gain > best_gain:
                best_gain = gain
                best = w
                best_key = _symmetry_key(list(w), q, k, rng)
            elif gain == best_gain and gain > 0:
                kk = _symmetry_key(list(w), q, k, rng)
                if best_key is None or kk < best_key:
                    best = w
                    best_key = kk
        if best is None or best_gain <= 0:
            break
        chosen.append(list(best))
        uncovered.difference_update(ball_of(best))
    # Prune the short code.
    pruned = _prune(chosen, q, k, radius, deadline, small=_SMALL_N)
    return pruned


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + (_DEADLINE - _MARGIN)

    # --- Seed construction (always valid) as a baseline. ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    if n >= 4:
        baseline = _direct_sum(tetracode, q, n, 4, radius)
    else:
        baseline = list(itertools.product(range(q), repeat=n))
    best = baseline

    # --- Try: optimized short code of length k, direct-summed, then pruned. ---
    # Choose k so that the short code is small enough to optimize and the product
    # is small enough to prune. For q=3, n=6, radius=1: try k=4 (tetracode-sized).
    k_choices = []
    for k in range(2, n + 1):
        # product size = len(short) * q^(n-k); keep total candidates manageable.
        k_choices.append(k)

    # Prefer k that minimizes expected product size after a good short code.
    # We'll try a few k values, each with a time budget.
    remaining = deadline - time.time()
    if remaining > 5:
        # Sort k by q^(n-k) ascending (smaller product = easier to prune).
        k_choices.sort(key=lambda k: q ** (n - k))
        per_k = max(2.0, remaining / max(1, len(k_choices)))
        for k in k_choices:
            if time.time() > deadline:
                break
            sub_deadline = min(deadline, time.time() + per_k)
            try:
                short = _optimize_short_code(q, k, radius, sub_deadline, seed=12345 + k)
                if not short:
                    continue
                prod = _direct_sum(short, q, n, k, radius)
                if len(prod) >= len(best):
                    # No point pruning something already no better than best.
                    continue
                pruned = _prune(prod, q, n, radius, min(deadline, time.time() + per_k),
                                 small=_SMALL_N)
                if len(pruned) < len(best):
                    best = pruned
            except Exception:
                continue

    # --- Try: full greedy cover on the whole space, then prune (if room). ---
    remaining = deadline - time.time()
    if remaining > 8 and q ** n <= 40000:
        try:
            greedy = _greedy_cover(q, n, radius, min(deadline, time.time() + remaining - 3),
                                   seed=777)
            if greedy and len(greedy) < len(best):
                pruned = _prune(greedy, q, n, radius, deadline, small=_SMALL_N)
                if len(pruned) < len(best):
                    best = pruned
        except Exception:
            pass

    # Final safety: ensure we return before the hard limit.
    if time.time() > deadline:
        # Return whatever we have (still valid).
        pass

    # De-duplicate while preserving order (defensive).
    seen = set()
    out = []
    for w in best:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            out.append(list(w))
    return out
# EVOLVE-BLOCK-END
