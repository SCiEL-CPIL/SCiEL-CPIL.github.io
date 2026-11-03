"""Football pool / ternary covering code K_3(6,1) — submodular local search."""
import time
import random
import itertools

# ---------- index helpers ----------

def _word_to_idx(word, q):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx

def _idx_to_word(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return tuple(w)

# ---------- precompute balls & adjacency ----------

def _precompute(q, n, radius):
    """Return (balls, adj): balls[i]=list of indices within radius of i;
    adj[i]=list of indices at distance exactly radius (the 'ball center' candidates for i)."""
    total = q ** n
    balls = [None] * total
    adj = [None] * total
    for idx in range(total):
        w = _idx_to_word(idx, q, n)
        ball = [idx]
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig:
                    continue
                w[i] = v
                ball.append(_word_to_idx(w, q))
                w[i] = orig
        balls[idx] = ball
        adj[idx] = ball[1:]
    return balls, adj

# ---------- initial construction ----------

def _initial_construction(q, n, radius):
    if n < 4:
        return [list(itertools.product(range(q), repeat=n))]
    # ternary tetracode (9 words of length 4, radius-1 covering of 4-ary? actually of length-4 ternary)
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    result = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            result.append(tuple(h + list(tail)))
    return result

# ---------- coverage ----------

def _compute_coverage(codewords, balls, q):
    total = q ** 6 if False else len(balls)
    cover_count = [0] * total
    for cw in codewords:
        for w in balls[cw]:
            cover_count[w] += 1
    return cover_count

def _uncovered(cover_count):
    return [i for i, c in enumerate(cover_count) if c == 0]

# ---------- moves ----------

def _try_remove(codewords, cover_count, balls):
    """Remove a redundant codeword (every word in its ball has cover_count>=2)."""
    for i, cw in enumerate(codewords):
        redundant = True
        for w in balls[cw]:
            if cover_count[w] <= 1:
                redundant = False
                break
        if redundant:
            for w in balls[cw]:
                cover_count[w] -= 1
            new_codewords = codewords[:i] + codewords[i+1:]
            return new_codewords, cw
    return None

def _best_add(codewords, cover_count, balls, adj, candidates):
    """Among candidate centers, pick one maximizing number of currently-uncovered words covered."""
    best_cw = None
    best_gain = 0
    for cw in candidates:
        gain = 0
        for w in balls[cw]:
            if cover_count[w] == 0:
                gain += 1
        if gain > best_gain:
            best_gain = gain
            best_cw = cw
    return best_cw, best_gain

def _try_add(codewords, cover_count, balls, adj, uncovered, rng, deadline):
    if not uncovered:
        return None
    # candidate centers: uncovered words + their neighbors
    cand = set()
    for ui in uncovered:
        cand.add(ui)
        for nb in adj[ui]:
            cand.add(nb)
    if len(cand) > 300:
        cand = set(rng.sample(list(cand), 300))
    else:
        cand = list(cand)
    best_cw, best_gain = _best_add(codewords, cover_count, balls, adj, cand)
    if best_cw is not None and best_gain > 0:
        for w in balls[best_cw]:
            cover_count[w] += 1
        new_codewords = codewords + [best_cw]
        return new_codewords
    return None

def _try_swap(codewords, cover_count, balls, adj, uncovered, rng, deadline):
    """Submodular marginal-coverage swap: replace a codeword with a better center.
    A swap is valid if after removing cw and adding new_cw, no word becomes uncovered.
    Prefer swaps that also cover more uncovered words (helps future pruning)."""
    if not uncovered:
        return None
    n_sample = min(len(codewords), 30)
    sample_idx = rng.sample(range(len(codewords)), n_sample)
    for si in sample_idx:
        if time.time() > deadline:
            break
        cw = codewords[si]
        # words only covered by cw
        critical = [w for w in balls[cw] if cover_count[w] == 1]
        if not critical:
            continue
        # candidate replacements near critical words
        cand = set()
        for cw_word in critical[:6]:
            cand.add(cw_word)
            for nb in adj[cw_word]:
                cand.add(nb)
        if len(cand) > 80:
            cand = set(rng.sample(list(cand), 80))
        cand = list(cand)
        # simulate removal of cw
        for w in balls[cw]:
            cover_count[w] -= 1
        # find candidate that covers all critical words
        best_new = None
        best_gain = -1
        for new_cw in cand:
            if new_cw == cw:
                continue
            covers_all = True
            gain = 0
            for w in balls[new_cw]:
                if cover_count[w] == 0:
                    gain += 1
            for w in critical:
                if w not in balls[new_cw]:
                    covers_all = False
                    break
            if covers_all and gain > best_gain:
                best_gain = gain
                best_new = new_cw
        # restore removal
        for w in balls[cw]:
            cover_count[w] += 1
        if best_new is not None and best_new != cw:
            # commit swap: remove cw, add best_new
            for w in balls[cw]:
                cover_count[w] -= 1
            for w in balls[best_new]:
                cover_count[w] += 1
            new_codewords = codewords[:si] + [best_new] + codewords[si+1:]
            return new_codewords
    return None

# ---------- local search ----------

def _local_search(codewords, q, n, radius, deadline, rng):
    balls, adj = _precompute(q, n, radius)
    total = q ** n
    cover_count = _compute_coverage(codewords, balls, q)
    best_size = len(codewords)
    best_codewords = codewords[:]
    no_improve = 0
    max_no_improve = 80
    while time.time() < deadline:
        improved = False
        # prune redundant
        while time.time() < deadline:
            res = _try_remove(codewords, cover_count, balls)
            if res is None:
                break
            codewords, _ = res
            improved = True
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = codewords[:]
        # cover gaps
        unc = _uncovered(cover_count)
        if unc:
            while time.time() < deadline and unc:
                res = _try_add(codewords, cover_count, balls, adj, unc, rng, deadline)
                if res is None:
                    break
                codewords = res
                improved = True
                unc = _uncovered(cover_count)
            # prune again after adding
            while time.time() < deadline:
                res = _try_remove(codewords, cover_count, balls)
                if res is None:
                    break
                codewords, _ = res
                improved = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = codewords[:]
        # submodular swap to improve structure (may enable future removals)
        unc = _uncovered(cover_count)
        if not unc:
            res = _try_swap(codewords, cover_count, balls, adj, unc, rng, deadline)
            if res is not None:
                codewords = res
                improved = True
                # prune after swap
                while time.time() < deadline:
                    res2 = _try_remove(codewords, cover_count, balls)
                    if res2 is None:
                        break
                    codewords, _ = res2
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = codewords[:]
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        else:
            no_improve = 0
    return best_codewords

# ---------- entry point ----------

def construct(q, n, radius):
    start = time.time()
    deadline = start + 115  # safety margin before 120s (evaluator kills at 150s)
    rng = random.Random(12345)
    codewords = _initial_construction(q, n, radius)
    codewords = _local_search(codewords, q, n, radius, deadline, rng)
    seen = set()
    valid = []
    for cw in codewords:
        if cw not in seen:
            seen.add(cw)
            valid.append(cw)
    return [list(cw) for cw in valid]
