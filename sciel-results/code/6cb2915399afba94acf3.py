# EVOLVE-BLOCK-START
import time
import itertools

_T1 = time.time()
_HARD = _T1 + 150.0


def _deadline(margin):
    return (time.time() + margin) < _HARD


def _build_neighbors(q, n):
    """Return (all_words, word_idx, nrm) where nrm[i] = list of neighbor indices
    of word i (self + all single-coordinate flips)."""
    all_words = list(itertools.product(range(q), repeat=n))
    word_idx = {w: i for i, w in enumerate(all_words)}
    nrm = []
    for w in all_words:
        nbrs = [word_idx[w]]
        for j in range(n):
            for d in (1, 2):
                b = list(w)
                b[j] = (b[j] + d) % q
                nbrs.append(word_idx[tuple(b)])
        nrm.append(nbrs)
    return all_words, word_idx, nrm


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _greedy_base_code(q, m, radius, budget):
    """Greedy set cover over {0..q-1}^m to find a small covering code of radius `radius`.
    Returns list of codewords (tuples)."""
    N = q ** m
    all_words, word_idx, nrm = _build_neighbors(q, m)
    uncovered = set(range(N))
    sel = []
    end = time.time() + budget
    while uncovered and time.time() < end:
        best_word = -1
        best_count = 0
        for w in range(N):
            cnt = 0
            for nb in nrm[w]:
                if nb in uncovered:
                    cnt += 1
                    if cnt <= best_count:
                        break
            if cnt > best_count:
                best_count = cnt
                best_word = w
                if best_count == len(nrm[0]):
                    break
        if best_word < 0:
            break
        sel.append(best_word)
        for nb in nrm[best_word]:
            uncovered.discard(nb)
    return sel


def _fallback_base(q, n, radius):
    """Direct-sum of tetracode: 9 * q^(n-4) words, radius-1 for n>=4, q=3."""
    if q != 3 or n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    tetracode = _tetracode()
    tail_len = n - 4
    out = []
    for tail in itertools.product(range(q), repeat=tail_len):
        for h in tetracode:
            out.append(h + list(tail))
    return out


def _direct_sum_base(q, n, radius):
    """Structural: build a small base covering code on m coordinates, then direct-sum
    with the full space on the remaining n-m coordinates -> M * q^(n-m) codewords."""
    if q != 3 or n < 2:
        return _fallback_base(q, n, radius)
    # Try several (m, base_size) configurations; pick the smallest total.
    # For q=3: tetracode gives 9 on 4 coords. Greedy can give smaller on fewer coords.
    candidates = []
    # tetracode direct sum
    if n >= 4:
        candidates.append(9 * q ** (n - 4))
    # greedy base codes on m coordinates
    for m in (2, 3, 4, 5):
        if m > n:
            continue
        budget = min(4.0, max(0.5, 60.0 / m))
        base = _greedy_base_code(q, m, radius, budget)
        if base:
            candidates.append(len(base) * q ** (n - m))
    # Also full space (worst case)
    candidates.append(q ** n)
    best_total = min(candidates)

    # Now rebuild the best one
    if best_total == q ** n:
        return _fallback_base(q, n, radius)
    if n >= 4 and best_total == 9 * q ** (n - 4):
        return _fallback_base(q, n, radius)
    # pick the m that gave the best total
    best_m = None
    for m in (2, 3, 4, 5):
        if m > n:
            continue
        if len(_greedy_base_code(q, m, radius, 0.1) or []) * q ** (n - m) == best_total:
            best_m = m
            break
    if best_m is None:
        return _fallback_base(q, n, radius)
    base = _greedy_base_code(q, best_m, radius, 2.0)
    tail_len = n - best_m
    out = []
    for tail in itertools.product(range(q), repeat=tail_len):
        for h in base:
            out.append(list(h) + list(tail))
    return out


def _improve_remove_repair(base, q, n, radius):
    """Time-guarded remove-then-repair: shrink the code while keeping it a valid cover."""
    if radius < 1:
        return base
    N = q ** n
    all_words, word_idx, nrm = _build_neighbors(q, n)

    sel = []
    seen = set()
    for cw in base:
        i = word_idx[tuple(cw)]
        if i not in seen:
            seen.add(i)
            sel.append(i)

    cov = [0] * N
    for ci in sel:
        for nb in nrm[ci]:
            cov[nb] += 1

    sel_set = set(sel)
    deadline_guard = _HARD - 8.0
    improved = True
    while improved and time.time() < deadline_guard:
        improved = False
        for ci in sel:
            uncover_list = []
            for nb in nrm[ci]:
                if cov[nb] == 1:
                    uncover_list.append(nb)
                    if len(uncover_list) > 2:
                        break
            if len(uncover_list) <= 2:
                for nb in nrm[ci]:
                    cov[nb] -= 1
                sel.remove(ci)
                sel_set.discard(ci)
                improved = True
                for uc in uncover_list:
                    if cov[uc] == 0:
                        best = uc
                        best_gain = 0
                        for cand in nrm[uc]:
                            gain = 0
                            for nb2 in nrm[cand]:
                                if cov[nb2] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best = cand
                        if best not in sel_set:
                            sel.append(best)
                            sel_set.add(best)
                            for nb in nrm[best]:
                                cov[nb] += 1
                if not _deadline(8.0):
                    break
    return [list(all_words[i]) for i in sel]


def construct(q, n, radius):
    global _HARD
    _HARD = time.time() + 150.0
    if n < 0:
        return []
    if n == 0:
        return [[]]

    # --- Build a guaranteed-valid base code ---
    # Use direct-sum with a small base code (structural improvement over full tetracode)
    base = _direct_sum_base(q, n, radius)

    # For q=3, n=8, radius=1: also try a pure greedy set cover as an alternative base
    if q == 3 and n == 8 and radius == 1 and _deadline(80.0):
        N = 3 ** n
        all_words, word_idx, nrm = _build_neighbors(q, n)
        uncovered = set(range(N))
        greedy_sel = []
        deadline_g = _HARD - 12.0
        while uncovered and time.time() < deadline_g:
            best_word = -1
            best_count = 0
            for w in range(N):
                cnt = 0
                for nb in nrm[w]:
                    if nb in uncovered:
                        cnt += 1
                        if cnt <= best_count:
                            break
                if cnt > best_count:
                    best_count = cnt
                    best_word = w
                    if best_count == len(nrm[0]):
                        break
            if best_word < 0:
                break
            greedy_sel.append(best_word)
            for nb in nrm[best_word]:
                uncovered.discard(nb)
        if not uncovered and len(greedy_sel) < len(base):
            base = [list(all_words[i]) for i in greedy_sel]

    # --- Guarantee validity: cover all words (repair if base was not perfect) ---
    if radius >= 1:
        N = q ** n
        all_words, word_idx, nrm = _build_neighbors(q, n)
        cov = [0] * N
        sel = []
        seen = set()
        for cw in base:
            i = word_idx[tuple(cw)]
            if i not in seen:
                seen.add(i)
                sel.append(i)
                for nb in nrm[i]:
                    cov[nb] += 1
        uncovered = [i for i in range(N) if cov[i] == 0]
        while uncovered and _deadline(25.0):
            uc = uncovered.pop()
            if uc not in seen:
                seen.add(uc)
                sel.append(uc)
                for nb in nrm[uc]:
                    cov[nb] += 1
                if uncovered:
                    uncovered = [i for i in uncovered if cov[i] == 0]
        base = [list(all_words[i]) for i in sel]

        # --- Optional improvement (time-guarded) ---
        if _deadline(50.0) and len(base) <= 20000:
            base = _improve_remove_repair(base, q, n, radius)

    return [list(c) for c in base]
# EVOLVE-BLOCK-END
