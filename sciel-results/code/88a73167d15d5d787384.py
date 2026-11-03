"""Ternary covering code K_3(6,1): greedy packing + local search (remove/swap)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

_DEADLINE_MARGIN = 8.0  # seconds of safety before the 150s kill


def _all_words(q, n):
    return [list(w) for w in itertools.product(range(q), repeat=n)]


def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def _neighbors(w, q, n):
    """All words at Hamming distance <= 1 from w (the ball of radius 1)."""
    ball = [w]
    for i in range(n):
        for v in range(q):
            if v == w[i]:
                continue
            nb = w[:]
            nb[i] = v
            ball.append(nb)
    return ball


def construct(q, n, radius):
    start = time.time()
    deadline = start + 150.0 - _DEADLINE_MARGIN

    def now():
        return time.time()

    def time_ok():
        return now() < deadline

    # ----- Build the full universe and distance-1 ball for every word -----
    words = _all_words(q, n)
    W = len(words)  # 729 for q=3,n=6
    idx = {w: i for i, w in enumerate(words)}

    # Ball of radius 1 around each word, stored as a list of indices.
    ball = []
    for w in words:
        ball.append([idx[x] for x in _neighbors(w, q, n)])

    # ----- Greedy packing: maximal set with pairwise distance >= 3 -----
    # This guarantees no two chosen words are within distance 2, so none is
    # "hugely redundant" and the cover is well spread.
    rng = random.Random(20240607)
    order = list(range(W))
    rng.shuffle(order)

    chosen = []
    chosen_set = set()
    for wi in order:
        if not time_ok():
            break
        w = words[wi]
        ok = True
        for c in chosen:
            if _hamming(w, c) < 3:  # must be distance >= 3 from all chosen
                ok = False
                break
        if ok:
            chosen.append(w)
            chosen_set.add(wi)

    # ----- Compute uncovered set from the current code -----
    covered = [False] * W
    for c in chosen:
        for j in ball[idx[c]]:
            covered[j] = True
    uncovered = [j for j in range(W) if not covered[j]]
    uncovered_set = set(uncovered)
    best = list(chosen)
    best_unc = len(uncovered)

    # ----- Local search loop -----
    # Removal: drop any codeword whose ball is fully covered by the others.
    # Swap: drop one codeword, add one (from a candidate pool) that covers the
    # most of the resulting uncovered set.
    candidate_pool = list(range(W))

    progress = True
    while progress and time_ok():
        progress = False

        # --- Removal pass ---
        for pos in range(len(chosen) - 1, -1, -1):
            if not time_ok():
                break
            c = chosen[pos]
            ci = idx[c]
            # count how many words in c's ball are uncovered by the rest
            redundant = True
            for j in ball[ci]:
                if not covered[j]:
                    redundant = False
                    break
            if redundant:
                # remove c: its ball was fully covered, so covered stays valid
                chosen.pop(pos)
                progress = True
                if len(chosen) < len(best):
                    best = list(chosen)
                    best_unc = len(uncovered)

        if not time_ok():
            break

        # --- Recompute uncovered after removals ---
        covered = [False] * W
        for c in chosen:
            for j in ball[idx[c]]:
                covered[j] = True
        uncovered_set = set(j for j in range(W) if not covered[j])
        best_unc = len(uncovered_set)

        # --- Swap pass ---
        improved_swap = True
        while improved_swap and time_ok():
            improved_swap = False
            # For each candidate removal, find best addition.
            # We try removals in an order that prioritizes those that free up
            # the most uncovered words (i.e., whose ball has many uncovered).
            rem_order = sorted(
                range(len(chosen)),
                key=lambda p: -sum(1 for j in ball[idx[chosen[p]]] if j in uncovered_set),
            )
            for pos in rem_order:
                if not time_ok():
                    break
                c = chosen[pos]
                ci = idx[c]
                # uncovered if we remove c
                new_unc = set(uncovered_set)
                for j in ball[ci]:
                    if j not in covered[j] or True:
                        pass
                # recompute which words become uncovered when c is removed
                # (a word j becomes uncovered if it was only covered by c)
                # cheaper: for each j in ball[ci], check if covered by another chosen
                lost = []
                for j in ball[ci]:
                    # is j covered by some other chosen codeword?
                    alt = False
                    for k in range(len(chosen)):
                        if k == pos:
                            continue
                        cj = idx[chosen[k]]
                        # j covered by chosen[k] iff _hamming(words[j], chosen[k])<=1
                        if j in ball[cj]:
                            alt = True
                            break
                    if not alt:
                        lost.append(j)
                if not lost:
                    continue
                # find best addition covering most of lost
                best_gain = 0
                best_add = None
                for a in candidate_pool:
                    ai = a
                    gain = 0
                    for j in ball[ai]:
                        if j in lost:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_add = ai
                if best_add is not None and best_gain >= len(lost):
                    # swap improves or ties uncovered count but same size;
                    # only accept if it reduces uncovered (gain==len(lost))
                    # and keep size same -> accept to enable further reductions
                    chosen[pos] = words[best_add]
                    uncovered_set = set()
                    covered = [False] * W
                    for cc in chosen:
                        for j in ball[idx[cc]]:
                            covered[j] = True
                    uncovered_set = set(j for j in range(W) if not covered[j])
                    best_unc = len(uncovered_set)
                    improved_swap = True
                    if len(chosen) < len(best) or (len(chosen) == len(best) and best_unc < len(best_unc)):
                        best = list(chosen)
                        best_unc = best_unc
                    break  # restart removal ordering after a swap

        if len(chosen) < len(best) or (len(chosen) == len(best) and len(uncovered_set) < best_unc):
            best = list(chosen)
            best_unc = len(uncovered_set)

    # ----- Final: ensure a valid cover -----
    # If best is not a full cover (best_unc>0), fall back to a guaranteed cover.
    # We always have a valid cover available: the greedy code itself is a cover
    # only if it's maximal w.r.t. distance>=3, which it is, but that does NOT
    # guarantee radius-1 coverage. So we must verify and, if needed, add words.
    def is_cover(code):
        cov = [False] * W
        for c in code:
            for j in ball[idx[c]]:
                cov[j] = True
        return all(cov)

    # Prefer the smallest valid code among candidates we've seen.
    candidates = [best]
    if not is_cover(best):
        # build a guaranteed cover from scratch via greedy set cover
        cov = [False] * W
        cover_code = []
        remaining = set(range(W))
        while remaining and time_ok():
            best_word = None
            best_gain = 0
            for a in candidate_pool:
                gain = 0
                for j in ball[a]:
                    if j in remaining:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_word = a
            if best_word is None or best_gain == 0:
                break
            cover_code.append(words[best_word])
            for j in ball[best_word]:
                if j in remaining:
                    remaining.discard(j)
        candidates.append(cover_code)

    # pick the smallest valid cover
    valid = [c for c in candidates if is_cover(c)]
    if not valid:
        # ultimate fallback: direct-sum construction (always valid for n>=4)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        valid = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    valid.sort(key=len)
    result = valid[0]

    # de-duplicate just in case
    seen = set()
    out = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            out.append(list(w))
    return out
# EVOLVE-BLOCK-END
