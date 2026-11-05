# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 142.0  # safety margin under the 150 s wall-clock kill

def _ball_indices(word, q, n, radius):
    """Indices of all words within Hamming distance <= radius of `word`."""
    base = 0
    p = 1
    for i in range(n - 1, -1, -1):
        base += word[i] * p
        p *= q
    out = [base]
    if radius >= 1:
        for i in range(n):
            step = 1
            for j in range(n - 1, i, -1):
                step *= q
            for d in range(1, q):
                out.append(base + d * step)
    if radius >= 2:
        for i in range(n):
            si = 1
            for j in range(n - 1, i, -1):
                si *= q
            for k in range(i + 1, n):
                sk = 1
                for j in range(n - 1, k, -1):
                    sk *= q
                for d1 in range(1, q):
                    for d2 in range(1, q):
                        out.append(base + d1 * si + d2 * sk)
    return out


def _word_from_index(idx, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = idx % q
        idx //= q
    return w


def construct(q, n, radius):
    t0 = time.time()
    rng = random.Random(12345)
    total = q ** n
    R = radius

    # Precompute balls for every word (bounded: 6561*17 for the target case).
    balls = []
    for idx in range(total):
        balls.append(_ball_indices(_word_from_index(idx, q, n), q, n, R))

    def deadline():
        return time.time() - t0 > _DEADLINE

    # ---- Phase 0: strong structural seed (direct product with tetracode) ----
    if q == 3 and n >= 4 and R == 1:
        tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_words = []
        for tail in range(3 ** (n - 4)):
            tw = _word_from_index(tail, 3, n - 4)
            for h in tet:
                seed_words.append(h + tw)
    else:
        seed_words = []

    # ---- Phase 1: greedy set cover ----
    covered = [False] * total
    chosen = []
    for w in seed_words:
        idx = 0
        p = 1
        for i in range(n - 1, -1, -1):
            idx += w[i] * p
            p *= q
        for c in balls[idx]:
            if not covered[c]:
                covered[c] = True
        chosen.append(idx)

    # Greedy: repeatedly pick the word covering the most uncovered points.
    # Maintain, for each candidate word, its marginal gain.
    gain = [0] * total
    for w in chosen:
        for c in balls[w]:
            if not covered[c]:
                gain[w] += 1
    # For words not yet chosen, gain = full ball size.
    for w in range(total):
        if w not in chosen:
            gain[w] = len(balls[w])

    import heapq
    # max-heap on gain
    heap = [(-gain[w], w) for w in range(total)]
    heapq.heapify(heap)
    in_chosen = [False] * total
    for w in chosen:
        in_chosen[w] = True

    while True:
        if deadline():
            break
        # pop best valid
        while heap:
            negg, w = heapq.heappop(heap)
            if in_chosen[w]:
                continue
            if -negg != gain[w]:
                continue
            break
        else:
            break
        if gain[w] <= 0:
            break
        in_chosen[w] = True
        chosen.append(w)
        # update gains of neighbours (words sharing a covered point)
        for c in balls[w]:
            if covered[c]:
                continue
            covered[c] = True
            for w2 in balls[c]:
                if not in_chosen[w2] and gain[w2] > 0:
                    gain[w2] -= 1
                    heapq.heappush(heap, (-gain[w2], w2))
        if all(covered):
            break

    if all(covered):
        best = list(chosen)
    else:
        best = None

    # ---- Phase 2: iterative improvement (swap / drop / add) ----
    if best is not None:
        chosen_set = set(best)
        # recount coverage
        cov = [0] * total
        for w in best:
            for c in balls[w]:
                cov[c] += 1

        def coverage_ok():
            return all(x >= 1 for x in cov)

        improved = True
        while improved and not deadline():
            improved = False
            # try dropping a redundant word
            for w in list(best):
                if deadline():
                    break
                if cov[balls[w][0]] >= 2:  # quick: some point in its ball still covered
                    # check full redundancy
                    redundant = True
                    for c in balls[w]:
                        if cov[c] <= 1:
                            redundant = False
                            break
                    if redundant:
                        for c in balls[w]:
                            cov[c] -= 1
                        best.remove(w)
                        chosen_set.discard(w)
                        improved = True
                        break
            if improved:
                continue
            # try swap: remove w, add best word that covers the newly uncovered set
            uncovered = None
            for w in list(best):
                if deadline():
                    break
                # compute points that would become uncovered if w removed
                newly = [c for c in balls[w] if cov[c] == 1]
                if not newly:
                    # dropping w alone is safe (redundant) - already handled
                    continue
                # find candidate w2 maximizing coverage of `newly`
                best_w2 = -1
                best_gain = 0
                for w2 in chosen_set:
                    if w2 == w:
                        continue
                    g = 0
                    for c in newly:
                        if c in balls[w2]:
                            g += 1
                    if g > best_gain:
                        best_gain = g
                        best_w2 = w2
                if best_w2 != -1 and best_gain == len(newly):
                    # perform swap
                    for c in balls[w]:
                        cov[c] -= 1
                    for c in balls[best_w2]:
                        cov[c] += 1
                    best.remove(w)
                    chosen_set.discard(w)
                    best.append(best_w2)
                    chosen_set.add(best_w2)
                    improved = True
                    break

        if coverage_ok():
            best = list(best)
        else:
            best = None

    # ---- Phase 3: final safety greedy to guarantee coverage ----
    if best is None or not all(x >= 1 for x in cov) if best is not None else True:
        # rebuild coverage from scratch to be safe
        cov = [0] * total
        if best is not None:
            for w in best:
                for c in balls[w]:
                    cov[c] += 1
        still_needed = [c for c in range(total) if cov[c] == 0]
        if still_needed:
            # greedy add
            chosen_set = set(best) if best is not None else set()
            if best is None:
                best = []
            while still_needed and not deadline():
                best_w = -1
                best_g = 0
                for c in still_needed:
                    if deadline():
                        break
                    for w2 in balls[c]:
                        if w2 in chosen_set:
                            continue
                        # count how many of still_needed w2 covers
                        g = 0
                        for c2 in balls[w2]:
                            if cov[c2] == 0:
                                g += 1
                        if g > best_g:
                            best_g = g
                            best_w = w2
                    if best_w != -1:
                        break
                if best_w == -1 or best_g == 0:
                    break
                chosen_set.add(best_w)
                best.append(best_w)
                for c in balls[best_w]:
                    cov[c] += 1
                still_needed = [c for c in still_needed if cov[c] == 0]

    # Convert indices back to words
    return [_word_from_index(w, q, n) for w in best]
# EVOLVE-BLOCK-END
