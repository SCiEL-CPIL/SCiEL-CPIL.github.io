"""Football pool K_3(7,1): greedy max-coverage + local search."""
# EVOLVE-BLOCK-START
import time
import heapq
import itertools
import numpy as np

_DEADLINE = 150.0
_TIME_MARGIN = 15.0


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + _DEADLINE - _TIME_MARGIN

    if n == 7 and q == 3 and radius == 1:
        return _solve_71()

    # Generic fallback: tetracode direct-sum when applicable
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(3) for b in range(3)]
        out = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=n - 4):
                out.append(h + list(tail))
        return out
    # Brute-force fallback (tiny spaces only)
    all_words = list(itertools.product(range(q), repeat=n))
    return [list(w) for w in all_words]


def _solve_71():
    t0 = time.time()
    deadline = t0 + _DEADLINE - _TIME_MARGIN

    # Precompute all 2187 words and their balls (word + 21 neighbors)
    words = np.array(list(itertools.product(range(3), repeat=7)),
                     dtype=np.int64)  # (2187, 7)
    N = 2187
    ball_size = 1 + 7 * 2  # 15

    # Build ball index array: for each word, list of word indices in its radius-1 ball
    ball_idx = np.empty((N, ball_size), dtype=np.int64)
    for i in range(N):
        w = words[i]
        j = 0
        ball_idx[i, j] = i
        j += 1
        for pos in range(7):
            base = w[pos]
            for delta in (1, 2):
                v = (base + delta) % 3
                # index = sum w[k] * 3^k ; changing pos by delta changes by delta*3^pos
                ball_idx[i, j] = i + delta * (3 ** pos)
                j += 1
    # ball_mask: (N, N) bool -- word i is covered by codeword c iff ball_mask[c, i]
    ball_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        ball_mask[ball_idx[i], i] = True

    # Greedy max-coverage (lazy evaluation)
    covered = np.zeros(N, dtype=bool)
    chosen = []
    chosen_set = set()
    # heap entries: (-gain, seq, codeword)
    seq = 0
    heap = []
    # Initial gains = 15 for all
    for c in range(N):
        heapq.heappush(heap, (-ball_size, seq, c))
        seq += 1

    while len(chosen) < N:
        if time.time() > deadline:
            break
        # Pop until valid
        neg_gain, _, c = heapq.heappop(heap)
        if c in chosen_set:
            continue
        # Compute current gain
        gain = int(np.sum(~covered[ball_mask[c]]))
        if gain <= 0:
            continue
        chosen.append(c)
        chosen_set.add(c)
        covered[ball_mask[c]] = True
        # Push updated gains for codewords not yet chosen (lazy: push all, filter on pop)
        for c2 in range(N):
            if c2 in chosen_set:
                continue
            g2 = int(np.sum(~covered[ball_mask[c2]]))
            if g2 > 0:
                heapq.heappush(heap, (-g2, seq, c2))
                seq += 1

    # Local search: removal + swap
    if time.time() > deadline:
        return _finalize(chosen)

    # Removal pass (iterate until stable)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(chosen) - 1, -1, -1):
            c = chosen[i]
            # words only covered by c
            # covered_by_c = ball_mask[c]
            # others cover: union of ball_mask[j] for j != i
            # simpler: check if any word in c's ball is not covered by others
            others = [j for j in range(len(chosen)) if j != i]
            if not others:
                # only one left, can't remove
                continue
            # compute union of others' balls
            union = np.zeros(N, dtype=bool)
            for j in others:
                union |= ball_mask[chosen[j]]
            if union[ball_mask[c]].all():
                chosen.pop(i)
                chosen_set.discard(c)
                changed = True
                break

    # Swap pass: try moving each codeword to a better position
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(chosen)):
            c = chosen[i]
            # For each candidate new position nc, check coverage without c but with nc
            # gain: does nc cover words that c covers but others don't? and does it keep coverage?
            # Simpler: try all nc, count coverage; accept if full coverage maintained
            # To be efficient, only try nc that cover the "unique" words of c
            others = [chosen[j] for j in range(len(chosen)) if j != i]
            union_others = np.zeros(N, dtype=bool)
            for j in others:
                union_others |= ball_mask[chosen[j]]
            # words that must be covered by c (not covered by others)
            must = ball_mask[c] & ~union_others
            if not must.any():
                # c is redundant; try removing
                chosen.pop(i)
                chosen_set.discard(c)
                improved = True
                break
            # Find nc that covers all 'must' words and is not already chosen
            # nc must be in intersection of balls of all 'must' words
            # i.e., nc within distance 1 of each must word
            # Compute candidate nc set
            must_words = np.where(must)[0]
            candidates = set(range(N))
            for mw in must_words:
                candidates &= set(np.where(ball_mask[:, mw])[0].tolist())
            # try to find an nc in candidates not in chosen_set that covers must
            # pick the one that maximizes additional coverage (optional)
            # For simplicity, pick first valid; but we want to keep count same
            # Actually swapping keeps count; we need nc that covers must AND the rest covered
            # Since nc covers must and others cover their part, need union(nc, others) == all
            # Check: are all words covered by union_others or nc?
            # We need: for every word w, w in union_others or w in ball_mask[nc]
            # = ~union_others subset of ball_mask[nc]
            uncovered = ~union_others
            best_nc = None
            for nc in candidates:
                if nc in chosen_set:
                    continue
                if ball_mask[nc][uncovered].all():
                    best_nc = nc
                    break
            if best_nc is not None:
                chosen[i] = best_nc
                chosen_set.discard(c)
                chosen_set.add(best_nc)
                improved = True
                break

    # Final removal sweep
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(chosen) - 1, -1, -1):
            c = chosen[i]
            others = [chosen[j] for j in range(len(chosen)) if j != i]
            if not others:
                continue
            union = np.zeros(N, dtype=bool)
            for j in others:
                union |= ball_mask[chosen[j]]
            if union[ball_mask[c]].all():
                chosen.pop(i)
                chosen_set.discard(c)
                changed = True
                break

    return _finalize(chosen)


def _finalize(chosen):
    # Deduplicate and convert to lists
    seen = set()
    out = []
    for c in chosen:
        tup = tuple(c)
        if tup not in seen:
            seen.add(tup)
            out.append(list(c))
    return out
# EVOLVE-BLOCK-END
