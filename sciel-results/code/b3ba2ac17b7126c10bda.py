"""Fresh design: greedy independent-set init + fast local search for K_3(6,1)."""
import time
import numpy as np

_DEADLINE_FRAC = 0.90
_TIME_LIMIT = 150.0


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + _TIME_LIMIT * _DEADLINE_FRAC

    pow3 = [1]
    for _ in range(n):
        pow3.append(pow3[-1] * q)
    total = pow3[n]

    offsets = []
    for pos in range(n):
        p = pow3[pos]
        for d in range(1, q):
            offsets.append(d * p)
            offsets.append(-d * p)
    offsets = np.asarray(offsets, dtype=np.int64)
    n_off = len(offsets)

    # --- initial code: greedy independent set on distance-2 graph ---
    def within2(a, b):
        diff = 0
        x, y = a, b
        for _ in range(n):
            if (x % q) != (y % q):
                diff += 1
                if diff > 2:
                    return False
            x //= q
            y //= q
        return True

    code = []
    code_set = set()
    for w in range(total):
        ok = True
        for c in code:
            if within2(w, c):
                ok = False
                break
        if ok:
            code.append(w)
            code_set.add(w)

    best = list(code)
    best_size = len(code)

    cov = np.zeros(total, dtype=np.int32)
    for w in code:
        cov[w] += 1
        for off in offsets:
            cov[w + off] += 1

    def unique_count_at(cov, w):
        c = 1 if cov[w] == 1 else 0
        for off in offsets:
            if cov[w + off] == 1:
                c += 1
        return c

    def ball_unique(cov, w):
        # number of words in radius-1 ball of w with cov==1
        return unique_count_at(cov, w)

    def ball_zero(cov, w):
        c = 1 if cov[w] == 0 else 0
        for off in offsets:
            if cov[w + off] == 0:
                c += 1
        return c

    no_improve = 0
    while time.time() < deadline:
        improved = False

        # pass 1: removals
        for i in range(len(code)):
            w = code[i]
            if ball_unique(cov, w) == 0:
                code.pop(i)
                code_set.discard(w)
                cov[w] -= 1
                for off in offsets:
                    cov[w + off] -= 1
                improved = True
                break

        # pass 2: swaps
        if not improved:
            for i in range(len(code)):
                w = code[i]
                # current unique count contributed by w's ball (words only w covers)
                w_unique = ball_unique(cov, w)
                best_gain = 0
                best_new = None
                for off in offsets:
                    w2 = w + off
                    if w2 in code_set:
                        continue
                    # words that would become uniquely covered or newly covered by w2
                    # but not already covered by w (since w is leaving)
                    # Approximate gain: reduce total unique count.
                    # After swap, words in ball(w2) that are currently cov==0 become covered.
                    # Words in ball(w) that are currently cov==1 and not in ball(w2) become uncovered.
                    new_cover = ball_zero(cov, w2)
                    # subtract those in ball(w) too (overlap) - approximate by not subtracting
                    # since overlap is small; conservative: only count if not in ball(w)
                    # We accept if new_cover > w_unique (net reduction in unique count)
                    gain = new_cover - w_unique
                    if gain > best_gain:
                        best_gain = gain
                        best_new = w2
                if best_new is not None and best_gain > 0:
                    code[i] = best_new
                    code_set.discard(w)
                    code_set.add(best_new)
                    cov[w] -= 1
                    for off in offsets:
                        cov[w + off] -= 1
                    cov[best_new] += 1
                    for off in offsets:
                        cov[best_new + off] += 1
                    improved = True
                    break

        if len(code) < best_size:
            best = list(code)
            best_size = len(code)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 4:
                break

    result = []
    for w in best:
        word = []
        x = w
        for _ in range(n):
            word.append(x % q)
            x //= q
        result.append(word)
    return result
