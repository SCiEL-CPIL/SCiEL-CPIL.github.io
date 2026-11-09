"""Greedy set-covering for the ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import numpy as np


def _base3_codes(n, q):
    """All q^n words as integer codes (base-q, position i in the least significant slot)."""
    return list(range(q ** n))


def _code_to_vec(code, n, q):
    v = np.empty(n, dtype=np.int64)
    for i in range(n):
        v[i] = code % q
        code //= q
    return v


def _ball(code, n, q, radius):
    """All words within Hamming distance <= radius of `code`, as a set of integer codes."""
    vec = _code_to_vec(code, n, q)
    out = {code}
    if radius >= 1:
        for i in range(n):
            orig = vec[i]
            for d in range(1, q):
                new = (orig + d) % q
                if new == orig:
                    continue
                vec[i] = new
                # recompute code
                c = code
                for j in range(n):
                    c = c * q + vec[j]
                out.add(c)
                vec[i] = orig
    if radius >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = vec[i], vec[j]
                for di in range(1, q):
                    for dj in range(1, q):
                        ni = (oi + di) % q
                        nj = (oj + dj) % q
                        if ni == oi or nj == oj:
                            continue
                        vec[i], vec[j] = ni, nj
                        c = code
                        for k in range(n):
                            c = c * q + vec[k]
                        out.add(c)
                        vec[i], vec[j] = oi, oj
    return out


def _all_balls(codes, n, q, radius):
    return [_ball(c, n, q, radius) for c in codes]


def construct(q, n, radius):
    total = q ** n
    all_codes = _base3_codes(n, q)
    all_balls = _all_balls(all_codes, n, q, radius)

    # Greedy: start with nothing covered, repeatedly pick the code whose ball
    # covers the most currently-uncovered words.
    covered = np.zeros(total, dtype=bool)
    chosen = []
    remaining = list(range(total))

    # Precompute ball sizes for tie-breaking / ordering
    ball_sizes = [len(b) for b in all_balls]

    while not covered.all():
        best_idx = -1
        best_gain = -1
        # iterate over remaining candidates
        for idx in remaining:
            b = all_balls[idx]
            # gain = number of not-yet-covered words in this ball
            # convert ball (set of codes) to boolean and count uncovered
            # Efficient: iterate the set
            gain = 0
            for w in b:
                if not covered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                if gain == len(b):
                    break
        if best_idx == -1:
            break
        chosen.append(best_idx)
        for w in all_balls[best_idx]:
            covered[w] = True
        remaining.remove(best_idx)
        if not remaining:
            break

    # Convert chosen codes to vector form
    result = [_code_to_vec(c, n, q).tolist() for c in chosen]
    return result
# EVOLVE-BLOCK-END
