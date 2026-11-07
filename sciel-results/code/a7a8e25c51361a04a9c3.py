"""Rewritten program: DCT-basis shape search (DE + coordinate descent)."""
import numpy as np
import time

N = 50
DEADLINE = 125.0  # safety margin well under the 150s kill


def _evaluate(h):
    """Exact task score for nonnegative heights h (length N)."""
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m + 1] = g
    w = 1.0 / (m + 1)
    l2 = (w / 3.0) * np.sum(y[:m + 1] ** 2 + y[:m + 1] * y[1:m + 2] + y[1:m + 2] ** 2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


# DCT-II basis vectors (orthonormal) for N points.
_B = np.zeros((N, N))
for k in range(N):
    _B[k] = np.cos(np.pi * (np.arange(N) + 0.5) * k / N)
_B = _B / np.sqrt(N)


def _c2h(c):
    """Map DCT coefficients to nonnegative, max-1 heights."""
    h = _B.T @ c
    h = np.maximum(h, 0.0)
    mx = h.max()
    if mx <= 0:
        return np.zeros(N)
    return h / mx


def construct():
    start = time.time()

    def elapsed():
        return time.time() - start

    rng = np.random.default_rng(12345)

    def rand_c(scale):
        return rng.normal(0.0, scale, size=N)

    def score_c(c):
        return _evaluate(_c2h(c))

    # ---- seed (uniform) ----
    c0 = np.zeros(N)
    c0[0] = 1.0
    best_c = c0.copy()
    best_h = _c2h(best_c)
    best_s = _evaluate(best_h)

    # ---- quick local polish (coordinate descent in DCT space) ----
    def polish(c, iters, base_step):
        c = c.copy()
        s = score_c(c)
        for _ in range(iters):
            if elapsed() > DEADLINE:
                break
            for k in range(N):
                if elapsed() > DEADLINE:
                    break
                d = 0.015 * base_step
                for _ in range(6):
                    c[k] += d
                    s1 = score_c(c)
                    c[k] -= 2 * d
                    s2 = score_c(c)
                    c[k] += d
                    if s1 >= s2:
                        if s1 > s:
                            c[k] += d
                            s = s1
                        d *= 0.5
                    else:
                        if s2 > s:
                            c[k] -= d
                            s = s2
                        d *= 0.5
        return c, s

    best_c, best_s = polish(best_c, 6, 1.0)
    best_h = _c2h(best_c)

    # ---- differential evolution in DCT space ----
    pop = np.array([rand_c(1.0) for _ in range(40)])
    pop[0] = best_c
    fits = np.array([score_c(p) for p in pop])
    best_i = int(np.argmax(fits))
    best_c = pop[best_i].copy()
    best_s = fits[best_i]

    CR = 0.8
    F = 0.8
    it = 0
    while elapsed() < DEADLINE - 8:
        it += 1
        for i in range(pop.shape[0]):
            if elapsed() > DEADLINE - 8:
                break
            a, b, c = rng.choice(
                [j for j in range(pop.shape[0]) if j != i], size=3, replace=False)
            trial = pop[a] + F * (pop[b] - pop[c])
            trial = np.clip(trial, -8.0, 8.0)
            ft = score_c(trial)
            if ft >= fits[i]:
                pop[i] = trial
                fits[i] = ft
                if ft > best_s:
                    best_s = ft
                    best_c = trial.copy()

    # ---- final coordinate-descent polish from the DE winner ----
    best_c, best_s = polish(best_c, 40, 1.0)
    best_h = _c2h(best_c)

    # ---- tiny random restarts if time remains ----
    while elapsed() < DEADLINE - 4:
        c = rand_c(1.5)
        c, s = polish(c, 2, 1.0)
        if s > best_s:
            best_s = s
            best_c = c.copy()
            best_h = _c2h(c)

    best_h = _c2h(best_c)
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]
