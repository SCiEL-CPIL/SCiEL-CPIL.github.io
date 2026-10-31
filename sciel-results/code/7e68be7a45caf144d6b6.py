"""Low-rank Gaussian-mixture design for the first autocorrelation inequality.

We represent the step function (on [-1/4,1/4]) as a nonnegative mixture of
Gaussian bumps.  A mixture can approximate a flat-top (box) function, whose
autoconvolution peak is comparatively low.  We optimize the continuous
parameters (centers, widths, weights) with a bounded population-based search
(differential-evolution style plus coordinate refinement), then discretize the
smooth profile onto the 600 equal intervals and return the bin heights.
"""

import time
import numpy as np

N = 600
A = 0.25
DX = (2.0 * A) / N          # width of each interval = 1/600
# Bin-center x grid on [-A, A]
X = np.linspace(-A + DX / 2.0, A - DX / 2.0, N)

# Evaluation grid (a bit finer than bins for a smooth peak estimate)
M = 2000
XE = np.linspace(-A + DX / 2.0, A - DX / 2.0, M)
DXE = (2.0 * A) / M

DEADLINE = 140.0            # wall-clock safety margin (limit is 150s)
START = time.time()


def _profile_from_params(params):
    """Smooth value f(x) on the evaluation grid from a param vector.

    params layout:
      weights[0:K]   (nonnegative, scaled to sum=1)
      centers[K:2K]  (in [-A, A])
      widths[2K:3K]  (in [wmin, A])
    """
    K = 4
    w = params[0:K].copy()
    c = params[K:2 * K]
    s = params[2 * K:3 * K]
    # normalize weights to sum 1
    ws = w.sum()
    if ws <= 0:
        w = np.ones(K) / K
    else:
        w = w / ws
    # build f(x) as weighted sum of gaussians
    f = np.zeros_like(XE)
    for k in range(K):
        f += w[k] * np.exp(-0.5 * ((XE - c[k]) / s[k]) ** 2)
    return f


def _score_from_profile(f):
    """Normalized maximum of autoconvolution.  Lower is better."""
    conv = np.convolve(f, f, mode='full')
    peak = conv.max()
    area = f.sum() * DXE
    return peak / (area * area)


def _score_from_params(params):
    f = _profile_from_params(params)
    return _score_from_profile(f)


def _sample_params(rng, K=4):
    w = rng.uniform(0.5, 2.0, size=K)
    c = rng.uniform(-A * 0.5, A * 0.5, size=K)
    s = rng.uniform(0.03, A, size=K)
    return np.concatenate([w, c, s])


def _clamp_params(params, K=4):
    p = params.copy()
    # weights: keep nonnegative, small floor
    p[0:K] = np.clip(p[0:K], 1e-4, None)
    # centers
    p[K:2 * K] = np.clip(p[K:2 * K], -A, A)
    # widths
    p[2 * K:3 * K] = np.clip(p[2 * K:3 * K], 0.02, A)
    return p


def construct():
    K = 4
    dim = 3 * K
    rng = np.random.default_rng(12345)

    # ---- initial population ----
    POP = 40
    pop = np.zeros((POP, dim))
    for i in range(POP):
        pop[i] = _sample_params(rng, K)
    scores = np.array([_score_from_params(pop[i]) for i in range(POP)])
    best_idx = int(np.argmin(scores))
    best = pop[best_idx].copy()
    best_score = scores[best_idx]

    # ---- differential-evolution style population search ----
    F = 0.8
    CR = 0.9
    gen = 0
    while time.time() < DEADLINE - 20.0:
        gen += 1
        order = np.argsort(scores)
        for i in range(POP):
            # pick three distinct others
            a, b, c = rng.choice(POP, size=3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = rng.choice(POP, size=3, replace=False)
            trial = pop[a] + F * (pop[b] - pop[c])
            # crossover
            mask = rng.random(dim) < CR
            if not mask.any():
                mask[rng.integers(dim)] = True
            trial = np.where(mask, trial, pop[i])
            trial = _clamp_params(trial, K)
            ts = _score_from_params(trial)
            if ts <= scores[i]:
                pop[i] = trial
                scores[i] = ts
                if ts < best_score:
                    best_score = ts
                    best = trial.copy()
        # occasional re-injection of fresh random individuals
        if gen % 5 == 0:
            for i in rng.choice(POP, size=3, replace=False):
                pop[i] = _sample_params(rng, K)
                scores[i] = _score_from_params(pop[i])
                if scores[i] < best_score:
                    best_score = scores[i]
                    best = pop[i].copy()

    # ---- coordinate refinement (local hill-climbing) on the best ----
    cur = best.copy()
    cur_score = best_score
    step = 0.05
    iters = 0
    while time.time() < DEADLINE - 2.0 and iters < 400:
        iters += 1
        improved = False
        for d in range(dim):
            for direction in (1.0, -1.0):
                cand = cur.copy()
                cand[d] += direction * step
                cand = _clamp_params(cand, K)
                cs = _score_from_params(cand)
                if cs < cur_score:
                    cur = cand
                    cur_score = cs
                    improved = True
        if not improved:
            step *= 0.6
            if step < 1e-4:
                break

    # ---- discretize the smooth profile onto the N bins ----
    f = _profile_from_params(cur)
    # interpolate from eval grid to bin centers
    heights = np.interp(X, XE, f)
    # ensure nonnegative
    heights = np.clip(heights, 0.0, None)
    return [float(h) for h in heights]
