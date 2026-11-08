"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 115.0  # safety margin before the 150 s kill
_N = N


def _overlaps(h):
    """S(k) = sum_i h[i] * (1 - h[(i+k) mod N]) for all shifts k."""
    n = len(h)
    # circular cross-correlation of h with (1 - h)
    # For each shift k, sum_i h[i] * (1 - h[(i+k) mod n])
    # Build via FFT-free approach using roll for small n (n=100 -> 100 rolls, fine)
    one_minus = 1.0 - h
    S = np.empty(n)
    for k in range(n):
        S[k] = np.dot(h, np.roll(one_minus, k))
    return S


def _softmax(S, beta):
    """Smooth max of S via log-sum-exp (scaled)."""
    m = np.max(S)
    return m + np.log(np.sum(np.exp(beta * (S - m)))) / beta


def construct():
    t0 = time.time()
    n = _N

    # Start from the seed (all 0.5) and perturb slightly to break symmetry.
    rng = np.random.default_rng(12345)
    h = np.full(n, 0.5)

    # Zero-sum perturbation direction space: use a fixed random zero-sum vector
    d0 = rng.standard_normal(n)
    d0 -= d0.mean()
    d0 /= (np.linalg.norm(d0) + 1e-12)

    best_h = h.copy()
    best_val = _softmax(_overlaps(h), 50.0)

    # Projected gradient descent on the smooth surrogate, keeping sum(h)=n/2 and 0<=h<=1.
    # Parameterize via zero-sum deltas to preserve the sum constraint exactly.
    beta = 50.0
    lr = 0.02
    steps = 4000

    for it in range(steps):
        if time.time() > _DEADLINE:
            break
        S = _overlaps(h)
        # gradient of softmax w.r.t. S
        e = np.exp(beta * (S - np.max(S)))
        e = e / e.sum()
        # dS[k]/dh[i]: S[k] = sum_i h[i]*(1-h[i+k]); d/dh[i] = (1-h[i+k]) - h[i-k]
        # Build gradient g[i] = sum_k e[k] * ((1-h[(i+k)%n]) - h[(i-k)%n])
        g = np.zeros(n)
        for k in range(n):
            ek = e[k]
            if ek < 1e-12:
                continue
            g += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
        # Move in zero-sum direction only (preserve sum)
        g -= g.mean()
        h_new = h - lr * g
        # Clip to [0,1] then re-center to keep sum = n/2
        h_new = np.clip(h_new, 0.0, 1.0)
        h_new += (n / 2.0 - h_new.sum()) / n
        h_new = np.clip(h_new, 0.0, 1.0)
        # Final exact sum fix via tiny redistribution
        h_new += (n / 2.0 - h_new.sum()) / n
        h = h_new

        val = _softmax(_overlaps(h), beta)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # High-precision polish: a few more steps with smaller lr on the best point
    if time.time() < _DEADLINE - 5.0:
        h = best_h.copy()
        beta2 = 100.0
        lr2 = 0.005
        for it in range(2000):
            if time.time() > _DEADLINE:
                break
            S = _overlaps(h)
            e = np.exp(beta2 * (S - np.max(S)))
            e = e / e.sum()
            g = np.zeros(n)
            for k in range(n):
                ek = e[k]
                if ek < 1e-12:
                    continue
                g += ek * ((1.0 - np.roll(h, k)) - np.roll(h, -k))
            g -= g.mean()
            h_new = np.clip(h - lr2 * g, 0.0, 1.0)
            h_new += (n / 2.0 - h_new.sum()) / n
            h_new = np.clip(h_new, 0.0, 1.0)
            h_new += (n / 2.0 - h_new.sum()) / n
            h = h_new
            val = _softmax(_overlaps(h), beta2)
            if val < best_val:
                best_val = val
                best_h = h.copy()

    # Enforce exact sum and bounds one last time
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h += (n / 2.0 - best_h.sum()) / n
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h += (n / 2.0 - best_h.sum()) / n

    return best_h
# EVOLVE-BLOCK-END
