import time
import numpy as np
from scipy.optimize import linprog

N = 100
SCALE = 2.0 / N
DEADLINE = 140.0  # wall-clock safety margin before the 150s kill


def _compute_O(h):
    """Overlap of h with shifted complement 1-h for all shifts k=0..N-1.
    O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    """
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)

    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)

    return np.concatenate([O_pos, O_neg])


def _build_lp():
    """Minimize t s.t. t >= O_k(h) for all k, 0<=h<=1, sum(h)=N/2.
    Variables x = [t, h_0..h_{N-1}].
    O_k(h) = SCALE * (s_k - dot(h, p_k))  =>  t - SCALE*dot(h,p_k) >= SCALE*s_k
    """
    idx = np.arange(N)
    s_pos = np.empty(N)
    s_neg = np.empty(N)
    p_pos = np.empty((N, N))
    p_neg = np.empty((N, N))
    for k in range(N):
        s_pos[k] = np.sum(h=None) if False else 0.0  # placeholder, overwritten below
    # precompute sums of h over index ranges
    for k in range(N):
        s_pos[k] = np.sum(np.ones(N - k)) * 0.0  # will use sum h[k:], handled in A
        s_neg[k] = np.sum(np.ones(N - k)) * 0.0
    # Build A_ub rows: t - SCALE * dot(h, p_k) >= SCALE * s_k
    #   => -t + SCALE*dot(h, p_k) <= -SCALE * s_k
    A_rows = []
    b_rows = []
    for k in range(N):
        # O_pos[k] = SCALE*(sum_{j=k}^{N-1} h[j] - dot(h[k:], h[:N-k]))
        # sum_{j=k}^{N-1} h[j] = dot(h, e_pos[k]) where e_pos[k][j]=1 if j>=k
        e_pos = (idx >= k).astype(float)
        p = np.zeros(N)
        p[:N - k] = h_none  # placeholder
        # p_k[j] = h[j-k] for j in [k, N-1]  -> as linear form in h: p_k[j] contributes h[j]*h[j-k]
        # dot(h, p_k) = sum_j h[j] * p_k[j] with p_k[j] = h[j-k] for j>=k else 0
        # This is the cross-correlation term; as a linear form it's sum_{j=k}^{N-1} h[j] * h[j-k].
        # Represented as row: for each j, coefficient of h[j] is h[j-k] (a constant w.r.t. variable? No!)
        # NOTE: O_k is LINEAR in h only if the OTHER factor is fixed. Here both are h -> it's QUADRATIC.
        # Re-examine: overlap of h with (1-h shifted). O = sum h[j]*(1 - h_shifted[j])
        #   = sum h[j] - sum h[j]*h_shifted[j]. The product h[j]*h[j-k] is QUADRATIC in h.
        # So the overlap is NOT linear in h. The LP claim only holds for bang-bang (h in {0,1})
        # where h[j]*h[j-k] = h[j] AND h[j-k] is linear. For continuous h it's quadratic.
        # => Fall back: solve LP on the bang-bang (binary) relaxation is infeasible/incorrect.
        # We therefore keep the LP only as a heuristic via the known fact that the optimum
        # of this max-overlap problem is attained at bang-bang points (extreme points), so we
        # search bang-bang configurations directly.
        pass
    # The overlap is quadratic; we drop the LP and rely on bang-bang local search + FISTA.
    return None


def _project(z):
    """Project z onto {x: 0<=x<=1, sum(x)=N/2} via bisection on the water level."""
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)


def _soft_max_grad(h, tau):
    O = _compute_O(h)
    Omax = float(O.max())
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    w_pos, w_neg = w[:N], w[N:]

    K = np.arange(N)[:, None]
    I = np.arange(N)[None, :]
    ge = (I >= K)
    lt = (I < (N - K))
    idx_p = np.clip(I - K, 0, N - 1)
    idx_m = np.clip(I + K, 0, N - 1)
    ge_f = ge.astype(float)
    lt_f = lt.astype(float)

    Gpos = SCALE * (ge_f - ge * h[idx_p] - lt * h[idx_m])
    Gneg = SCALE * (lt_f - lt * h[idx_m] - ge * h[idx_p])

    grad = w_pos @ Gpos + w_neg @ Gneg
    return Omax, grad


def _fista(h_init, deadline):
    h = _project(h_init.copy())
    h_prev = h.copy()
    t = 1.0
    L = 10.0
    for i in range(3000):
        if time.time() > deadline - 2.0:
            break
        Omax, grad = _soft_max_grad(h, 50.0)
        step = 1.0 / L
        y_proj = _project(h - step * grad)
        O_new = float(_compute_O(y_proj).max())
        if O_new < Omax or i == 0:
            h = y_proj
            L = L / 1.5
        else:
            L = L * 1.5
            continue
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        h_prev = h.copy()
        h = _project(h + ((t - 1) / t_new) * (h - h_prev))
        t = t_new
    return h


def _bang_bang(h):
    """Round to bang-bang keeping exactly N/2 ones."""
    h_bb = (h >= 0.5).astype(float)
    diff = int(round(N / 2.0 - h_bb.sum()))
    if diff > 0:
        zeros = np.where(h_bb == 0)[0]
        if len(zeros) >= diff:
            h_bb[zeros[:diff]] = 1.0
    elif diff < 0:
        ones = np.where(h_bb == 1)[0]
        if len(ones) >= -diff:
            h_bb[ones[: -diff]] = 0.0
    return h_bb


def _local_search_bb(h_bb, deadline, iters):
    """Swap-based local search on bang-bang configurations."""
    best_val = float(_compute_O(h_bb).max())
    ones = np.where(h_bb == 1)[0]
    zeros = np.where(h_bb == 0)[0]
    for _ in range(iters):
        if time.time() > deadline - 2.0:
            break
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = ones[np.random.randint(len(ones))]
        j = zeros[np.random.randint(len(zeros))]
        h_test = h_bb.copy()
        h_test[i] = 0.0
        h_test[j] = 1.0
        val = float(_compute_O(h_test).max())
        if val < best_val - 1e-9:
            h_bb = h_test
            best_val = val
            ones = np.where(h_bb == 1)[0]
            zeros = np.where(h_bb == 0)[0]
    return h_bb, best_val


def construct():
    start = time.time()
    deadline = start + DEADLINE

    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())

    idx = np.arange(N)
    inits = [seed.copy()]
    for f in (1, 2, 3, 5, 8):
        inits.append(0.5 + 0.30 * np.cos(2.0 * np.pi * f * idx / N))
        inits.append(0.5 + 0.40 * np.sign(np.cos(2.0 * np.pi * f * idx / N)))

    rng = np.random.default_rng(0)
    for _ in range(6):
        inits.append(rng.random(N))

    for init in inits:
        if time.time() > deadline - 8.0:
            break
        try:
            h_cont = _fista(init, deadline)
            val_cont = float(_compute_O(h_cont).max())
            if val_cont < best_val:
                best_h, best_val = h_cont, val_cont

            # Bang-bang refinement
            h_bb = _bang_bang(h_cont)
            val_bb = float(_compute_O(h_bb).max())
            if val_bb < best_val:
                best_h, best_val = h_bb, val_bb

            h_bb2, val_bb2 = _local_search_bb(h_bb, deadline, 4000)
            if val_bb2 < best_val:
                best_h, best_val = h_bb2, val_bb2
        except Exception:
            continue

    return best_h
