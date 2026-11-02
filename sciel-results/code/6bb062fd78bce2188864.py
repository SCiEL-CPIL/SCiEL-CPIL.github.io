import time
import numpy as np

N = 100
SCALE = 2.0 / N
RNG = np.random.default_rng(2024)

def _compute_O(h):
    """Compute overlaps for all shifts."""
    h = np.asarray(h, dtype=float)
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    return np.concatenate([O_pos, O_neg])

def _project(z):
    """Project z onto {x: sum(x)=N/2, 0<=x_i<=1} via bisection on threshold."""
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _soft_max(O, beta):
    m = O.max()
    return m + np.log(np.sum(np.exp(beta * (O - m)))) / beta

def _soft_max_grad(O, beta):
    m = O.max()
    e = np.exp(beta * (O - m))
    return e / e.sum()

def _grad_O(h, k, idx):
    """Gradient of O_k w.r.t h. idx in [0,2N). k = idx if idx<N else idx-N."""
    g = np.zeros(N)
    if idx < N:
        k = idx
        # O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j] h[j-k])
        for j in range(k, N):
            g[j] += SCALE * (1 - h[j - k])
            g[j - k] -= SCALE * h[j]
    else:
        k = idx - N
        # O_neg[k] = SCALE * (sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j] h[j+k])
        for j in range(N - k):
            g[j] += SCALE * (1 - h[j + k])
            g[j + k] -= SCALE * h[j]
    return g

def _grad_soft_max(h, beta):
    O = _compute_O(h)
    w = _soft_max_grad(O, beta)
    g = np.zeros(N)
    for idx in range(2 * N):
        gi = _grad_O(h, idx, idx)
        g += w[idx] * gi
    return g

def _optimize(h_init, deadline, beta_schedule):
    """Smooth surrogate optimization via accelerated projected gradient."""
    h = _project(h_init.copy())
    y = h.copy()
    t = 1.0
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    # Lipschitz estimate
    L = 0.5
    it = 0
    while time.time() < deadline - 8.0:
        it += 1
        # beta anneals
        frac = min(1.0, it / 2000.0)
        beta = beta_schedule[0] + (beta_schedule[1] - beta_schedule[0]) * frac
        grad = _grad_soft_max(y, beta)
        step = 1.0 / L
        h_new = _project(y - step * grad)
        # backtracking: check smoothness
        f_new = _soft_max(_compute_O(h_new), beta)
        f_y = _soft_max(_compute_O(y), beta)
        # simple: accept and adapt L
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y_new = h_new + ((t - 1.0) / t_new) * (h_new - h)
        h = h_new
        y = np.clip(y_new, 0.0, 1.0)
        # re-project to maintain sum
        y = _project(y)
        t = t_new
        val = float(_compute_O(h).max())
        if val < best_val:
            best_val = val
            best_h = h.copy()
    return best_h

def _local_search_bb(h, deadline):
    """Bang-bang local search: swap 1s and 0s to reduce max overlap."""
    h_bb = (h >= 0.5).astype(float)
    # fix sum
    target = N // 2
    cur = int(h_bb.sum())
    if cur < target:
        zeros = np.where(h_bb == 0)[0]
        # pick largest h values
        order = np.argsort(-h[zeros])
        need = target - cur
        for idx in order[:need]:
            h_bb[zeros[idx]] = 1.0
    elif cur > target:
        ones = np.where(h_bb == 1)[0]
        order = np.argsort(h[ones])
        need = cur - target
        for idx in order[:need]:
            h_bb[ones[idx]] = 0.0

    best_val = float(_compute_O(h_bb).max())
    # greedy local search
    ones = np.where(h_bb == 1)[0]
    zeros = np.where(h_bb == 0)[0]
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        ones = np.where(h_bb == 1)[0]
        zeros = np.where(h_bb == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        # try all swaps (N^2/4 ~ 625, cheap)
        best_i, best_j, best_v = None, None, best_val
        for i in ones:
            for j in zeros:
                h_bb[i] = 0.0
                h_bb[j] = 1.0
                v = float(_compute_O(h_bb).max())
                if v < best_v - 1e-12:
                    best_v = v
                    best_i, best_j = i, j
                h_bb[i] = 1.0
                h_bb[j] = 0.0
        if best_i is not None:
            h_bb[best_i] = 0.0
            h_bb[best_j] = 1.0
            best_val = best_v
            improved = True
    return h_bb

def construct():
    start = time.time()
    deadline = start + 140.0

    idx = np.arange(N)
    inits = []
    inits.append(np.full(N, 0.5))
    for f in (1, 2, 3, 4, 5, 7, 11):
        inits.append(0.5 + 0.40 * np.cos(2.0 * np.pi * f * idx / N))
        inits.append(0.5 + 0.45 * np.sign(np.cos(2.0 * np.pi * f * idx / N)))
    for _ in range(6):
        inits.append(RNG.random(N))
    # random bang-bang inits
    for _ in range(4):
        x = RNG.random(N)
        x = (x >= np.sort(x)[N // 2]).astype(float)
        inits.append(x)

    best_h = inits[0].copy()
    best_val = float(_compute_O(best_h).max())

    beta_sched = (5.0, 40.0)
    for init in inits:
        if time.time() > deadline - 12.0:
            break
        try:
            h_cont = _optimize(init, deadline, beta_sched)
            val_cont = float(_compute_O(h_cont).max())
            if val_cont < best_val:
                best_h, best_val = h_cont, val_cont
            # bang-bang polish
            h_bb = _local_search_bb(h_cont, deadline)
            val_bb = float(_compute_O(h_bb).max())
            if val_bb < best_val:
                best_h, best_val = h_bb, val_bb
        except Exception:
            continue

    # final polish on best
    if time.time() < deadline - 5.0:
        try:
            h_bb = _local_search_bb(best_h, deadline)
            val_bb = float(_compute_O(h_bb).max())
            if val_bb < best_val:
                best_h, best_val = h_bb, val_bb
        except Exception:
            pass

    return best_h
