N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _ac_max_abs(ac):
    return float(np.max(np.abs(ac)))

def _full_score(h):
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _ac_max_abs(ac) / (s * s)

def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    # Best-so-far across all strategies (scale-invariant score)
    best_h = np.ones(N, dtype=np.float64)
    best_val = _full_score(best_h)

    # ---------- Binary ±1 search with fast incremental autoconvolution ----------
    # For a ±1 sequence, flipping bit i changes only lags in [i-(N-1), i+(N-1)].
    def _init_state(b):
        # b: ±1 array
        n = len(b)
        ac = np.zeros(2 * n - 1, dtype=np.float64)
        for k in range(2 * n - 1):
            lo = max(0, k - n + 1)
            hi = min(k + 1, n)
            ac[k] = float(np.dot(b[lo:hi], b[k + 1 - hi:k + 1 - lo]))
        return ac

    def _flip(ac, b, i):
        n = len(b)
        # lags k where i participates: 0 <= i <= k and 0 <= k-i <= n-1
        lo = max(0, i - (n - 1))
        hi = min(2 * n - 2, i + (n - 1))
        for k in range(lo, hi + 1):
            j = k - i
            ac[k] -= 2.0 * b[i] * b[j]
        b[i] = -b[i]

    def _unflip(ac, b, i):
        # b[i] is now the NEW value; undo the contribution of the new value
        n = len(b)
        lo = max(0, i - (n - 1))
        hi = min(2 * n - 2, i + (n - 1))
        for k in range(lo, hi + 1):
            j = k - i
            ac[k] -= 2.0 * b[i] * b[j]
        b[i] = -b[i]

    def _steepest_local(b, ac, max_passes=200):
        n = len(b)
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
            best_i = -1
            best_new = _ac_max_abs(ac)
            for i in range(n):
                _flip(ac, b, i)
                nv = _ac_max_abs(ac)
                _unflip(ac, b, i)
                if nv < best_new - 1e-12:
                    best_new = nv
                    best_i = i
            if best_i < 0:
                break
            _flip(ac, b, best_i)
        return _ac_max_abs(ac)

    def _anneal(b0, t_start, t_end, n_iter, seed=0):
        rng = np.random.RandomState(seed)
        b = b0.copy()
        ac = _init_state(b)
        cur = _ac_max_abs(ac)
        best_b = b.copy()
        best_val_local = cur
        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break
            frac = it / max(1, n_iter - 1)
            T = t_start * (t_end / t_start) ** frac
            i = rng.randint(N)
            _flip(ac, b, i)
            new = _ac_max_abs(ac)
            if new < cur or (T > 1e-9 and rng.rand() < np.exp(-(new - cur) / T)):
                cur = new
                if cur < best_val_local - 1e-12:
                    best_val_local = cur
                    best_b = b.copy()
            else:
                _unflip(ac, b, i)
        return best_b, best_val_local

    # --- Initialize with several diverse ±1 sequences ---
    inits = []
    # 1. All +1
    inits.append(np.ones(N, dtype=np.float64))
    # 2. Alternating
    inits.append(np.where(np.arange(N) % 2 == 0, 1.0, -1.0))
    # 3. Random
    for s in range(3):
        rng = np.random.RandomState(1000 + s)
        inits.append(rng.choice([1.0, -1.0], size=N))
    # 4. Block structure (few sign changes)
    rng = np.random.RandomState(42)
    b = np.ones(N)
    for _ in range(5):
        b[rng.randint(N):] = -b[rng.randint(N):]
    inits.append(b)

    for idx, b0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        b = b0.copy()
        ac = _init_state(b)
        # Local polish
        _steepest_local(b, ac, max_passes=150)
        cur = _ac_max_abs(ac)
        if cur < best_val - 1e-12:
            best_val = cur
            best_h = b.copy()
        # Anneal from this state
        b_a, v_a = _anneal(b, t_start=1.0, t_end=0.001, n_iter=8000, seed=idx)
        if v_a < best_val - 1e-12:
            best_val = v_a
            best_h = b_a.copy()
            # Re-polish the annealed result
            ac_a = _init_state(b_a)
            _steepest_local(b_a, ac_a, max_passes=100)
            v_a2 = _ac_max_abs(ac_a)
            if v_a2 < best_val - 1e-12:
                best_val = v_a2
                best_h = b_a.copy()

    # ---------- Continuous refinement (smooth, non-binary) ----------
    # Try to improve the best binary solution by allowing small real perturbations
    # using a soft-max differentiable objective with BFGS.
    if time.time() < deadline - 15.0:
        import scipy.optimize as opt

        def _obj_soft(x):
            n = len(x)
            size = 2 * n - 1
            fsize = 1
            while fsize < size:
                fsize *= 2
            H = np.fft.rfft(x, fsize)
            ac = np.fft.irfft(H * H, fsize)
            s = np.sum(x)
            if abs(s) < 1e-6:
                return 1e10
            eps = 0.05
            # soft max of |ac|
            a = np.abs(ac) / eps
            m = np.max(a)
            soft = m + np.log(np.sum(np.exp(a - m)))
            return soft * eps / (s * s)

        h0 = best_h.copy()
        # Normalize to sum 1 for stability
        s0 = np.sum(h0)
        if abs(s0) > 1e-6:
            h0 = h0 / s0
        try:
            res = opt.minimize(_obj_soft, h0, method='BFGS',
                               options={'maxiter': 30, 'gtol': 1e-5})
            h_opt = res.x
            v_opt = _full_score(h_opt)
            if v_opt < best_val - 1e-12:
                best_val = v_opt
                best_h = h_opt
        except Exception:
            pass

    # Safety: ensure we never return worse than all-ones
    base_val = _full_score(np.ones(N))
    if best_val > base_val:
        best_val = base_val
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
