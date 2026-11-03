import numpy as np
from scipy.optimize import minimize
import time

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _autoconv(h):
    """Compute linear autoconvolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _score_raw(h):
    """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) with proper DELTA scaling."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    ac = _autoconv(h)
    ac_max = DELTA * np.max(ac)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def elapsed():
        return time.time() - start_time

    def optimize_slsqp(h_init, max_iter=800, ftol=1e-10):
        """
        Optimize h using SLSQP.
        Minimize M subject to:
          sum(h) == 1
          M - (h*h)_k >= 0 for all k
          h_i >= 0
        """
        h_init = np.asarray(h_init, dtype=np.float64)
        c_init = _autoconv(h_init)
        M_init = DELTA * np.max(c_init)
        x0 = np.concatenate([h_init, [M_init]])

        def obj(x):
            return x[N]

        def obj_grad(x):
            grad = np.zeros(N + 1)
            grad[N] = 1.0
            return grad

        def con_sum(x):
            return np.sum(x[:N]) - 1.0

        def con_sum_grad(x):
            grad = np.zeros(N + 1)
            grad[:N] = 1.0
            return grad

        def con_autoconv(x):
            h = x[:N]
            M = x[N]
            c = _autoconv(h)
            return M - DELTA * c

        def con_autoconv_grad(x):
            h = x[:N]
            grad = np.zeros((2 * N - 1, N + 1))
            for k in range(2 * N - 1):
                derivs = np.zeros(N)
                for i in range(N):
                    j = k - i
                    if 0 <= j < N:
                        derivs[i] = 2.0 * DELTA * h[j]
                grad[k, :N] = -derivs
                grad[k, N] = 1.0
            return grad

        bounds = [(0.0, None)] * N + [(0.0, None)]
        cons = [
            {'type': 'eq', 'fun': con_sum, 'jac': con_sum_grad},
            {'type': 'ineq', 'fun': con_autoconv, 'jac': con_autoconv_grad}
        ]

        try:
            res = minimize(
                obj, x0, method='SLSQP', jac=obj_grad,
                bounds=bounds, constraints=cons,
                options={'maxiter': max_iter, 'ftol': ftol}
            )
            if res.success:
                h_res = res.x[:N]
                h_res = np.clip(h_res, 0, None)
                s = np.sum(h_res)
                if s > 1e-12:
                    h_res /= s
                else:
                    return None, 0.0
                return h_res, _score_raw(h_res)
        except Exception:
            pass
        return None, 0.0

    # Generate diverse initializations
    np.random.seed(42)
    x_idx = np.arange(N)
    inits = []

    # 1. Uniform
    inits.append(np.ones(N) / N)

    # 2. Sparse random supports
    for trial in range(25):
        k = np.random.randint(2, 18)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)

    # 3. Concentrated Gaussians
    for c in np.linspace(2, N - 3, 12):
        for w in np.linspace(0.5, N / 3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w) ** 2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)

    # 4. Triangular
    for c in np.linspace(2, N - 3, 10):
        for w in np.linspace(2, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)

    # 5. Random smooth (convolved with small kernel)
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5) / 5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(h_r, 0)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)

    # 6. Geometric progressions
    for base in [2, 3]:
        support = []
        val = 1
        while val < N:
            support.append(val)
            val *= base
        if len(support) > 0:
            h_geom = np.zeros(N)
            h_geom[support] = 1.0 / len(support)
            inits.append(h_geom)

    # Optimize each initialization
    best_h = inits[0].copy()
    best_score = _score_raw(best_h)

    for h_init in inits:
        if elapsed() > deadline - 10:
            break
        h_opt, s_opt = optimize_slsqp(h_init, max_iter=500, ftol=1e-9)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Additional random restarts if time allows
    while elapsed() < deadline - 15:
        K = np.random.randint(2, 20)
        support = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[support] = 1.0 / K
        h_opt, s_opt = optimize_slsqp(h_rand, max_iter=300, ftol=1e-9)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Final high-precision polish pass
    if elapsed() < deadline - 5:
        h_opt, s_opt = optimize_slsqp(best_h, max_iter=2000, ftol=1e-12)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt.copy()

    # Ensure non-negative and normalized
    best_h = np.clip(best_h, 0, None)
    s = np.sum(best_h)
    if s > 1e-12:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0

    return list(best_h)
