"""
Strategy: Multi-start optimization with improved smooth surrogate and Nelder-Mead polish.
The value function is scale-invariant (value = l2/(norm_1*norm_inf)), so only the shape of h matters.
We use a smooth log-sum-exp approximation for max(|g|) during optimization, then polish
with Nelder-Mead on the exact value for final refinement.

Key ideas:
1. Smooth surrogate: replace norm_inf = max(|g|) with a soft-max (log-sum-exp) with high beta.
2. Logarithmic objective for better conditioning of the ratio.
3. Multiple diverse initializations + perturbation of best.
4. Time-budgeted search with safety margin.
5. Final Nelder-Mead polish on exact value for high-precision local refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    """Compute the exact value metric for given heights."""
    h = np.asarray(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')

    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0

    h = np.maximum(0, h)

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)

    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)

    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))

    if norm_1 == 0 or norm_inf == 0:
        return 0.0

    value = l2 / (norm_1 * norm_inf)
    return value


def compute_value_smooth(h, beta=200.0):
    """
    Compute a smooth surrogate of the value metric using log-sum-exp for norm_inf.
    """
    h = np.asarray(h, dtype=float)
    h = np.maximum(0, h)

    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)

    y_sq_terms = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq_terms)

    norm_1 = np.sum(np.abs(g)) / (m + 1)

    abs_g = np.abs(g)
    max_abs = np.max(abs_g)
    if max_abs < 1e-30:
        return 0.0

    scaled = beta * abs_g
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        max_scaled = 700
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_norm_inf = log_sum / beta

    if norm_1 == 0 or smooth_norm_inf < 1e-30:
        return 0.0

    value = l2 / (norm_1 * smooth_norm_inf)
    return value


def _coordinate_descent(h, max_iter=50, tol=1e-8, deadline=None):
    """
    Cyclic coordinate descent on individual heights.
    For each coordinate, do a 1D golden-section search over [0, upper].
    """
    h = np.maximum(0, np.asarray(h, dtype=float)).copy()
    best_val = compute_value(h)
    if best_val <= 0:
        return h, best_val

    max_h = np.max(h)
    if max_h < 1e-10:
        max_h = 1.0
    upper = 3.0 * max_h

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break

            a, b = 0.0, upper
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c = b - gr * (b - a)
            d = a + gr * (b - a)

            h[i] = c
            fc = compute_value(h)
            h[i] = d
            fd = compute_value(h)

            for _ in range(25):
                if deadline is not None and time.time() > deadline:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    h[i] = c
                    fc = compute_value(h)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    h[i] = d
                    fd = compute_value(h)

            best_x = c if fc < fd else d
            best_f = fc if fc < fd else fd
            h[i] = best_x

            if best_f > best_val:
                best_val = best_f
                improved = True

        if not improved:
            break

    h = np.maximum(0, h)
    final_val = compute_value(h)
    if final_val > best_val:
        best_val = final_val
    return h, best_val


def _generate_initializations():
    """Generate diverse initial height profiles."""
    x = np.linspace(0, N - 1, N)
    inits = []

    inits.append(np.ones(N))
    inits.append(np.exp(-0.5 * ((x - N/2) / (N/4))**2))
    peak1 = np.exp(-0.5 * ((x - N*0.25) / (N/8))**2)
    peak2 = np.exp(-0.5 * ((x - N*0.75) / (N/8))**2)
    inits.append(peak1 + peak2)
    plateau = np.zeros(N)
    plateau[N//4 : 3*N//4] = 1.0
    inits.append(plateau)
    inits.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    inits.append(1 - 2 * np.abs(x - (N-1)/2) / (N-1))
    inits.append(np.sqrt(np.maximum(0, 1 - ((2*x/(N-1) - 1))**2)))
    inits.append(np.exp(-0.5 * ((x - N/2) / (N/10))**2))
    inits.append(np.maximum(0, 1 - np.abs(x - (N-1)/2) / (N/3)))
    step = np.ones(N)
    step[:N//5] = 0.2
    step[-N//5:] = 0.2
    inits.append(step)
    ds = np.ones(N)
    ds[:N//8] = 0.1
    ds[-N//8:] = 0.1
    ds[N//3 : 2*N//3] = 1.5
    inits.append(ds)
    spike = np.zeros(N)
    spike[N//2] = 1.0
    inits.append(spike)
    ds2 = np.zeros(N)
    ds2[N//4] = 1.0
    ds2[3*N//4] = 1.0
    inits.append(ds2)
    inits.append(np.exp(-0.5 * ((x - N*0.4) / (N/5))**2))
    inits.append(np.exp(-0.5 * ((x - N*0.35) / (N/6))**2) + 0.3 * np.exp(-0.5 * ((x - N*0.65) / (N/4))**2))

    rng = np.random.RandomState(42)
    for _ in range(10):
        freqs = rng.randint(1, 8, size=6)
        phases = rng.uniform(0, 2*np.pi, size=6)
        amps = rng.uniform(0.5, 1.0, size=6)
        profile = np.zeros(N)
        for f, p, a in zip(freqs, phases, amps):
            profile += a * np.cos(2 * np.pi * f * x / N + p)
        profile = np.maximum(0.1, profile)
        inits.append(profile)

    for _ in range(10):
        points = np.sort(rng.uniform(0, N, size=5))
        heights = rng.uniform(0.5, 2.0, size=5)
        profile = np.interp(x, points, heights)
        inits.append(profile)

    for _ in range(10):
        profile = rng.uniform(0.1, 2.0, N)
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        kernel /= kernel.sum()
        profile = np.convolve(profile, kernel, mode='same')
        profile = np.maximum(0.1, profile)
        inits.append(profile)

    # New: Concentrated center with decay
    for sigma in [2, 3, 5]:
        inits.append(np.exp(-0.5 * ((x - N/2) / sigma)**2))

    # New: Power-law decay from center
    for p in [1.0, 2.0, 3.0]:
        inits.append(np.maximum(0.01, 1.0 - np.abs(x - N/2) / (N/2))**p)

    # New: Cosine window variants
    for width in [0.3, 0.5, 0.7]:
        center = N/2
        half_w = width * N/2
        window = np.zeros(N)
        mask = np.abs(x - center) < half_w
        window[mask] = 0.5 * (1 + np.cos(np.pi * (x[mask] - center) / half_w))
        inits.append(window)

    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def objective_smooth(h, beta=200.0):
        h = np.maximum(0, h)
        val = compute_value_smooth(h, beta=beta)
        if val <= 0:
            return 1e10
        return -val

    best_val = 0.0
    best_h = np.ones(N)

    seed_val = compute_value(np.ones(N))
    best_val = seed_val
    best_h = np.ones(N).copy()

    inits = _generate_initializations()

    # First pass: evaluate all initializations directly with exact value
    for init in inits:
        if time.time() > deadline - 20:
            break
        init = np.asarray(init, dtype=float)
        val = compute_value(init)
        if val > best_val:
            best_val = val
            best_h = init.copy()

    # Second pass: coordinate descent on top initializations
    init_vals = [(compute_value(np.asarray(init, dtype=float)), init) for init in inits]
    init_vals.sort(key=lambda x: -x[0])

    for idx in range(min(15, len(init_vals))):
        if time.time() > deadline - 15:
            break
        init = np.asarray(init_vals[idx][1], dtype=float).copy()
        h_cd, val_cd = _coordinate_descent(init, max_iter=30, deadline=deadline - 10)
        if val_cd > best_val:
            best_val = val_cd
            best_h = h_cd.copy()

    # Third pass: L-BFGS-B with smooth surrogate on top starts
    all_starts = [iv[1] for iv in init_vals[:10]]
    rng = np.random.RandomState(123)
    for _ in range(15):
        perturbed = best_h + rng.normal(0, 0.3, N)
        perturbed = np.maximum(0, perturbed)
        all_starts.append(perturbed)

    for idx, init in enumerate(all_starts):
        if time.time() > deadline - 12:
            break
        init = np.asarray(init, dtype=float).copy()
        try:
            res = minimize(
                lambda h: objective_smooth(h, beta=100.0),
                init,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = np.maximum(0, res.x)
            val_opt = compute_value(h_opt)
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Fourth pass: Coordinate descent + L-BFGS-B polish on best
    if time.time() < deadline - 10:
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            h_cd, val_cd = _coordinate_descent(best_h, max_iter=20, deadline=deadline - 8)
            if val_cd > best_val:
                best_val = val_cd
                best_h = h_cd.copy()
            try:
                res = minimize(
                    lambda h: objective_smooth(h, beta=500.0),
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                h_opt = np.maximum(0, res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
            if time.time() < deadline - 8:
                h_cd, val_cd = _coordinate_descent(best_h, max_iter=15, deadline=deadline - 6)
                if val_cd > best_val:
                    best_val = val_cd
                    best_h = h_cd.copy()

    # Fifth pass: Nelder-Mead polish with adaptive perturbation
    if time.time() < deadline - 5:
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            scale = 0.1 * (0.9 ** attempt)
            perturbed = best_h + np.random.normal(0, scale, N)
            perturbed = np.maximum(0, perturbed)
            try:
                def nm_objective(h):
                    h = np.maximum(0, h)
                    val = compute_value(h)
                    if val <= 0:
                        return 1e10
                    return -val
                res = minimize(
                    nm_objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                h_opt = np.maximum(0, res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # Sixth pass: Final high-precision L-BFGS-B + coordinate descent
    if time.time() < deadline - 2:
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(
                    lambda h: objective_smooth(h, beta=1000.0),
                    best_h,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-12}
                )
                h_opt = np.maximum(0, res.x)
                val_opt = compute_value(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
            if time.time() < deadline - 1:
                h_cd, val_cd = _coordinate_descent(best_h, max_iter=25, deadline=deadline - 0.5)
                if val_cd > best_val:
                    best_val = val_cd
                    best_h = h_cd.copy()

    best_h = np.maximum(0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print(f"Score: {compute_value(h)}")
