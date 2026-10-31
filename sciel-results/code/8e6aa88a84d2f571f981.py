import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def shape_score(h):
    """Scale-invariant score: minimize ||h*h||_inf * ||h||_1 / ||h||_2^2"""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq < 1e-15:
        return 0.0
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l1 * conv_peak / l2_sq

def log_shape_score(log_h):
    """Shape score in log-space for gradient-based optimization."""
    h = np.exp(log_h)
    return shape_score(h)

def log_shape_grad(log_h):
    """Gradient of shape score w.r.t. log_h via finite differences."""
    h = np.exp(log_h)
    eps = 1e-7
    grad = np.zeros_like(log_h)
    for i in range(N):
        log_h_plus = log_h.copy()
        log_h_plus[i] += eps
        log_h_minus = log_h.copy()
        log_h_minus[i] -= eps
        grad[i] = (log_shape_score(log_h_plus) - log_shape_score(log_h_minus)) / (2 * eps)
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_h = np.ones(N) / N
    best_shape = shape_score(best_h)

    # Generate diverse initial candidates
    candidates = []

    # 1. Uniform
    candidates.append(np.ones(N) / N)

    # 2. Sparse patterns: pick k indices with equal weights
    for k in range(1, 15):
        for _ in range(5):
            idx = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            h /= np.sum(h)
            candidates.append(h)

    # 3. Sparse with geometric weights
    for k in range(2, 12):
        for _ in range(3):
            idx = np.random.choice(N, k, replace=False)
            idx = np.sort(idx)
            weights = np.array([0.5 ** (k - 1 - i) for i in range(k)])
            h = np.zeros(N)
            h[idx] = weights
            h /= np.sum(h)
            candidates.append(h)

    # 4. Block of consecutive ones
    for k in range(1, 20):
        for start in range(0, max(1, N - k + 1), max(1, (N - k) // 8)):
            h = np.zeros(N)
            h[start:start+k] = 1.0
            h /= np.sum(h)
            candidates.append(h)

    # 5. Alternating patterns
    for period in range(2, 12):
        for offset in range(period):
            h = np.zeros(N)
            for i in range(N):
                if (i - offset) % period == 0:
                    h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                candidates.append(h)

    # 6. Gaussian-like profiles
    for sigma in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        for center in [N//4, N//2, 3*N//4]:
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            h += 1e-12
            h /= np.sum(h)
            candidates.append(h)

    # 7. Power law profiles
    for p in [0.5, 1.0, 1.5, 2.0, 3.0]:
        x = np.arange(N)
        center = N / 2.0
        h = (1.0 + (x - center) ** 2) ** (-p)
        h += 1e-12
        h /= np.sum(h)
        candidates.append(h)

    # 8. Random Dirichlet
    for _ in range(100):
        alpha = np.random.dirichlet(np.ones(N))
        candidates.append(alpha)

    # 9. Random sparse with weights
    for _ in range(100):
        k = np.random.randint(2, 10)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = np.random.dirichlet(np.ones(k))
        h /= np.sum(h)
        candidates.append(h)

    # 10. Two-point patterns
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            h /= np.sum(h)
            candidates.append(h)

    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = shape_score(h)
        if s < best_shape:
            best_shape = s
            best_h = h.copy()

    # Gradient-based optimization using L-BFGS-B in log-space
    if time.time() < deadline:
        # Start from best candidate, convert to log-space
        h_init = best_h.copy()
        # Ensure all values are positive for log
        h_init = np.maximum(h_init, 1e-10)
        log_h_init = np.log(h_init)

        try:
            result = minimize(
                log_shape_score,
                log_h_init,
                method='L-BFGS-B',
                jac=log_shape_grad,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            if time.time() < deadline and result.success:
                h_opt = np.exp(result.x)
                s_opt = shape_score(h_opt)
                if s_opt < best_shape:
                    best_shape = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Multiple random restarts with L-BFGS-B
    if time.time() < deadline:
        num_restarts = 30
        for r in range(num_restarts):
            if time.time() > deadline:
                break

            # Create diverse starting point
            if r % 3 == 0:
                # Sparse start
                k = np.random.randint(2, 10)
                idx = np.random.choice(N, k, replace=False)
                h = np.zeros(N)
                h[idx] = np.random.dirichlet(np.ones(k))
            elif r % 3 == 1:
                # Perturbed best
                h = best_h.copy()
                mask = np.random.rand(N) < 0.5
                h[mask] *= np.random.uniform(0.1, 5.0, mask.sum())
                h = np.maximum(h, 1e-10)
                h /= np.sum(h)
            else:
                # Gaussian-like start
                sigma = np.random.uniform(0.5, 8.0)
                center = np.random.uniform(0, N)
                x = np.arange(N)
                h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
                h += 1e-10
                h /= np.sum(h)

            h = np.maximum(h, 1e-10)
            log_h_init = np.log(h)

            try:
                result = minimize(
                    log_shape_score,
                    log_h_init,
                    method='L-BFGS-B',
                    jac=log_shape_grad,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
                )
                if time.time() < deadline and result.success:
                    h_opt = np.exp(result.x)
                    s_opt = shape_score(h_opt)
                    if s_opt < best_shape:
                        best_shape = s_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

    # Final polish with more iterations from the best point
    if time.time() < deadline:
        h_init = np.maximum(best_h, 1e-10)
        log_h_init = np.log(h_init)
        try:
            result = minimize(
                log_shape_score,
                log_h_init,
                method='L-BFGS-B',
                jac=log_shape_grad,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            if result.success:
                h_opt = np.exp(result.x)
                s_opt = shape_score(h_opt)
                if s_opt < best_shape:
                    best_shape = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Scale to maximize the actual ratio
    # ratio = l2_sq / (l1 * conv_peak)
    # For fixed shape, if we scale h by c: l2_sq -> c^2 * l2_sq, l1 -> c * l1, conv_peak -> c^2 * conv_peak
    # So ratio -> c^2 * l2_sq / (c * l1 * c^2 * conv_peak) = l2_sq / (l1 * c * conv_peak)
    # Ratio is proportional to 1/c, so smaller c gives larger ratio
    # But we need nonnegative heights; use a small but not zero scale
    scale = 1e-6
    final_h = best_h * scale

    return final_h.tolist()
