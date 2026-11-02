import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 600

def compute_score(h):
    s = np.sum(h)
    if s == 0:
        return 0.0
    n_fft = 2 * N - 1
    H = np.fft.fft(h, n=n_fft)
    conv = np.real(np.fft.ifft(H * H))[:2 * N - 1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def compute_score_fast(h):
    """Faster score computation using real FFT."""
    s = np.sum(h)
    if s == 0:
        return 0.0
    n_fft = 2 * N - 1
    # Use real FFT for efficiency
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * np.conj(H), n=n_fft)[:2 * N - 1]
    max_conv = np.max(conv)
    return max_conv / (s * s)

def smooth_max_surrogate(conv, beta=100.0):
    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return (max_val + log_sum / beta)

def objective(params, beta=100.0):
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return 1e10
    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * np.conj(H), n=n_fft)[:2 * N - 1]
    smoothed_peak = smooth_max_surrogate(conv, beta)
    return smoothed_peak / (s * s)

def objective_grad(params, beta=100.0):
    h = np.maximum(params, 0.0)
    s = np.sum(h)
    if s < 1e-10:
        return np.zeros_like(params)

    n_fft = 2 * N - 1
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * np.conj(H), n=n_fft)[:2 * N - 1]

    max_val = np.max(conv)
    shifted = beta * (conv - max_val)
    shifted = np.clip(shifted, -100, 100)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    weights = exp_shifted / sum_exp

    # Use rfft/ifft for gradient
    W = np.fft.rfft(weights, n=n_fft)
    grad_h_full = np.fft.irfft(W * np.conj(H), n=n_fft)[:N]
    # The gradient of ||conv||^2 w.r.t. h is 2 * corr(conv, h)
    # But with our normalization, it's simpler
    grad_h = 2.0 * grad_h_full

    smooth_max_val = smooth_max_surrogate(conv, beta)
    grad_obj = grad_h / (s * s) - 2 * smooth_max_val / (s * s * s)

    grad_obj[h <= 1e-12] = 0.0

    return grad_obj

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    best_h = np.ones(N) / N
    best_score = compute_score(best_h)

    np.random.seed(42)

    initializations = []

    # 1. Uniform
    initializations.append(np.ones(N) / N)

    # 2. Gaussian
    x = np.linspace(-1, 1, N)
    gauss = np.exp(-x ** 2 / (2 * 0.5 ** 2))
    gauss /= np.sum(gauss)
    initializations.append(gauss)

    # 3. Linear ramp
    lin = np.linspace(0.1, 1.0, N)
    lin /= np.sum(lin)
    initializations.append(lin)

    # 4. Sine
    sine = np.sin(np.pi * (x + 1) / 2) ** 2
    sine /= np.sum(sine)
    initializations.append(sine)

    # 5. Step function (left half)
    step = np.zeros(N)
    step[:N // 2] = 1.0
    step /= np.sum(step)
    initializations.append(step)

    # 6. Random
    rand_h = np.random.rand(N)
    rand_h /= np.sum(rand_h)
    initializations.append(rand_h)

    # 7. Triangular
    tri = np.abs(np.linspace(-1, 1, N))
    tri = 1.0 - tri
    tri /= np.sum(tri)
    initializations.append(tri)

    # 8. Two-block: high-low-high (multiple variants)
    for frac in [0.25, 0.3, 0.33, 0.4]:
        two_block = np.zeros(N)
        two_block[:int(N * frac)] = 1.0
        two_block[int(N * (1 - frac)):] = 1.0
        if np.sum(two_block) > 0:
            two_block /= np.sum(two_block)
            initializations.append(two_block)

    # 9. Cosine bump
    cos_bump = np.cos(np.pi * x) ** 2
    cos_bump = np.maximum(cos_bump, 0)
    cos_bump /= np.sum(cos_bump)
    initializations.append(cos_bump)

    # 10. Two-block with smooth transition
    for frac in [0.25, 0.3, 0.35]:
        two_block_smooth = np.zeros(N)
        left_end = int(N * frac)
        right_start = int(N * (1 - frac))
        two_block_smooth[:left_end] = 1.0
        two_block_smooth[right_start:] = 1.0
        # Smooth edges
        edge_width = max(1, N // 50)
        if left_end > edge_width:
            for i in range(edge_width):
                two_block_smooth[left_end - edge_width + i] = i / edge_width
        if right_start < N - edge_width:
            for i in range(edge_width):
                two_block_smooth[right_start + i] = 1.0 - i / edge_width
        if np.sum(two_block_smooth) > 0:
            two_block_smooth /= np.sum(two_block_smooth)
            initializations.append(two_block_smooth)

    # 11. Three-block pattern
    three_block = np.zeros(N)
    third = N // 3
    three_block[:third // 2] = 1.0
    three_block[third + third // 2:2 * third - third // 2] = 1.0
    three_block[N - third // 2:] = 1.0
    if np.sum(three_block) > 0:
        three_block /= np.sum(three_block)
        initializations.append(three_block)

    bounds = [(0, None)] * N

    for h0 in initializations:
        if time.time() > deadline - 30:
            break

        try:
            # Phase 1: lower beta for smoother landscape
            res = minimize(
                objective,
                h0,
                method='L-BFGS-B',
                jac=objective_grad,
                args=(50.0,),
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )

            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)

            # Phase 2: refine with higher beta if time permits
            if time.time() < deadline - 20:
                res2 = minimize(
                    objective,
                    h_opt,
                    method='L-BFGS-B',
                    jac=objective_grad,
                    args=(100.0,),
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = res2.x
                h_opt = np.maximum(h_opt, 0)

            score_opt = compute_score(h_opt)

            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()

        except Exception:
            pass

    # Final polish with high beta and tight tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective,
                best_h,
                method='L-BFGS-B',
                jac=objective_grad,
                args=(200.0,),
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Additional: try coordinate descent / local refinement
    if time.time() < deadline - 5:
        try:
            h_local = best_h.copy()
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                improved = False
                for i in range(0, N, 10):  # Step by 10 for speed
                    if time.time() > deadline - 5:
                        break
                    for delta in [-0.01, -0.005, 0.0, 0.005, 0.01]:
                        h_test = h_local.copy()
                        h_test[i] = max(0, h_local[i] + delta)
                        if np.sum(h_test) > 1e-10:
                            score_test = compute_score(h_test)
                            if score_test < best_score - 1e-12:
                                h_local[i] = h_test[i]
                                best_score = score_test
                                best_h = h_local.copy()
                                improved = True
                                break
                if not improved:
                    break
        except Exception:
            pass

    best_h = np.maximum(best_h, 0)
    return best_h.tolist()
