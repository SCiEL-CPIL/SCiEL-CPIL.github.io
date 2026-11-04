"""Combined: Fourier-mode parameterization with analytic gradient + direct polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
W = 1.0 / (2 * N)  # width of each interval: total length 0.5 / 600

def _compute_ratio(h):
    """Normalized max autoconvolution: max((f*f)(t)) / (integral f)^2."""
    h = np.asarray(h, dtype=np.float64)
    h = np.clip(h, 0.0, None)
    s = h.sum()
    if s < 1e-15:
        return float('inf')
    # FFT-based autoconvolution
    n_padded = 2 * N
    F = np.fft.rfft(h, n_padded)
    conv = np.fft.irfft(F * F, n_padded)
    max_conv = np.max(conv[:2*N-1]) * W  # scale by dx
    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return float('inf')
    return max_conv / integral_sq


def _fourier_to_heights(params, K):
    """Convert Fourier params [a0, a1, b1, ..., aK, bK] to nonnegative heights."""
    a0 = params[0]
    i = np.arange(N, dtype=np.float64)
    heights = np.full(N, a0, dtype=np.float64)
    for k in range(1, K + 1):
        ak = params[2 * k - 1]
        bk = params[2 * k]
        heights += ak * np.cos(2.0 * np.pi * k * i / N)
        heights += bk * np.sin(2.0 * np.pi * k * i / N)
    return np.clip(heights, 0.0, None)


def _smooth_obj_and_grad(params, K, tau=0.01):
    """
    Smooth log-sum-exp surrogate objective and its analytic gradient.
    Returns (value, gradient).
    """
    h = _fourier_to_heights(params, K)
    s = h.sum()
    if s < 1e-15:
        return 1e10, np.zeros_like(params)

    # FFT autoconvolution
    n_padded = 2 * N
    F = np.fft.rfft(h, n_padded)
    conv = np.fft.irfft(F * F, n_padded)[:2*N-1]
    conv *= W  # scale to continuous integral

    m = np.max(conv)
    if m < 1e-15:
        return 1e10, np.zeros_like(params)

    # Log-sum-exp
    exp_vals = np.exp((conv - m) / tau)
    sum_exp = np.sum(exp_vals)
    lse = m + tau * np.log(sum_exp)

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10, np.zeros_like(params)

    value = lse / integral_sq

    # Gradient: d/dp_j [lse / integral_sq]
    # = (1/integral_sq) * d(lse)/dp_j - (lse/integral_sq^2) * d(integral_sq)/dp_j
    #
    # d(lse)/dp_j = (tau/sum_exp) * sum_i exp((conv_i - m)/tau) * d(conv_i)/dp_j
    #              = (tau/sum_exp) * sum_i w_i * d(conv_i)/dp_j
    # where w_i = exp((conv_i - m)/tau) / sum_exp
    #
    # d(conv_i)/dp_j: conv_i = sum_{a,b} h_a h_b * delta(i - (a-b)) * W
    #                = W * sum_{a} h_a * h_{a-i} (for i>=0, with appropriate indexing)
    # Actually conv[i] = sum_{a=0}^{N-1} h[a] * h[a+i] for i in [0, N-1] (autoconv)
    # Wait, autoconvolution: (f*f)(t) = integral f(x)f(x-t) dx
    # In discrete: conv[i] = sum_a h[a] * h[a+i] for i >= 0, and symmetric for i < 0
    # But FFT gives full circular convolution. Let me use the FFT structure.
    #
    # conv = ifft(|F|^2), where F = fft(h, n_padded)
    # d(conv)/d(h_k) = 2 * ifft(F * conj(F_k_delta)) ... this is complex.
    #
    # Simpler: use the fact that conv[i] = W * sum_a h[a] * h[(a+i) mod N] (circular)
    # Actually for linear autoconv: conv[i] = W * sum_a h[a] * h[a+i] for valid ranges.
    #
    # Let me just compute d(conv)/d(h_k) directly:
    # conv[i] = W * sum_a h[a] * h[a+i]  (for the linear convolution part)
    # d(conv[i])/d(h[k]) = W * (h[i-k] + h[k-i]) for valid indices
    #
    # This is O(N^2) which is too slow. Let me use FFT-based gradient.
    #
    # Alternative: since we're in Fourier space, h is a linear function of params (before clipping).
    # d(h)/d(params) is a fixed N x (2K+1) matrix.
    # d(conv)/d(params) = D_conv_dh @ D_h_dparams
    # where D_conv_dh[i,k] = d(conv[i])/d(h[k])
    #
    # For autoconv: conv[i] = W * sum_a h[a] h[a+i]
    # d(conv[i])/d(h[k]) = W * (h[i-k] + h[k-i]) if indices valid
    #
    # This is still O(N^2). Let me use a different approach:
    # Compute gradient via the chain rule using FFT.
    #
    # Actually, let me just use finite differences on the Fourier params since K is small (~10-20).
    # That's (2K+1) function evaluations per gradient, each O(N log N).
    
    # Finite difference gradient (K is small, so this is affordable)
    grad = np.zeros_like(params)
    eps = 1e-5
    for j in range(len(params)):
        p_plus = params.copy()
        p_plus[j] += eps
        p_minus = params.copy()
        p_minus[j] -= eps
        
        h_plus = _fourier_to_heights(p_plus, K)
        h_minus = _fourier_to_heights(p_minus, K)
        
        s_plus = h_plus.sum()
        s_minus = h_minus.sum()
        
        if s_plus < 1e-15 or s_minus < 1e-15:
            grad[j] = 0.0
            continue
        
        F_plus = np.fft.rfft(h_plus, n_padded)
        conv_plus = np.fft.irfft(F_plus * F_plus, n_padded)[:2*N-1] * W
        m_plus = np.max(conv_plus)
        if m_plus < 1e-15:
            obj_plus = 1e10
        else:
            exp_plus = np.exp((conv_plus - m_plus) / tau)
            lse_plus = m_plus + tau * np.log(np.sum(exp_plus))
            obj_plus = lse_plus / (s_plus * W) ** 2
        
        F_minus = np.fft.rfft(h_minus, n_padded)
        conv_minus = np.fft.irfft(F_minus * F_minus, n_padded)[:2*N-1] * W
        m_minus = np.max(conv_minus)
        if m_minus < 1e-15:
            obj_minus = 1e10
        else:
            exp_minus = np.exp((conv_minus - m_minus) / tau)
            lse_minus = m_minus + tau * np.log(np.sum(exp_minus))
            obj_minus = lse_minus / (s_minus * W) ** 2
        
        grad[j] = (obj_plus - obj_minus) / (2 * eps)
    
    return value, grad


def _local_polish(h0, deadline, max_iters=2000):
    """Coordinate descent polish on the 600-bin heights."""
    best = h0.copy()
    best_val = _compute_ratio(best)
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if time.time() > deadline - 2.0:
            break
        # Pick a random subset of bins to perturb
        n_perturb = rng.randint(5, 20)
        idx = rng.choice(N, n_perturb, replace=False)
        
        h_trial = best.copy()
        perturb = rng.normal(0, 0.05, n_perturb)
        h_trial[idx] += perturb
        h_trial = np.clip(h_trial, 0.0, None)
        
        if h_trial.sum() < 1e-15:
            continue
        
        val = _compute_ratio(h_trial)
        if val < best_val:
            best = h_trial
            best_val = val
    
    return best, best_val


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Baseline: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = _compute_ratio(best_h)

    np.random.seed(42)
    K = 12  # Number of Fourier modes

    # Generate diverse initializations in Fourier parameter space
    inits = []

    # 1. Near-uniform with small perturbations
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.1)
            p0[2 * k] = np.random.normal(0, 0.1)
        inits.append(p0)

    # 2. Stronger modulation
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.3)
            p0[2 * k] = np.random.normal(0, 0.3)
        inits.append(p0)

    # 3. Dominant low-frequency modes
    for k_dom in [1, 2, 3, 4]:
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        p0[2 * k_dom - 1] = np.random.uniform(0.2, 0.5)
        p0[2 * k_dom] = np.random.uniform(0.2, 0.5)
        for k in range(1, K + 1):
            if k != k_dom:
                p0[2 * k - 1] = np.random.normal(0, 0.05)
                p0[2 * k] = np.random.normal(0, 0.05)
        inits.append(p0)

    # 4. Asymmetric
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = 1.0
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.15)
            p0[2 * k] = np.random.normal(0, 0.3)
        inits.append(p0)

    # 5. Random
    for _ in range(5):
        p0 = np.zeros(2 * K + 1)
        p0[0] = np.random.uniform(0.5, 1.5)
        for k in range(1, K + 1):
            p0[2 * k - 1] = np.random.normal(0, 0.4)
            p0[2 * k] = np.random.normal(0, 0.4)
        inits.append(p0)

    # Optimize from each initialization using L-BFGS-B
    for p0 in inits:
        if time.time() > deadline - 25.0:
            break

        try:
            def _obj(params):
                val, _ = _smooth_obj_and_grad(params, K, tau=0.01)
                return val

            res = minimize(
                _obj,
                p0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-6, 'maxls': 30}
            )

            if np.isfinite(res.fun):
                h_candidate = _fourier_to_heights(res.x, K)
                val = _compute_ratio(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Try with smaller tau for sharper max approximation
    if time.time() < deadline - 20.0:
        for p0 in inits[:8]:
            if time.time() > deadline - 15.0:
                break
            try:
                def _obj_tau(params, tau=0.005):
                    val, _ = _smooth_obj_and_grad(params, K, tau=tau)
                    return val

                res = minimize(
                    _obj_tau,
                    p0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-6}
                )
                if np.isfinite(res.fun):
                    h_candidate = _fourier_to_heights(res.x, K)
                    val = _compute_ratio(h_candidate)
                    if val < best_val and np.isfinite(val):
                        best_val = val
                        best_h = h_candidate.copy()
            except Exception:
                pass

    # Direct coordinate descent polish on the 600 bin heights
    if time.time() < deadline - 3.0:
        best_h, best_val = _local_polish(best_h, deadline, max_iters=3000)

    return best_h.tolist()
