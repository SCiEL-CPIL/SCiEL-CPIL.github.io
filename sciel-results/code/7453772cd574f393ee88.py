"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT sizes for fast convolution
    fft_size = 1  # smallest power of 2 >= 2*N-1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        """Fast convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def smooth_objective(x, beta=60.0):
        """Smooth surrogate using log-sum-exp of conv entries."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        # Numerically stable log-sum-exp
        max_shifted = np.max(shifted)
        if max_shifted > 500:  # overflow protection
            return 1e10
        log_sum_exp = max_c + max_shifted + np.log(np.sum(np.exp(beta * (shifted - max_shifted)))) / beta
        val = abs(2.0 * N * log_sum_exp / (s * s))
        if not np.isfinite(val):
            return 1e10
        return val

    def smooth_grad(x, beta=60.0):
        """Analytic gradient of the smooth surrogate, vectorized."""
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)

        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        max_shifted = np.max(shifted)
        if max_shifted > 500:
            return np.zeros_like(h)

        weights = np.exp(beta * (shifted - max_shifted))
        sum_weights = np.sum(weights)
        if sum_weights < 1e-300:
            return np.zeros_like(h)
        weights /= sum_weights

        # grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[m+i] * h[m]
        # This is a cross-correlation between weights (length 2N-1) and h (length N)
        # grad_lse[i] = 2 * (weights * h)[i] in the appropriate convolution sense
        # Specifically: grad_lse[i] = 2 * sum_{m} weights[i+m] * h[m]
        # This is the cross-correlation of weights with h, shifted appropriately
        # Using FFT: correlate(weights, h) gives sum_k weights[k] * h[k-i] for various i
        # We want sum_m weights[i+m] * h[m]
        # Let's compute via direct approach but vectorized with stride tricks
        # Actually for N=400, we can use a matrix-free approach
        
        # Efficient: use the fact that this is a convolution-like operation
        # grad_lse[i] = 2 * <weights[i:i+N], h>
        # We can compute all of these at once using matrix multiplication
        # Create a Toeplitz-like matrix from weights
        # M[i, m] = weights[i+m] for i in [0,N-1], m in [0,N-1]
        # Then grad_lse = 2 * M @ h
        
        # Building the full matrix is O(N^2) = 160000 elements, which is fine
        idx = np.arange(N)
        M = weights[idx[:, None] + idx[None, :]]  # M[i,m] = weights[i+m]
        grad_lse = 2.0 * (M @ h)

        lse = max_c + max_shifted + np.log(sum_weights) / beta
        f = 2.0 * N * lse / (s * s)

        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)

        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh

        return grad

    rng = np.random.RandomState(42)

    # --- Generate diverse initial points ---
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # Random uniform
    for _ in range(10):
        initial_points.append(rng.uniform(-1, 1, N))

    # Bimodal: mostly zero with random spikes
    for _ in range(10):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)

    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)

    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.1 * rng.normal(0, 1, N))

    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [15, 25, 40, 60, 80, 120, 160, 200]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)
            h_double = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) - np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_double)
            h_off = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2)) + 0.3 * np.ones(N)
            initial_points.append(h_off)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7]:
        for _ in range(5):
            freqs = rng.randint(1, 30, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)

    # Sort by initial value (best first)
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break

        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1

        try:
            result = minimize(
                smooth_objective,
                init_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-5}
            )

            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass

        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()

    # --- Local polish using coordinate descent ---
    if time.time() - start_time < DEADLINE - 5:
        h_curr = best_h.copy()
        lr = 0.05
        improved = True
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                for delta in [lr, -lr, lr * 0.5, -lr * 0.5, lr * 0.25, -lr * 0.25]:
                    h_try = h_curr.copy()
                    h_try[coord] += delta
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try
                        h_curr = h_try
                        improved = True
                        break
            if not improved:
                lr *= 0.7
                if lr < 1e-4:
                    break

    # --- Perturbation phase ---
    if time.time() - start_time < DEADLINE - 3:
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 3:
                break
            h_pert = best_h + 0.1 * rng.normal(0, 1, N)
            val_pert, valid = evaluate(h_pert)
            if valid and val_pert < best_value:
                best_value = val_pert
                best_h = h_pert
            else:
                try:
                    result = minimize(
                        smooth_objective,
                        h_pert,
                        jac=smooth_grad,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-5}
                    )
                    h_opt = result.x
                    val_opt, valid = evaluate(h_opt)
                    if valid and val_opt < best_value:
                        best_value = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    return best_h
