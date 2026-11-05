"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill

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

    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    # ---- Fourier-domain approach ----
    # Parameterize h by its DFT amplitudes and phases.
    # H(k) for k=0..N-1 (full DFT). We work with the full complex DFT.
    # h = ifft(H), where H = A * exp(i*phi)
    # Constraint: sum(h) = H(0) must be nonzero.
    # Objective: minimize max(invFFT(|H|^2)) / |H(0)|^2
    #   Since invFFT(|H|^2)[0] = sum(|H|^2)/N (by Parseval), and the max is over all lags.
    #   Actually conv(h,h) = ifft(|fft(h)|^2) up to scaling.
    #   np.convolve(h,h)[l] = sum_i h[i]*h[i+l] for appropriate l.
    #   The full autoconvolution has 2N-1 entries.
    #   fft(h) has N entries. |fft(h)|^2 has N entries. ifft(|fft(h)|^2) gives N entries
    #   which correspond to the circular auto-correlation. The linear autoconvolution is
    #   related but not identical. For N=400, let's just work in time domain but use
    #   Fourier insight for initialization.

    # Better structural approach: use the fact that we want h such that its
    # autoconvolution is as flat as possible. This is related to Barker sequences
    # and low-autocorrelation sequences.
    #
    # New structural idea: parameterize h = base + perturbation where base is chosen
    # to have flat autoconvolution, then optimize the perturbation in a reduced space.
    #
    # Actually, let's try a completely different optimization: use the log-sum-exp
    # smooth approximation but optimize in a Fourier basis instead of Chebyshev.

    # Fourier basis: h = sum_k c_k * cos(2*pi*k*n/N) + d_k * sin(2*pi*k*n/N)
    # This is the real part of the DFT. Let's use K frequency components.

    n_arr = np.arange(N)
    K_freq = 20  # number of frequency components

    # Precompute Fourier basis (real and imaginary parts)
    fourier_basis = np.zeros((2 * K_freq, N))
    for k in range(1, K_freq + 1):
        fourier_basis[2 * (k - 1)] = np.cos(2 * np.pi * k * n_arr / N)
        fourier_basis[2 * (k - 1) + 1] = np.sin(2 * np.pi * k * n_arr / N)
    # Add DC component
    fourier_basis[0] = 1.0  # DC
    # Actually let's redo: index 0 = DC, then pairs for k=1..K_freq-1
    K_total = 1 + 2 * (K_freq - 1)  # DC + 2*(K_freq-1)
    fourier_basis = np.zeros((K_total, N))
    fourier_basis[0] = 1.0
    for k in range(1, K_freq):
        fourier_basis[2 * (k - 1) + 1] = np.cos(2 * np.pi * k * n_arr / N)
        fourier_basis[2 * (k - 1) + 2] = np.sin(2 * np.pi * k * n_arr / N)

    def coeffs_to_h_fourier(coeffs):
        h = np.zeros(N)
        for k in range(min(len(coeffs), K_total)):
            h += coeffs[k] * fourier_basis[k]
        return h

    def evaluate_coeffs_fourier(coeffs):
        h = coeffs_to_h_fourier(coeffs)
        return evaluate(h)

    rng = np.random.RandomState(42)

    # --- Global search over Fourier coefficients using differential evolution ---
    if time.time() - start_time < DEADLINE - 40:
        try:
            bounds = [(-3.0, 3.0)] * K_total
            bounds[0] = (1.0, 5.0)  # DC component must keep sum nonzero

            def de_objective(coeffs):
                h = coeffs_to_h_fourier(coeffs)
                val, valid = evaluate(h)
                if not valid:
                    return 1e10
                return val

            de_result = differential_evolution(
                de_objective,
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=12,
                tol=1e-6,
                callback=lambda xk, convergence: (
                    time.time() - start_time > DEADLINE - 45
                )
            )

            h_de = coeffs_to_h_fourier(de_result.x)
            val_de, valid_de = evaluate(h_de)
            if valid_de and val_de < best_value:
                best_value = val_de
                best_h = h_de.copy()
                print(f"DE Fourier found: {val_de:.6f}")
        except Exception as e:
            print(f"DE Fourier failed: {e}")

    # --- Smooth objective with log-sum-exp for local optimization ---
    BETA = 300.0

    def smooth_objective_and_grad(x):
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        max_shifted = np.max(shifted)
        if max_shifted > 500:
            return 1e10, np.zeros_like(h)

        beta = BETA
        scaled = beta * (shifted - max_shifted)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        if sum_exp < 1e-300:
            return 1e10, np.zeros_like(h)

        lse = max_c + max_shifted + np.log(sum_exp) / beta
        weights = exp_scaled / sum_exp

        # Gradient of LSE w.r.t. h: d_lse/dh[i] = 2 * sum_m weights[i+m] * h[m]
        idx = np.arange(N)
        M = weights[idx[:, None] + idx[None, :]]
        grad_lse = 2.0 * (M @ h)

        f_val = 2.0 * N * lse / (s * s)
        if not np.isfinite(f_val):
            return 1e10, np.zeros_like(h)

        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)

        if f_val >= 0:
            grad = df_dh
        else:
            grad = -df_dh

        return abs(f_val), grad

    def smooth_objective(x):
        val, _ = smooth_objective_and_grad(x)
        return val

    def smooth_grad(x):
        _, grad = smooth_objective_and_grad(x)
        return grad

    # --- Generate diverse initial points ---
    initial_points = []
    initial_points.append(np.ones(N).astype(float))

    # Random
    for _ in range(10):
        initial_points.append(rng.uniform(-1, 1, N))

    # Fourier-synthesized with flat-ish spectrum
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
        initial_points.append(np.cos(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * (freq + 1) * t))

    # Multi-frequency with random phases (aim for flat spectrum)
    for n_comp in [5, 8, 12, 15]:
        for _ in range(3):
            freqs = rng.choice(np.arange(1, 30), size=n_comp, replace=False)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)

    # Bimodal
    for _ in range(6):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)

    # Sort by initial value
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # --- Optimize from each initial point using L-BFGS-B ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE - 5:
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
                options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-6}
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

    # --- Final polish with exact objective ---
    if time.time() - start_time < DEADLINE - 3:
        h_curr = best_h.copy()

        def exact_objective(h):
            s = np.sum(h)
            if abs(s) < 1e-12:
                return 1e10
            conv = np.convolve(h, h)
            max_conv = np.max(conv)
            val = abs(2.0 * N * max_conv / (s * s))
            return val if np.isfinite(val) else 1e10

        try:
            result = minimize(
                exact_objective,
                h_curr,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-8}
            )
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # --- Perturbation phase ---
    if time.time() - start_time < DEADLINE - 2:
        for _ in range(6):
            if time.time() - start_time > DEADLINE - 2:
                break
            h_pert = best_h + 0.05 * rng.normal(0, 1, N)
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
                        options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-6}
                    )
                    h_opt = result.x
                    val_opt, valid = evaluate(h_opt)
                    if valid and val_opt < best_value:
                        best_value = val_opt
                        best_h = h_opt.copy()
                except Exception:
                    pass

    return best_h
