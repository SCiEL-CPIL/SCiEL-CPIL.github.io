import time
import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    def _elapsed():
        return time.time() - start_time

    def _time_left():
        return deadline - _elapsed()

    def _autoconv(h):
        """Compute discrete autoconvolution of h (length N) -> length 2N-1."""
        return np.convolve(h, h, mode='full')

    def _ratio(h):
        """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) with L1 normalization."""
        s = np.sum(h)
        if s <= 1e-15:
            return 0.0
        h_norm = h / s
        q = np.sum(h_norm ** 2)
        conv = _autoconv(h_norm)
        m = np.max(conv)
        if m <= 1e-15:
            return 0.0
        return q / m

    def _ratio_neg(h):
        """Negative ratio for minimization."""
        return -_ratio(h)

    def _dft_parameterize(z):
        """Convert complex DFT coefficients z to non-negative time-domain h with sum=1."""
        # z has length N, representing DFT of h
        # h = IDFT(z), then take real part and clip to non-negative
        h = np.real(np.fft.ifft(z))
        h = np.maximum(h, 0.0)
        s = np.sum(h)
        if s <= 1e-15:
            return np.ones(N) / N
        return h / s

    def _optimize_from_z(z0):
        """Optimize from initial DFT parameterization z0."""
        if _time_left() < 5.0:
            return None

        def objective(z):
            h = _dft_parameterize(z)
            return _ratio_neg(h)

        # Split z into real and imaginary parts for optimization
        def obj_flat(z_flat):
            z = z_flat[:N] + 1j * z_flat[N:]
            h = _dft_parameterize(z)
            return _ratio_neg(h)

        # Initial point
        z0_real = np.real(z0)
        z0_imag = np.imag(z0)
        x0 = np.concatenate([z0_real, z0_imag])

        # Constraints: none needed since we clip to non-negative
        result = minimize(
            obj_flat,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )

        if result.success or result.fun < objective(z0):
            z_opt = result.x[:N] + 1j * result.x[N:]
            h_opt = _dft_parameterize(z_opt)
            return h_opt

        return None

    def _local_optimize(h0):
        """Local optimization from time-domain h0 using log-parameterization."""
        if _time_left() < 5.0:
            return None

        # Use log-parameterization: h_i = exp(x_i), then normalize
        # Only optimize over positive entries
        pos_mask = h0 > 1e-10
        if np.sum(pos_mask) < 2:
            return None

        def obj_log(x):
            h = np.zeros(N)
            h[pos_mask] = np.exp(x)
            s = np.sum(h)
            if s <= 1e-15:
                return 0.0
            h_norm = h / s
            return _ratio_neg(h_norm)

        x0 = np.log(h0[pos_mask] + 1e-15)

        result = minimize(
            obj_log,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
        )

        if result.success:
            h_opt = np.zeros(N)
            h_opt[pos_mask] = np.exp(result.x)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                return h_opt

        return None

    def _find_sidon_set(k, N):
        """Find a Sidon set of size k in [0, N)."""
        best_set = None
        best_score = -1

        # Try multiple random initializations
        for trial in range(50):
            if _time_left() < 3.0:
                break

            # Start with random set
            current = np.random.choice(N, k, replace=False)

            # Local search: try swapping elements
            improved = True
            while improved and _time_left() > 2.0:
                improved = False
                for i in range(k):
                    for j in range(N):
                        if j not in current:
                            new_set = list(current)
                            new_set[i] = j
                            new_set = np.array(sorted(new_set))

                            # Compute score for this set
                            h = np.zeros(N)
                            h[new_set] = 1.0 / k
                            sc = _ratio(h)

                            if sc > best_score:
                                best_score = sc
                                best_set = new_set.copy()

                            # Check if this improves over current
                            h_cur = np.zeros(N)
                            h_cur[current] = 1.0 / k
                            sc_cur = _ratio(h_cur)

                            if sc > sc_cur + 1e-10:
                                current = new_set
                                improved = True
                                break
                    if improved:
                        break

        return best_set

    # Generate initial candidates
    candidates = []

    # 1. Uniform
    candidates.append(np.ones(N) / N)

    # 2. Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for k in range(4, min(len(primes), 15) + 1):
        h = np.zeros(N)
        h[primes[:k]] = 1.0 / k
        candidates.append(h)

    # 3. Sidon sets
    for k in [5, 7, 9, 11]:
        sidon = _find_sidon_set(k, N)
        if sidon is not None:
            h = np.zeros(N)
            h[sidon] = 1.0 / k
            candidates.append(h)

    # 4. Random sparse
    np.random.seed(42)
    for i in range(20):
        if _time_left() < 10.0:
            break
        k = np.random.randint(4, 16)
        idx = np.random.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0 / k
        candidates.append(h)

    # 5. Gaussian-like profiles
    for sigma in [3, 5, 7, 10]:
        x = np.arange(N)
        h = np.exp(-((x - N // 2) ** 2) / (2 * sigma ** 2))
        h = h / np.sum(h)
        candidates.append(h)

    # Evaluate all candidates
    best_h = candidates[0]
    best_score = _ratio(best_h)

    for h in candidates:
        if _time_left() < 10.0:
            break
        sc = _ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # Optimize the best candidates using DFT parameterization
    top_candidates = sorted(candidates, key=lambda h: _ratio(h), reverse=True)[:5]

    for h0 in top_candidates:
        if _time_left() < 15.0:
            break

        # Convert to DFT
        z0 = np.fft.fft(h0)

        # Optimize
        h_opt = _optimize_from_z(z0)
        if h_opt is not None:
            sc = _ratio(h_opt)
            if sc > best_score:
                best_score = sc
                best_h = h_opt.copy()

        # Also try local optimization in time domain
        h_local = _local_optimize(h0)
        if h_local is not None:
            sc = _ratio(h_local)
            if sc > best_score:
                best_score = sc
                best_h = h_local.copy()

    # Final local optimization on the best result
    if _time_left() > 10.0:
        h_final = _local_optimize(best_h)
        if h_final is not None:
            sc = _ratio(h_final)
            if sc > best_score:
                best_score = sc
                best_h = h_final.copy()

    return best_h.tolist()
# EVOLVE-BLOCK-END
