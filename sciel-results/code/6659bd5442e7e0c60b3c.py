import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140

    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def compute_ac(x):
        """Compute autoconvolution using FFT."""
        X = np.fft.fft(x, n=fft_size)
        ac = np.fft.ifft(X * np.conj(X))
        ac = np.real(ac)
        ac_linear = np.concatenate([ac[:N], ac[fft_size-N+1:fft_size]])
        return ac_linear

    def compute_score(x):
        S = np.sum(x)
        if abs(S) < 1e-10:
            return np.inf
        ac = compute_ac(x)
        max_ac = np.max(np.abs(ac))
        return max_ac / (S * S)

    def smooth_surrogate(x, temperature=0.01):
        """
        Smooth surrogate for max|ac| / S^2 using log-sum-exp.
        Returns (value, gradient) for L-BFGS-B.
        """
        S = np.sum(x)
        if abs(S) < 1e-10:
            return 1e10, np.zeros(N)
        
        ac = compute_ac(x)
        abs_ac = np.abs(ac)
        
        # Log-sum-exp for smooth max
        max_val = np.max(abs_ac)
        if max_val < 1e-10:
            return 0.0, np.zeros(N)
        
        # Soft max: sum(exp((abs_ac - max_val)/temp)) * max_val
        shifted = (abs_ac - max_val) / temperature
        log_sum_exp = max_val / temperature + temperature * np.log(np.sum(np.exp(shifted)))
        soft_max = log_sum_exp
        
        value = soft_max / (S * S)
        
        # Gradient via chain rule
        # d(soft_max)/d(ac_i) = exp((abs_ac_i - max_val)/temp) / sum(exp((abs_ac - max_val)/temp)) * sign(ac_i)
        exp_shifted = np.exp(shifted)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_ac = weights * np.sign(ac)
        
        # d(ac)/dx via FFT
        X = np.fft.fft(x, n=fft_size)
        # ac = ifft(X * conj(X)), so dac/dx_j involves cross terms
        # For gradient, we use the fact that ac_k = sum_i x_i x_{i+k}
        # This is complex; use numerical differentiation as fallback for gradient
        # Actually, let's compute gradient numerically for reliability
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            v_plus = compute_score(x_plus)
            x_minus = x.copy()
            x_minus[i] -= eps
            v_minus = compute_score(x_minus)
            grad[i] = (v_plus - v_minus) / (2 * eps)
        
        return value, grad

    best_score = np.inf
    best_x = np.ones(N)

    # First, evaluate the baseline
    baseline_score = compute_score(np.ones(N))
    best_score = baseline_score
    best_x = np.ones(N).copy()

    np.random.seed(42)

    # Local search with random restarts and L-BFGS-B polishing
    for restart in range(200):
        if time.time() > deadline - 15:
            break

        # Diverse initializations
        r = np.random.random()
        if r < 0.3:
            # Binary initialization
            x = np.random.choice([-1.0, 1.0], size=N)
        elif r < 0.6:
            # Gaussian initialization
            x = np.random.randn(N)
        else:
            # Sparse random
            x = np.zeros(N)
            n_nonzero = np.random.randint(N//4, N//2)
            indices = np.random.choice(N, n_nonzero, replace=False)
            x[indices] = np.random.choice([-1.0, 1.0], size=n_nonzero)
        
        if abs(np.sum(x)) < 1e-6:
            x[0] += 1.0

        current_score = compute_score(x)
        
        # Try L-BFGS-B with smooth surrogate
        if time.time() < deadline - 20:
            try:
                # Use a simpler smooth objective: sum of squared ac values / S^2
                # This is differentiable and related to the max
                def smooth_obj(v):
                    xx = v.reshape(N)
                    S = np.sum(xx)
                    if abs(S) < 1e-10:
                        return 1e10
                    ac = compute_ac(xx)
                    # Use sum of squares as smooth surrogate
                    return np.sum(ac * ac) / (S * S)

                def smooth_grad(v):
                    xx = v.reshape(N)
                    S = np.sum(xx)
                    if abs(S) < 1e-10:
                        return np.zeros(N)
                    ac = compute_ac(xx)
                    # Numerical gradient for reliability
                    eps = 1e-7
                    grad = np.zeros(N)
                    for i in range(N):
                        xx_plus = xx.copy()
                        xx_plus[i] += eps
                        S_plus = np.sum(xx_plus)
                        ac_plus = compute_ac(xx_plus)
                        v_plus = np.sum(ac_plus * ac_plus) / (S_plus * S_plus)
                        
                        xx_minus = xx.copy()
                        xx_minus[i] -= eps
                        S_minus = np.sum(xx_minus)
                        ac_minus = compute_ac(xx_minus)
                        v_minus = np.sum(ac_minus * ac_minus) / (S_minus * S_minus)
                        
                        grad[i] = (v_plus - v_minus) / (2 * eps)
                    return grad

                result = minimize(
                    smooth_obj, x.copy(), jac=smooth_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'gtol': 1e-6}
                )
                if result.success and result.fun < current_score:
                    x_opt = result.x.reshape(N)
                    opt_score = compute_score(x_opt)
                    if opt_score < current_score:
                        x = x_opt
                        current_score = opt_score
            except Exception:
                pass

        # Local refinement with random steps
        for it in range(50):
            if time.time() > deadline - 10:
                break

            i = np.random.randint(N)
            improved = False

            for delta in [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]:
                x_trial = x.copy()
                x_trial[i] += delta
                s = compute_score(x_trial)
                if s < current_score:
                    x = x_trial
                    current_score = s
                    improved = True
                    break

            if not improved and np.random.random() < 0.1:
                x[i] += np.random.randn()

            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()

        if time.time() > deadline - 15:
            break

    # Final polish: try L-BFGS-B on the best solution found
    if time.time() < deadline - 5:
        try:
            def smooth_obj2(v):
                xx = v.reshape(N)
                S = np.sum(xx)
                if abs(S) < 1e-10:
                    return 1e10
                ac = compute_ac(xx)
                return np.sum(ac * ac) / (S * S)

            def smooth_grad2(v):
                xx = v.reshape(N)
                S = np.sum(xx)
                if abs(S) < 1e-10:
                    return np.zeros(N)
                eps = 1e-7
                grad = np.zeros(N)
                for i in range(N):
                    xx_plus = xx.copy()
                    xx_plus[i] += eps
                    S_plus = np.sum(xx_plus)
                    ac_plus = compute_ac(xx_plus)
                    v_plus = np.sum(ac_plus * ac_plus) / (S_plus * S_plus)

                    xx_minus = xx.copy()
                    xx_minus[i] -= eps
                    S_minus = np.sum(xx_minus)
                    ac_minus = compute_ac(xx_minus)
                    v_minus = np.sum(ac_minus * ac_minus) / (S_minus * S_minus)

                    grad[i] = (v_plus - v_minus) / (2 * eps)
                return grad

            result = minimize(
                smooth_obj2, best_x.copy(), jac=smooth_grad2,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            if result.success:
                x_polished = result.x.reshape(N)
                polished_score = compute_score(x_polished)
                if polished_score < best_score:
                    best_score = polished_score
                    best_x = x_polished
        except Exception:
            pass

    return best_x
