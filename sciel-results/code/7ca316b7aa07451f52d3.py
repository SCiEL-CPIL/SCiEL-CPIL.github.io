"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def compute_autocorr(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        R = np.zeros(N)
        R[0] = acf[0]
        R[1:N] = acf[1:N]
        return R

    def get_max_autocorr(R):
        return np.max(np.abs(R[1:N]))

    def soft_max_sidelobe(R, beta=10.0):
        """Differentiable approximation of max|sidelobes| using soft-max."""
        abs_R = np.abs(R[1:N])
        # Soft max: (1/beta) * log(sum(exp(beta * x)))
        max_val = np.max(abs_R)
        shifted = abs_R - max_val  # For numerical stability
        return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta

    def continuous_objective(x, beta=10.0):
        """
        Objective function for continuous relaxation.
        x: real-valued vector of length N
        Returns: soft max of sidelobes of autocorrelation of tanh(x)
        """
        h = np.tanh(x)
        R = compute_autocorr(h)
        return soft_max_sidelobe(R, beta)

    def continuous_gradient(x, beta=10.0):
        """
        Gradient of the continuous objective using automatic differentiation via finite differences
        or analytical derivation. For simplicity, use numerical gradient.
        """
        # Use finite differences for gradient
        eps = 1e-6
        grad = np.zeros(N)
        for i in range(N):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (continuous_objective(x_plus, beta) - continuous_objective(x_minus, beta)) / (2 * eps)
        return grad

    def discrete_polish(h, R, current_val, max_passes=100, time_limit=None):
        """Greedy local search with incremental O(N) updates for discrete ±1 values."""
        if time_limit is None:
            time_limit = deadline - 5.0
        h = h.copy()
        R = R.copy()
        k = np.arange(1, N)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > time_limit:
                break
            improved = False
            passes += 1
            order = np.random.permutation(N)
            for i in order:
                if time.time() > time_limit:
                    break
                # Delta for flipping h[i]
                idx1 = i + k
                valid1 = idx1 < N
                term1 = np.zeros(N-1)
                term1[valid1] = h[idx1[valid1]]
                idx2 = i - k
                valid2 = idx2 >= 0
                term2 = np.zeros(N-1)
                term2[valid2] = h[idx2[valid2]]
                delta = -2.0 * h[i] * (term1 + term2)
                R[1:N] += delta
                new_val = get_max_autocorr(R)
                if new_val < current_val:
                    h[i] *= -1.0
                    current_val = new_val
                    improved = True
                else:
                    R[1:N] -= delta
        return h, R, current_val

    def continuous_optimize(x0, beta=10.0, maxiter=1000, time_limit=None):
        """
        Run L-BFGS-B on the continuous relaxation.
        Returns: optimized x, h=tanh(x), R, val
        """
        if time_limit is None:
            time_limit = deadline - 10.0
        
        def objective_and_grad(x):
            obj = continuous_objective(x, beta)
            grad = continuous_gradient(x, beta)
            return obj, grad
        
        # Use L-BFGS-B with callback to check time
        n_evals = [0]
        def callback(xk):
            n_evals[0] += 1
            if time.time() > time_limit:
                raise ValueError("Time limit exceeded")
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter, 'maxfun': 2000, 'callback': callback}
            )
        except (ValueError, Exception):
            # If time limit exceeded, use the last known good point
            result = None
        
        if result is not None and result.success:
            x_opt = result.x
        else:
            x_opt = x0
        
        h = np.tanh(x_opt)
        R = compute_autocorr(h)
        val = get_max_autocorr(R)
        return x_opt, h, R, val

    def generate_initialization(method='random', rng=None):
        """Generate initial real-valued vector for continuous optimization."""
        if rng is None:
            rng = np.random.RandomState()
        
        if method == 'random':
            return rng.randn(N)
        elif method == 'zero':
            return np.zeros(N)
        elif method == 'small_random':
            return 0.1 * rng.randn(N)
        else:
            return rng.randn(N)

    rng = np.random.RandomState(42)
    best_h = None
    best_R = None
    best_val = float('inf')
    best_x = None

    # Phase 1: Multiple random initializations with continuous optimization
    n_restarts = 5
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Different initialization strategies
        if r == 0:
            x0 = generate_initialization('random', rng)
        elif r == 1:
            x0 = generate_initialization('small_random', rng)
        elif r == 2:
            x0 = generate_initialization('zero', rng) + 0.5 * rng.randn(N)
        else:
            x0 = generate_initialization('random', rng)
        
        # Run continuous optimization with different beta values
        beta = 8.0 if r % 2 == 0 else 12.0
        x_opt, h, R, val = continuous_optimize(x0, beta=beta, maxiter=500, time_limit=deadline - 15.0)
        
        # Discretize: h is already in [-1,1] from tanh, but we need ±1
        h_discrete = np.sign(h)
        # Handle zeros: assign +1
        h_discrete[h_discrete == 0] = 1.0
        
        R_discrete = compute_autocorr(h_discrete)
        val_discrete = get_max_autocorr(R_discrete)
        
        # Also try polishing the continuous result directly (keep as ±1 by thresholding)
        h_polished, R_polished, val_polished = discrete_polish(h_discrete, R_discrete, val_discrete, max_passes=50, time_limit=deadline - 10.0)
        
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished.copy()
            best_R = R_polished.copy()
            best_x = x_opt.copy()

    # Phase 2: Refine with more aggressive continuous optimization if time permits
    if best_h is not None and time.time() < deadline - 20.0:
        # Use the best x as starting point for further refinement
        x_refine = best_x.copy() if best_x is not None else rng.randn(N)
        
        # Try with higher beta for sharper max approximation
        x_opt2, h2, R2, val2 = continuous_optimize(x_refine, beta=15.0, maxiter=300, time_limit=deadline - 10.0)
        
        h2_discrete = np.sign(h2)
        h2_discrete[h2_discrete == 0] = 1.0
        
        R2_discrete = compute_autocorr(h2_discrete)
        val2_discrete = get_max_autocorr(R2_discrete)
        
        h2_polished, R2_polished, val2_polished = discrete_polish(h2_discrete, R2_discrete, val2_discrete, max_passes=100, time_limit=deadline - 5.0)
        
        if val2_polished < best_val:
            best_val = val2_polished
            best_h = h2_polished.copy()
            best_R = R2_polished.copy()

    # Phase 3: Final discrete polishing
    if best_h is not None and time.time() < deadline - 5.0:
        h_final, R_final, val_final = discrete_polish(best_h, best_R, best_val, max_passes=200, time_limit=deadline - 2.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final.copy()

    # Fallback: if something went wrong, return a simple random sequence
    if best_h is None:
        best_h = rng.choice([1, -1], N).astype(np.float64)

    return best_h
# EVOLVE-BLOCK-END
