"""Optimal flat +/-1 polynomial for N=24 (degree 23).

Uses a continuous convex relaxation (log-sum-exp surrogate for sup-norm)
solved via L-BFGS-B to guide the discrete search. This helps escape
local minima that trap simple greedy local search.
"""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - 5.0 - (time.time() - _START)  # 5 s safety margin


def _sup_norm_fast(coeffs, n_grid=2048):
    """Fast sup |P(e^{i theta})| over the unit circle via FFT."""
    L = 1 << (len(coeffs) - 1).bit_length()
    if L < n_grid:
        L = 1 << (n_grid - 1).bit_length()
    a = np.zeros(L, dtype=np.complex128)
    a[:len(coeffs)] = coeffs
    vals = np.fft.fft(a)
    return float(np.max(np.abs(vals)))


def _log_sum_exp_surrogate(coeffs, beta=100.0, z_points=None):
    """
    Compute the log-sum-exp surrogate of the sup-norm.
    f(c) = log(sum(exp(beta * |P(z_j)|)) / M)
    This is a smooth approximation to max(|P(z_j)|).
    """
    if z_points is None:
        z_points = np.exp(2j * np.pi * np.arange(2048) / 2048)
    
    # P(z) = sum c_k z^k
    # coeffs shape (N,), z_points shape (M,)
    # P shape (M,)
    # Use matrix multiplication or broadcasting
    # z_points[:, None] ** np.arange(N)[None, :] is expensive.
    # Better: FFT based evaluation is faster but log-sum-exp needs individual values.
    # We can compute P(z_j) for all j using FFT if we zero-pad.
    # But we need |P(z_j)| for specific z_j.
    # Let's stick to direct evaluation if M is small, or use FFT for all points.
    # Actually, for optimization, we need gradients.
    # Gradient of |P(z)| wrt c_k is Re(conj(P(z)) * z^k).
    # Gradient of log-sum-exp is weighted average of gradients.
    
    # For simplicity and speed, let's evaluate on a grid using FFT.
    # The grid points are z_j = exp(-2pi i j / L).
    # This matches the FFT definition.
    L = 2048
    a = np.zeros(L, dtype=np.complex128)
    a[:len(coeffs)] = coeffs
    vals = np.fft.fft(a)
    abs_vals = np.abs(vals)
    
    # Log-sum-exp for stability
    max_val = np.max(abs_vals)
    if max_val == 0:
        return 0.0, np.zeros_like(coeffs)
    
    # f = max_val + log(sum(exp(beta * (abs_vals - max_val)))) / M
    # Wait, the surrogate is log(sum(exp(beta * x))) / M.
    # This approximates (1/beta) * log(sum(exp(beta * x))) -> max(x).
    # So we want to minimize this.
    
    shifted = beta * (abs_vals - max_val)
    # exp(shifted) <= 1
    log_sum = np.log(np.sum(np.exp(shifted)))
    val = (max_val + log_sum / beta) # This is approx max. 
    # Actually, standard LogSumExp(x) = max(x) + log(sum(exp(x - max(x))))
    # We want min max(x).
    # So objective is LogSumExp(beta * abs_vals) / beta.
    
    obj = (max_val + np.log(np.sum(np.exp(shifted))) / beta)
    
    # Gradient
    # d/dx log(sum(exp(beta x))) = beta * softmax(beta x)
    # So grad of obj wrt abs_vals is softmax(beta * abs_vals)
    weights = np.exp(shifted) / np.sum(np.exp(shifted))
    
    # Gradient of abs_vals wrt coeffs
    # abs_vals[j] = |P(z_j)|
    # d|P|/dc_k = Re(conj(P(z_j)) z_j^k)
    # We have P(z_j) = vals[j]
    # z_j = exp(-2pi i j / L)
    # z_j^k = exp(-2pi i j k / L)
    # This is the inverse FFT matrix essentially.
    # grad_coeff[k] = sum_j weights[j] * Re(conj(vals[j]) * exp(-2pi i j k / L))
    # This is Re( IFFT(weights * conj(vals)) )?
    # Let's check.
    # IFFT(y)_k = (1/L) sum_j y_j exp(2pi i j k / L)
    # We have exp(-2pi i j k / L).
    # So it's conjugate of IFFT.
    # Let Y = weights * conj(vals)
    # IFFT(Y)_k = (1/L) sum_j weights[j] conj(vals[j]) exp(2pi i j k / L)
    # Conj(IFFT(Y))_k = (1/L) sum_j weights[j] vals[j] exp(-2pi i j k / L)
    # We want sum_j weights[j] Re(conj(vals[j]) exp(-2pi i j k / L))
    # = Re( sum_j weights[j] conj(vals[j]) exp(-2pi i j k / L) )
    # This is not exactly IFFT.
    
    # Let's compute gradient directly for small N=24.
    # k in 0..23, j in 0..2047.
    # grad[k] = sum_j weights[j] * Re(conj(vals[j]) * z_j^k)
    
    # Precompute z_powers?
    # z_j = exp(-2pi i j / L)
    # z_j^k = exp(-2pi i j k / L)
    
    # Vectorized:
    # j_idx = np.arange(L)
    # k_idx = np.arange(len(coeffs))
    # exponents = -2j * np.pi * np.outer(j_idx, k_idx) / L
    # z_powers = np.exp(exponents) # Shape (L, N)
    
    # This is 2048 * 24 complex numbers, ~49k, fine.
    
    j_idx = np.arange(L)
    k_idx = np.arange(len(coeffs))
    exponents = -2j * np.pi * np.outer(j_idx, k_idx) / L
    z_powers = np.exp(exponents)
    
    # grad_k = sum_j weights[j] * Re(conj(vals[j]) * z_powers[j, k])
    # conj_vals = np.conj(vals)
    # term = weights[:, None] * conj_vals[:, None] * z_powers
    # grad = np.real(np.sum(term, axis=0))
    
    conj_vals = np.conj(vals)
    # weights shape (L,), conj_vals shape (L,), z_powers shape (L, N)
    # term[j, k] = weights[j] * conj_vals[j] * z_powers[j, k]
    # grad[k] = sum_j term[j, k]
    grad = np.real(np.sum(weights[:, None] * conj_vals[:, None] * z_powers, axis=0))
    
    return obj, grad


def _continuous_relaxation(initial_coeffs, max_iter=1000):
    """
    Solve the continuous relaxation c_k in [-1, 1] using L-BFGS-B.
    """
    n = len(initial_coeffs)
    x0 = np.asarray(initial_coeffs, dtype=np.float64)
    
    def fun(x):
        val, grad = _log_sum_exp_surrogate(x, beta=50.0)
        return val, grad
        
    bounds = [(-1.0, 1.0) for _ in range(n)]
    
    res = minimize(fun, x0, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-9})
    return res.x


def _local_search(coeffs, max_outer=1000):
    """1-flip greedy local search minimizing sup-norm."""
    coeffs = np.asarray(coeffs, dtype=np.float64).copy()
    best = coeffs.copy()
    best_val = _sup_norm_fast(best)
    for _ in range(max_outer):
        if _remaining() < 2.0:
            break
        improved = False
        for j in range(len(coeffs)):
            if _remaining() < 1.0:
                break
            coeffs[j] = -coeffs[j]
            v = _sup_norm_fast(coeffs)
            if v < best_val - 1e-13:
                best_val = v
                best = coeffs.copy()
                improved = True
            else:
                coeffs[j] = -coeffs[j]
        if not improved:
            break
    return best, best_val


def construct():
    # Known optimal +/-1 sequence for N=24 (from previous attempts/tables)
    # This might be a local minimum. We will use it as a seed.
    seq = [1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1,
           1, -1, 1, 1, -1, 1, -1, 1]
    arr = np.array(seq, dtype=np.float64)
    
    best_arr = arr.copy()
    best_val = _sup_norm_fast(best_arr)
    
    # 1. Continuous Relaxation to find a better starting point
    if _remaining() > 10.0:
        try:
            # Run relaxation from the known sequence
            cont_sol = _continuous_relaxation(arr, max_iter=200)
            # Round to nearest +/-1
            disc_sol = np.sign(cont_sol)
            # Handle zeros (unlikely but possible)
            disc_sol[disc_sol == 0] = 1.0
            
            v_disc = _sup_norm_fast(disc_sol)
            if v_disc < best_val:
                best_val = v_disc
                best_arr = disc_sol
        except Exception:
            pass

    # 2. Local Search from the best candidate found so far
    if _remaining() > 5.0:
        try:
            refined, val = _local_search(best_arr, max_outer=500)
            if val < best_val:
                best_val = val
                best_arr = refined
        except Exception:
            pass

    # 3. Try a few random restarts if time permits
    if _remaining() > 20.0:
        for _ in range(5):
            if _remaining() < 5.0:
                break
            try:
                # Random start
                rand_arr = np.random.choice([-1.0, 1.0], size=len(arr))
                # Relax
                cont_sol = _continuous_relaxation(rand_arr, max_iter=100)
                disc_sol = np.sign(cont_sol)
                disc_sol[disc_sol == 0] = 1.0
                
                v_disc = _sup_norm_fast(disc_sol)
                if v_disc < best_val:
                    best_val = v_disc
                    best_arr = disc_sol
                    # Polish
                    if _remaining() > 2.0:
                        refined, val = _local_search(best_arr, max_outer=100)
                        if val < best_val:
                            best_val = val
                            best_arr = refined
            except Exception:
                pass

    return [int(round(x)) for x in best_arr]
