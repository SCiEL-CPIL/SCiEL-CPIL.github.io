"""
Combined Strategy: Smooth Surrogate Gradient Descent + Structured Local Search

The current program relies on Gerchberg-Saxton (GS) phase retrieval, which can get
stuck in local minima or converge slowly due to the hard ±1 constraint.
The inspiration provides a gradient-based approach using a smooth surrogate
(soft-max) for the non-smooth max absolute value objective.

This combined program:
1. Uses a fast FFT-based autoconvolution calculation.
2. Implements a smooth objective function using soft-max to approximate the
   maximum absolute autoconvolution side-lobe.
3. Computes the analytical gradient of this smooth objective with respect to the
   sequence elements.
4. Uses a projected gradient descent (or quasi-Newton like L-BFGS logic, but
   simplified to gradient steps with momentum/adaptive step size for robustness)
   on a continuous relaxation of the problem, followed by projection to ±1.
   Alternatively, and more robustly for this specific combinatorial problem,
   we use the gradient to guide a local search (Simulated Annealing or greedy
   flips) by identifying which bits are most likely to improve the objective.
   
   Actually, the most effective "structural change" suggested by the inspiration
   is to use the gradient information to perform a smart local search or to
   optimize the continuous variables and then round.
   
   Let's refine: The objective is non-smooth. Soft-max makes it smooth.
   We can optimize the continuous vector `y` (where `x = sign(y)`) to minimize
   the soft-max objective. However, the landscape of `sign(y)` is flat almost
   everywhere.
   
   Better approach: Use the gradient of the soft-max objective with respect to
   the *continuous* values to determine the "force" on each bit.
   If we treat the problem as optimizing over real numbers `z` and then
   thresholding, we can use gradient descent on `z`.
   
   However, a simpler and often more effective method for ±1 sequences is to
   use the gradient to guide **Coordinate Descent** or **Simulated Annealing**.
   Specifically, we can compute the gradient `g` of the smooth objective.
   Then, for each bit `k`, the change in objective if we flip `x[k]` is
   approximately `-2 * x[k] * g[k]` (since `dx[k] = -2x[k]` when flipping).
   We can use this to prioritize flips in a local search.
   
   Let's implement:
   1. Fast objective and gradient calculation (soft-max).
   2. A local search loop that uses the gradient to select the most promising
      flips, combined with random restarts and a deadline guard.
   3. We will also try a few steps of "continuous gradient descent" on a
      relaxed variable `y` where `x = sign(y)`, updating `y -= lr * grad`,
      then re-projecting. This is similar to the GS idea but guided by
      gradient rather than spectral constraints.
"""

import numpy as np
import time

N = 400
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _autoconv_fft(x):
    """Compute linear autoconvolution using FFT for speed."""
    L = 2 * N - 1
    L_fft = 1
    while L_fft < L:
        L_fft *= 2
    X = np.fft.rfft(x, n=L_fft)
    # Note: Using X * X gives circular convolution of x with x, which is
    # equivalent to linear autoconvolution if padded correctly.
    # For real x, X * conj(X) is power spectrum.
    # Autoconvolution C[k] = sum x[n] x[n+k].
    # In frequency domain, this corresponds to IFFT(|X|^2) for circular.
    # With zero padding to L_fft >= 2N-1, it becomes linear.
    C = np.fft.irfft(X * np.conj(X), n=L_fft)[:L]
    return C

def _objective_and_grad(x, tau=0.5):
    """
    Compute the smooth objective (soft-max of |ac| / sum^2) and its gradient.
    We exclude lag 0 (index 0) from the max calculation, as discussed in the
    current program's analysis.
    """
    s = np.sum(x)
    if abs(s) < 1e-10:
        return np.inf, np.zeros_like(x)
    
    ac = _autoconv_fft(x)
    # Exclude lag 0
    ac_side = ac[1:]
    abs_ac = np.abs(ac_side)
    
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(x)
    
    # Soft-max approximation
    shifted = (abs_ac - m) / tau
    shifted = np.clip(shifted, -50, 0)
    weights = np.exp(shifted)
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        val = m / (s * s)
        return val, np.zeros_like(x)
    
    weights /= w_sum
    soft_max = m + tau * np.log(w_sum)
    
    val = soft_max / (s * s)
    
    # Gradient computation
    # v[i] = w_i * sign(ac_side[i])
    # Note: ac_side corresponds to lags 1 to 2N-2.
    # The full ac array has indices 0 to 2N-2.
    # ac_side[i] corresponds to ac[i+1].
    
    # Let's map the weights back to the full ac array indices for gradient calc
    # or just work with the side lobes.
    # d(soft_max)/dx_k = sum_i w_i * sign(ac[i+1]) * (d(ac[i+1])/dx_k)
    # d(ac[l])/dx_k = x[l-k] + x[k-l] (with bounds)
    
    # Let's construct the gradient efficiently.
    # v_full = zeros(2N-1)
    # v_full[1:] = weights * sign(ac_side)
    
    v_full = np.zeros(2 * N - 1)
    v_full[1:] = weights * np.sign(ac_side)
    
    # grad_k = sum_{i=0}^{2N-2} v_full[i] * (x[i-k] + x[k-i])
    # This is a convolution-like operation.
    # Let's use FFT for this too, or a direct loop if N is small enough.
    # N=400, 2N-1=799.
    # Direct loop over k (400) and inner sum (up to 799) is 400*799 ~ 320K ops.
    # This is very fast in NumPy if vectorized, but a Python loop is slow.
    # Let's use FFT.
    
    # We want g[k] = sum_i v_full[i] * x[i-k] + sum_i v_full[i] * x[k-i]
    # Let A[k] = sum_i v_full[i] * x[i-k]
    # Let B[k] = sum_i v_full[i] * x[k-i]
    
    # A[k] is cross-correlation of v_full and x.
    # B[k] is cross-correlation of v_full and x, but with indices flipped?
    # B[k] = sum_i v_full[i] * x[k-i] = sum_j v_full[k-j] * x[j]
    # This is convolution of v_full and x at index k?
    # (v_full * x)[k] = sum_i v_full[i] x[k-i]. Yes, B is convolution.
    # A[k] = sum_i v_full[i] x[i-k] = sum_j v_full[j+k] x[j].
    # This is cross-correlation.
    
    L_fft = 1
    L_total = 2 * N - 1
    while L_fft < L_total + N - 1: # Need enough room for convolution
        L_fft *= 2
    
    V = np.fft.rfft(v_full, n=L_fft)
    X = np.fft.rfft(x, n=L_fft)
    
    # B[k] = ifft(V * X)[k]
    B = np.fft.irfft(V * X, n=L_fft)[:N]
    
    # A[k] = ifft(V * conj(X))[k] ?
    # Cross-correlation c[k] = sum x[i] y[i-k].
    # Here we want sum v[i] x[i-k].
    # This is ifft(FFT(v) * conj(FFT(x)))?
    # Let's check: ifft(FFT(v) * conj(FFT(x))) = v * x (correlation).
    # c[k] = sum_i v[i] x[i-k].
    # So A = ifft(V * conj(X)).
    A = np.fft.irfft(V * np.conj(X), n=L_fft)[:N]
    
    grad_soft_max = A + B
    
    # Gradient of objective val = soft_max / s^2
    # d(val)/dx_k = (grad_soft_max[k] * s^2 - soft_max * 2 * s) / s^4
    #             = grad_soft_max[k] / s^2 - 2 * soft_max / s^3
    
    grad_obj = grad_soft_max / (s * s) - 2 * soft_max / (s * s * s)
    
    return val, grad_obj

def construct():
    start_time = time.time()
    
    # Initialize with a good seed: All ones is the seed program.
    # But we know all ones has high side lobes.
    # Let's start with a random sequence or a structured one.
    # The seed program returns all ones.
    # We want to improve upon it.
    
    best_x = np.ones(N)
    best_val, _ = _objective_and_grad(best_x)
    
    # If the objective is inf (sum=0), we need to avoid that.
    # All ones has sum N, so it's fine.
    
    # Strategy:
    # 1. Gradient Descent on continuous variables y, projecting to x = sign(y).
    # 2. Local Search using gradient to guide flips.
    
    # Let's try a hybrid:
    # Use random restarts.
    # For each restart, run a few steps of continuous gradient descent,
    # then a local search phase.
    
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Initialize y (continuous)
        # Start from current best or random?
        # Let's try random first, then maybe perturb best.
        if restart == 0:
            y = np.ones(N)
        else:
            y = rng.choice([-1, 1], size=N).astype(float)
        
        x = np.sign(y)
        x[x == 0] = 1
        
        curr_val, curr_grad = _objective_and_grad(x)
        
        if curr_val < best_val:
            best_val = curr_val
            best_x = x.copy()
        
        # Phase 1: Continuous Gradient Descent
        # Update y, project to x.
        lr = 0.1
        for step in range(20):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Compute gradient at current x
            val, grad = _objective_and_grad(x)
            
            if val < best_val:
                best_val = val
                best_x = x.copy()
            
            # Update y
            # We want to minimize val.
            # y_new = y - lr * grad
            # But grad is w.r.t x.
            # Since x = sign(y), the gradient flow is tricky.
            # Approximation: move y in the direction that would decrease val
            # if x were continuous.
            y = y - lr * grad
            x_new = np.sign(y)
            x_new[x_new == 0] = 1
            
            if np.array_equal(x_new, x):
                # No change, try larger step or break
                # Let's just break if stuck for a while, or try random perturbation
                # For simplicity, break and go to local search
                break
            
            x = x_new
            lr *= 0.9 # Decay learning rate
        
        # Phase 2: Local Search (Greedy/SA)
        # Use gradient to find best flip.
        for ls_iter in range(100):
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            val, grad = _objective_and_grad(x)
            
            if val < best_val:
                best_val = val
                best_x = x.copy()
            
            # Find the bit that gives the largest improvement
            # Approximate change in val when flipping x[k]:
            # delta_val ≈ -2 * x[k] * grad[k]
            # We want to minimize val, so we want delta_val < 0.
            # So we want -2 * x[k] * grad[k] < 0 => x[k] * grad[k] > 0.
            # The magnitude of improvement is proportional to |x[k] * grad[k]|.
            
            # Calculate scores for all flips
            # score[k] = -x[k] * grad[k]
            # If score[k] > 0, flipping x[k] decreases the objective (approx).
            # We pick the k with the highest score[k].
            
            scores = -x * grad
            
            # Add some noise to avoid local minima (Simulated Annealing style)
            # Or just pick the best if any positive, else random.
            best_k = np.argmax(scores)
            
            if scores[best_k] > 0:
                x[best_k] *= -1
            else:
                # No improving move found.
                # Try a random move with low probability (annealing)
                # Or break.
                # Let's try a random move to escape local minima.
                if rng.random() < 0.1:
                    k = rng.integers(0, N)
                    x[k] *= -1
                else:
                    break
        
    return best_x

# Ensure the function is named construct and returns the sequence
# The evaluator expects construct() -> sequence

if __name__ == "__main__":
    # Simple test
    x = construct()
    val, _ = _objective_and_grad(x)
    print(f"Final Value: {val}")
    print(f"Sum: {np.sum(x)}")
