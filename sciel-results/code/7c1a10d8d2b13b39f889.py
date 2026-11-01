import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autocorr_fft(x):
    """Compute linear autocorrelation of x using FFT."""
    n_fft = 2 * len(x)
    X = np.fft.fft(x, n=n_fft)
    P = X * np.conj(X)
    R = np.fft.ifft(P).real
    return R[:len(x)]

def _calc_score(h):
    """
    Calculate the score (upper bound on C3) for a given step function h.
    Score = max_{k != 0} |(h*h)_k| / ||h||_2^2
    """
    corr = _autocorr_fft(h)
    # corr[0] is lag 0, corr[1] is lag 1, ..., corr[N-1] is lag N-1
    # We want max |corr[k]| for k != 0
    if len(corr) > 1:
        peak = np.max(np.abs(corr[1:]))
    else:
        peak = 0.0
    norm = np.sum(h**2)
    if norm == 0:
        return 0.0
    return peak / norm

def _soft_max(x, beta=10.0):
    """
    Soft approximation of max(x) using log-sum-exp.
    x is an array of non-negative values (e.g., |corr|).
    """
    if len(x) == 0:
        return 0.0
    # Shift for numerical stability
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    # We want to approximate max. 
    # log(sum(exp(beta * (x - x_max)))) + x_max - (1/beta)*log(len(x)) is a common form?
    # Actually, standard soft max: (1/beta) * log(sum(exp(beta * x)))
    # For large beta, this approaches max(x).
    # To avoid overflow, subtract max.
    shifted = x - x_max
    sum_exp = np.sum(np.exp(beta * shifted))
    return x_max + (1.0 / beta) * np.log(sum_exp)

def _objective_and_grad(x, beta=10.0):
    """
    Objective function for continuous relaxation.
    x is a vector of length N.
    We minimize the soft-max of the autocorrelation sidelobes.
    """
    # Compute autocorrelation
    R = _autocorr_fft(x)
    # We care about |R[1:N]|
    # Since R is real, |R[k]| = sqrt(R[k]^2)
    # To make it differentiable, we can minimize sum(R[k]^2) or soft-max of |R[k]|.
    # Let's use soft-max of |R[k]|.
    # But |R[k]| is not differentiable at 0.
    # Alternative: minimize sum(R[k]^2) for k != 0. This is the energy of sidelobes.
    # Or minimize max(R[k]^2).
    # Let's try minimizing the soft-max of R[k]^2 for k=1..N-1.
    
    sidelobes_sq = R[1:]**2
    val = _soft_max(sidelobes_sq, beta)
    
    # Gradient calculation
    # d/dx (soft_max(y)) where y = R[1:]^2
    # This is complex. Let's use finite differences or derive.
    # Actually, for L-BFGS-B, we can just provide the function and let it estimate gradient,
    # or provide gradient. Providing gradient is faster.
    
    # Let's compute gradient numerically for simplicity and robustness in this context,
    # or use a simpler objective like sum of squares first to see if it works.
    # Given the time constraint, let's try a simpler approach:
    # Minimize sum_{k=1}^{N-1} R[k]^2. This is equivalent to minimizing the energy of the sidelobes.
    # This is differentiable and convex in the continuous domain? No, non-convex.
    # But it's smooth.
    
    # Let's switch to minimizing the sum of squares of sidelobes for the continuous relaxation.
    # Objective: sum_{k=1}^{N-1} (R[k])^2
    # R[k] = sum_i x[i] x[i+k]
    # This is a quadratic form.
    
    # Actually, the prompt strategy suggests "initializing... to directly enforce flat autocorrelation sidelobes".
    # Minimizing sum of squares is a good proxy for flatness.
    
    # Let's compute the value and gradient of sum_{k=1}^{N-1} R[k]^2.
    # Let S = sum_{k=1}^{N-1} R[k]^2.
    # R = ifft(|fft(x)|^2).real
    # This is getting complicated to derive gradient analytically quickly.
    # Let's use the finite difference gradient for the continuous relaxation phase,
    # or just use scipy's minimize with method='L-BFGS-B' and let it approximate,
    # or use 'BFGS' which approximates Hessian.
    # However, N=400, so gradient estimation is feasible if we are careful.
    
    # Let's stick to the value for now and use a simple finite difference gradient 
    # or rely on scipy's internal numerical gradient if we don't provide one.
    # But providing gradient is much faster.
    
    # Let's derive the gradient for sum_{k=1}^{N-1} R[k]^2.
    # R = IFFT(FFT(x) * conj(FFT(x)))
    # Let X = FFT(x).
    # R = IFFT(|X|^2)
    # S = sum_{k=1}^{N-1} R[k]^2
    # This is a 4th order polynomial in x.
    
    # Alternative: Use the property that sum_{k} R[k]^2 = (1/N) sum_{m} |FFT(R)[m]|^2 ? No.
    # Parseval: sum R[k]^2 = (1/N) sum |FFT(R)[m]|^2.
    # FFT(R) = |X|^2.
    # So S = (1/N) sum_{m=0}^{N-1} |X[m]|^4 - R[0]^2 ? No, R[0] is included in sum R[k]^2 if we sum all.
    # We sum k=1 to N-1.
    # S = (1/N) sum_{m} |X[m]|^4 - R[0]^2.
    # R[0] = sum x[i]^2.
    
    # So minimize f(x) = (1/N) sum_{m} |X[m]|^4 - (sum x[i]^2)^2.
    # This is still non-convex.
    
    # Let's just use the value `val` from soft-max or sum-squares and let scipy handle it.
    # To be safe and fast, I will provide the gradient for the sum-of-squares objective.
    
    # Objective: f(x) = sum_{k=1}^{N-1} R[k]^2
    # R = ifft(|fft(x)|^2).real
    # Let's compute R and then use automatic differentiation? No, too slow.
    # Let's use finite differences for the continuous phase. N=400, 400 evaluations per gradient step.
    # Each evaluation is an FFT of size 800. Very fast.
    # 100 iterations * 400 evaluations * 1us/fft = 40ms. Acceptable.
    
    # Actually, let's just return the value and let L-BFGS-B estimate the gradient.
    # It uses a finite difference scheme internally.
    
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Continuous Relaxation
    # Initialize with random real values
    np.random.seed(42)
    
    # Try a few random initializations for the continuous relaxation
    num_continuous_tries = 5
    
    for i in range(num_continuous_tries):
        if time.time() > deadline - 20:
            break
            
        x0 = np.random.randn(N)
        
        # Define objective for scipy
        def obj(x):
            R = _autocorr_fft(x)
            # Minimize sum of squares of sidelobes
            return np.sum(R[1:]**2)
        
        # Use L-BFGS-B
        try:
            res = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8})
            x_opt = res.x
        except Exception:
            continue
        
        # Discretize: map to +/- 1
        # Threshold at 0
        h = np.sign(x_opt)
        # Handle zeros
        h[h == 0] = 1
        
        # Ensure we have a mix? Not strictly necessary.
        
        score = _calc_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
        # Also try the negative
        h_neg = -h
        score_neg = _calc_score(h_neg)
        if score_neg < best_score:
            best_score = score_neg
            best_h = h_neg.copy()

    if best_h is None:
        # Fallback to random
        best_h = np.random.choice([-1, 1], size=N)
        best_score = _calc_score(best_h)
        
    # Phase 2: Discrete Refinement (Bit Flipping)
    # Start from the best continuous-derived binary sequence
    h = best_h.copy()
    current_score = best_score
    
    # Local search
    # We will flip bits that reduce the max sidelobe.
    # To speed up, we can compute the change in correlation caused by a flip.
    # Flipping h[j] from +1 to -1 (or vice versa) changes h by -2*h[j] at index j.
    # Let delta = -2 * h[j] * e_j (where e_j is unit vector).
    # New h = h + delta.
    # New autocorrelation R_new = (h+delta)*(h+delta) = h*h + h*delta + delta*h + delta*delta
    # R_new[k] = R[k] + R_h_delta[k] + R_delta_h[k] + R_delta_delta[k]
    # Since h is real and symmetric autocorrelation, R_h_delta[k] = sum h[i] delta[i+k]
    # delta is non-zero only at j. So delta[i+k] is non-zero only if i+k=j => i=j-k.
    # So R_h_delta[k] = h[j-k] * delta[j]
    # Similarly R_delta_h[k] = delta[j-k] * h[j] ? No, convolution definition.
    # (f*g)[n] = sum f[i] g[n-i].
    # R_h_delta[k] = sum_i h[i] delta[i+k].
    # delta[m] is non-zero only at m=j. So i+k=j => i=j-k.
    # R_h_delta[k] = h[j-k] * delta[j].
    # R_delta_h[k] = sum_i delta[i] h[i+k].
    # delta[i] non-zero only at i=j.
    # R_delta_h[k] = delta[j] * h[j+k].
    # R_delta_delta[k] = delta[j-k]*delta[j] ? Only non-zero if j-k=j => k=0.
    # So for k != 0, R_delta_delta[k] = 0.
    # For k=0, R_delta_delta[0] = delta[j]^2 = 4.
    
    # So for k != 0:
    # R_new[k] = R[k] + delta[j] * (h[j-k] + h[j+k])
    # Note: indices must be valid. If j-k < 0 or >= N, term is 0.
    
    # We want to minimize max |R_new[k]| for k != 0.
    # This is still hard to update exactly for the max.
    # But we can just recompute the score if we suspect improvement, or use the update to estimate.
    # Given N=400, recomputing FFT is ~10us. 
    # 400 flips * 1000 iterations * 10us = 4ms. Very fast.
    # So we can just recompute _calc_score for each flip candidate.
    
    # However, _calc_score computes max. 
    # Let's do a simple hill climbing.
    
    iterations = 0
    stuck_count = 0
    max_stuck = 20
    
    while time.time() < deadline - 5:
        improved = False
        # Shuffle indices to avoid bias
        indices = np.random.permutation(N)
        
        for j in indices:
            if time.time() > deadline - 5:
                break
            
            old_val = h[j]
            h[j] = -h[j]
            
            new_score = _calc_score(h)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                # Keep the flip
            else:
                # Revert
                h[j] = old_val
        
        if improved:
            stuck_count = 0
            if current_score < best_score:
                best_score = current_score
                best_h = h.copy()
        else:
            stuck_count += 1
            if stuck_count > max_stuck:
                # Perturb
                num_perturb = 5
                for _ in range(num_perturb):
                    j = np.random.randint(N)
                    h[j] = -h[j]
                current_score = _calc_score(h)
                stuck_count = 0
                # Continue climbing from perturbed state
        
    return best_h

# Run a quick self-test to ensure it doesn't crash immediately
if __name__ == "__main__":
    h = construct()
    print(f"Score: {_calc_score(h)}")
