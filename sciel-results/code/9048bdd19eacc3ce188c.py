import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_corr(h):
        """Compute circular autocorrelation using FFT."""
        fft_h = np.fft.fft(h, n=2 * N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return corr

    def compute_overlaps(h):
        """Compute the vector of overlaps O_k = sum h_i (1 - h_{i+k}) * (1/N) is not needed if we just compare relative values, 
        but the score is the raw value. 
        The integral is 1, so dx = 1/N.
        Overlap_k = (1/N) * sum h_i (1 - h_{i+k}) = (1/N) * (N/2 - corr_k) = 0.5 - corr_k/N.
        We want to minimize max_k Overlap_k.
        """
        corr = compute_corr(h)
        # Overlap_k = 0.5 - corr_k / N
        # To avoid division by N in the gradient if we scale, let's just minimize max(0.5 - corr_k/N).
        # Or simply minimize max( (N/2 - corr_k) / N ).
        # Let's work with S_k = N/2 - corr_k. Minimizing max S_k / N is same as minimizing max S_k.
        return N / 2.0 - corr

    def project_to_polytope(h, S):
        """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 } using binary search."""
        if S < 0 or S > N:
            return np.clip(h, 0, 1)
            
        lo = np.min(h) - 1.0
        hi = np.max(h) + 1.0
        
        for _ in range(60):
            mid = (lo + hi) / 2.0
            val = np.sum(np.clip(h - mid, 0.0, 1.0))
            if val > S:
                lo = mid
            else:
                hi = mid
                
        lambda_star = (lo + hi) / 2.0
        return np.clip(h - lambda_star, 0.0, 1.0)

    def smooth_max(overlaps, alpha):
        """Soft max of overlaps."""
        max_o = np.max(overlaps)
        # Avoid overflow
        exps = np.exp(alpha * (overlaps - max_o))
        sum_exps = np.sum(exps)
        return max_o + (1.0 / alpha) * np.log(sum_exps)

    def objective_and_grad(params, alpha):
        """
        params: [h_0, ..., h_{N-1}, t]
        Objective: t + penalty(t < smooth_max(overlaps))
        We use a penalty method to handle the inequality constraint t >= smooth_max(overlaps)
        inside an unconstrained minimizer, or we can use SLSQP with constraints.
        SLSQP is better for constraints.
        """
        h = params[:N]
        t = params[N]
        
        # Ensure h is feasible? No, SLSQP handles bounds. 
        # But sum constraint is tricky for SLSQP if we just use bounds.
        # We will use SLSQP with:
        # 1. Bounds [0, 1] for h
        # 2. Equality constraint sum(h) = N/2
        # 3. Inequality constraint t - smooth_max(overlaps) >= 0
        
        overlaps = compute_overlaps(h)
        smax = smooth_max(overlaps, alpha)
        
        # Objective is just t
        val = t
        
        # Gradient
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        
        # Gradient of smooth_max w.r.t h
        # d(smax)/d(overlap_k) = exp(alpha(overlap_k - max_o)) / sum(exp(...))
        exps = np.exp(alpha * (overlaps - np.max(overlaps)))
        weights = exps / np.sum(exps)
        
        # d(overlap_k)/d(h_i)
        # overlap_k = N/2 - corr_k
        # corr_k = sum h_j h_{j+k}
        # d(corr_k)/d(h_i) = h_{i+k} + h_{i-k} (circular)
        # d(overlap_k)/d(h_i) = -(h_{i+k} + h_{i-k})
        
        # So d(smax)/d(h_i) = sum_k weights[k] * (-(h_{i+k} + h_{i-k}))
        # This can be computed via convolution/FFT for efficiency if needed, but N=100 is small.
        # Loop is O(N^2) = 10,000 ops. Very fast.
        
        idx = np.arange(N)
        # We need sum_k weights[k] * (h_{(i+k)%N} + h_{(i-k)%N})
        # Let W be weights.
        # term1_i = sum_k W_k h_{(i+k)%N} -> Circular Convolution of W and h
        # term2_i = sum_k W_k h_{(i-k)%N} -> Circular Convolution of W and h (reversed)
        
        # Using FFT for O(N log N)
        fft_w = np.fft.fft(weights)
        fft_h = np.fft.fft(h)
        conv1 = np.fft.ifft(fft_w * fft_h).real
        # h_{i-k} is correlation with W.
        # np.correlate or just fft of reversed W?
        # Actually, sum_k W_k h_{i-k} is the same as circular correlation.
        # In FFT domain, correlation of h and W is conj(fft_h) * fft_W? 
        # Let's just use the loop for N=100 to be safe and simple, or use np.roll.
        # Loop:
        # grad_h = - (conv(weights, h) + conv(weights, h[::-1])?) 
        # Let's stick to the explicit loop for clarity and correctness verification, it's fast enough.
        
        grad_h = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-15:
                # d(overlap_k)/d(h_i) = -(h_{i+k} + h_{i-k})
                # This contributes to grad_h[i]
                # grad_h[i] += weights[k] * (-(h[(i+k)%N] + h[(i-k)%N]))
                # To avoid N^2 python loops, use vectorized numpy operations for each k?
                # Or just use the FFT convolution derived above.
                pass
        
        # Let's use FFT convolution properly.
        # A_i = sum_k W_k h_{i+k}
        # B_i = sum_k W_k h_{i-k}
        # Note B_i = sum_k W_k h_{(i-k)%N}. Let j = -k. sum_j W_{-j} h_{i+j}.
        # So B is convolution of h and W_rev.
        
        fft_w = np.fft.fft(weights)
        fft_w_rev = np.fft.fft(weights[::-1])
        fft_h = np.fft.fft(h)
        
        A = np.fft.ifft(fft_w * fft_h).real
        B = np.fft.ifft(fft_w_rev * fft_h).real
        
        grad_h = -(A + B)
        
        grad[:N] = grad_h
        
        return val, grad

    def constraint_sum(params):
        return np.sum(params[:N]) - N / 2.0

    def constraint_epigraph(params, alpha):
        h = params[:N]
        t = params[N]
        overlaps = compute_overlaps(h)
        smax = smooth_max(overlaps, alpha)
        return t - smax

    def constraint_epigraph_grad(params, alpha):
        h = params[:N]
        t = params[N]
        overlaps = compute_overlaps(h)
        max_o = np.max(overlaps)
        exps = np.exp(alpha * (overlaps - max_o))
        weights = exps / np.sum(exps)
        
        # Gradient of (t - smax)
        # d/dt = 1
        # d/dh_i = - d(smax)/d(h_i)
        
        fft_w = np.fft.fft(weights)
        fft_w_rev = np.fft.fft(weights[::-1])
        fft_h = np.fft.fft(h)
        A = np.fft.ifft(fft_w * fft_h).real
        B = np.fft.ifft(fft_w_rev * fft_h).real
        grad_smax_h = -(A + B)
        
        grad = np.zeros(N + 1)
        grad[:N] = -grad_smax_h
        grad[N] = 1.0
        return grad

    # Initial guesses
    def get_initial_guesses():
        guesses = []
        # 1. 2-block
        # Minimize max overlap for 2 blocks.
        # Let block 1 height a, length k. Block 2 height b, length N-k.
        # a*k + b*(N-k) = N/2
        # Overlap is maximized at shift k or N-k?
        # For 2 blocks, the optimal is often related to golden ratio.
        # Try a few k values.
        for k in range(1, N):
            a_min = max(0.0, (N / 2.0 - N + k) / k)
            a_max = min(1.0, N / (2.0 * k))
            if a_min >= a_max:
                continue
            # Optimal a for fixed k?
            # Just pick a few candidates
            a_vals = np.linspace(a_min + 1e-6, a_max - 1e-6, 10)
            for a_val in a_vals:
                b_val = (N / 2.0 - k * a_val) / (N - k)
                if b_val < -1e-9 or b_val > 1.0 + 1e-9:
                    continue
                h = np.zeros(N)
                h[:k] = a_val
                h[k:] = b_val
                guesses.append(h.copy())
        
        # 2. Uniform
        guesses.append(np.full(N, 0.5))
        
        # 3. Random
        for _ in range(5):
            h = np.random.rand(N)
            h = project_to_polytope(h, N / 2.0)
            guesses.append(h)
            
        return guesses

    best_h = None
    best_score = float('inf')
    alpha = 200.0 # Smoothness parameter

    for guess in get_initial_guesses():
        if time.time() > deadline - 10.0:
            break
            
        x0 = np.concatenate([guess, [best_score if best_score < float('inf') else 0.5]])
        
        bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]
        
        constraints = [
            {'type': 'eq', 'fun': constraint_sum},
            {'type': 'ineq', 'fun': lambda p: constraint_epigraph(p, alpha), 
             'jac': lambda p: constraint_epigraph_grad(p, alpha)}
        ]
        
        try:
            res = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success or res.fun < best_score:
                h_opt = res.x[:N]
                # Project to ensure exact feasibility
                h_opt = project_to_polytope(h_opt, N / 2.0)
                score = np.max(compute_overlaps(h_opt)) / N # Raw value
                # Wait, the score is "upper bound on C5".
                # The task says "score is the upper bound on C5".
                # C5 is the constant in the inequality.
                # Usually C5 = max overlap / (integral * integral)?
                # Integral is 1. So C5 = max overlap.
                # My compute_overlaps returns N/2 - corr.
                # Overlap integral = (1/N) * (N/2 - corr).
                # So score = max_k (0.5 - corr_k/N).
                
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
                    
        except Exception as e:
            pass

    if best_h is None:
        return np.full(N, 0.5)

    # Final safety check
    if best_h.min() < -1e-9 or best_h.max() > 1 + 1e-9:
        return np.full(N, 0.5)
    if abs(best_h.sum() - N / 2.0) > 1e-6:
        return np.full(N, 0.5)

    return best_h
