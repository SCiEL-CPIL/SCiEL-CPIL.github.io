import numpy as np
import scipy.optimize
import time

N = 600
DX = 1.0 / 1200.0  # Width of each interval

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # The task asks for a nonnegative step function on [-1/4, 1/4] with 600 intervals.
    # The "normalized maximum of its autoconvolution" is the quantity to minimize.
    # Let h be the vector of step heights.
    # The integral is I = sum(h) * DX.
    # The autoconvolution at lag k is approx (1/DX) * sum_{j} h_j h_{j+k}.
    # Actually, the convolution of two step functions (piecewise constant) is piecewise linear? No.
    # If f(x) is a step function, f*f is the convolution.
    # (f*f)(t) = \int f(x) f(t-x) dx.
    # For step functions on a grid, this is effectively the discrete convolution scaled by DX.
    # Let's look at the seed. Seed: h = [1.0]*600.
    # sum(h) = 600. I = 600 * (1/1200) = 0.5.
    # Discrete conv c_k = sum h_j h_{j+k}.
    # For uniform h=1, c_0 = 600. c_1 = 599... c_k = 600-k.
    # The continuous convolution peak would be at t=0.
    # (f*f)(0) = \int f(x) f(-x) dx. Since f is even? No, f is on [-1/4, 1/4].
    # If f(x) = 1 on [-1/4, 1/4], then f*f is a triangle function.
    # Peak at 0: \int_{-1/4}^{1/4} 1 * 1 dx = 1/2.
    # Wait, the support of f*f is [-1/2, 1/2].
    # At t=0, (f*f)(0) = \int_{-1/4}^{1/4} f(x) f(-x) dx.
    # If f(x)=1, then f(-x)=1. Integral is 0.5.
    # Normalized max = max(f*f) / I^2?
    # If the score is an "upper bound on C1", and C1 is from ||f*f||_inf <= C1 ||f||_1^2.
    # Then C1_candidate = max(f*f) / (sum(h)*DX)^2.
    # For uniform: max(f*f) = 0.5. I = 0.5. I^2 = 0.25.
    # C1 = 0.5 / 0.25 = 2.0.
    # The seed score is 1.0. This implies the evaluator might be returning 1.0 for the seed?
    # "The search score is seed/value... 1.0 equals the seed program".
    # If the seed returns 1.0, and the raw value is 2.0, then maybe the evaluator computes `seed_value / value`?
    # Or maybe the evaluator just returns the raw value, and the "seed program" in the evaluator's context returns 1.0?
    # No, the prompt says: "score is the upper bound on C1... The evaluator measures the raw value (lower is better)."
    # "The search score is seed/value... 1.0 equals the seed program".
    # This implies: SearchScore = SeedScore / CurrentScore.
    # If SeedScore (the C1 value for the seed) is 2.0, and I find a C1 of 1.0, the search score is 2.0/1.0 = 2.0.
    # So I need to minimize the raw C1 value.
    
    # Let's verify the C1 calculation for the uniform case.
    # f(x) = 1 on [-1/4, 1/4].
    # ||f||_1 = 1/2.
    # (f*f)(t) = (1/2 - |t|) for |t| <= 1/2.
    # max(f*f) = 1/2.
    # C1 = (1/2) / (1/4) = 2.
    
    # So the goal is to minimize max(f*f) / (||f||_1^2).
    # Since the objective is scale-invariant, we can fix ||f||_1 = 1 (i.e., sum(h)*DX = 1).
    # Then we just need to minimize max(f*f).
    # max(f*f) is the peak of the autoconvolution.
    # The autoconvolution of a non-negative function is maximized when the mass is concentrated.
    # A delta function has infinite peak.
    # A step function with 600 intervals.
    # If we concentrate all mass in 1 interval:
    # h_i = M for one i, 0 elsewhere.
    # sum(h) = M. I = M*DX = 1 => M = 1/DX = 1200.
    # Autoconvolution: The peak will be at lag 0.
    # The discrete autoconvolution c_0 = sum h_j^2 = M^2.
    # But this is the discrete sum. How does it map to continuous?
    # The continuous conv is an approximation.
    # Let's stick to the discrete formulation which is likely what the evaluator uses, or a consistent continuous one.
    # If the evaluator uses:
    # 1. Construct continuous f from h.
    # 2. Compute f*f.
    # 3. Find max.
    # 4. Compute ||f||_1.
    # 5. Return max / ||f||_1^2.
    
    # If I use 1 interval:
    # f is 1200 on an interval of width 1/1200.
    # f*f will be a triangle of height 1200*1200*(1/1200) = 1200?
    # Let's check.
    # f(x) = 1200 on [0, 1/1200].
    # (f*f)(t) = \int 1200 * 1200 * (length of overlap) dx.
    # Overlap length for t in [0, 1/1200] is (1/1200 - t).
    # Max at t=0: 1440000 * (1/1200) = 1200.
    # ||f||_1 = 1.
    # C1 = 1200 / 1 = 1200.
    # This is WORSE than uniform (2.0).
    
    # Why? Because the uniform distribution spreads the mass out, reducing the peak of the convolution.
    # The convolution peak is related to the "energy" of the function.
    # We want to MINIMIZE the peak.
    # So we want to SPREAD the mass as much as possible?
    # Uniform is already quite spread.
    # Can we spread it more?
    # The support is fixed to [-1/4, 1/4].
    # Uniform uses the entire support.
    # Is there a shape that has a lower peak convolution than uniform?
    
    # Consider a "U-shape" (bimodal at edges).
    # h = [1, 0, 0, ..., 0, 1] (scaled).
    # Two deltas at the edges.
    # f(x) is two spikes at -1/4 and 1/4.
    # The self-convolution will have peaks at 0 (spike-spike), -1/2 (left-left), 1/2 (right-right).
    # Wait, the support of f*f is [-1/2, 1/2].
    # If f has support [-1/4, 1/4], then f*f has support [-1/2, 1/2].
    # The peaks at -1/2 and 1/2 come from the correlation of the edge with itself?
    # No, (f*f)(t) = \int f(x) f(t-x) dx.
    # If f is non-zero only at -1/4 and 1/4 (approx).
    # Then f(t-x) is non-zero if t-x is -1/4 or 1/4.
    # So x = t + 1/4 or x = t - 1/4.
    # If t=0: x=1/4 or x=-1/4. Both are in support.
    # So (f*f)(0) gets contributions from both edges.
    # If f is uniform, (f*f)(0) = 0.5.
    # If f is two deltas of height H at edges.
    # I = 1. Each delta has area 0.5.
    # f(x) = 0.5 / (w/2) on [1/4-w/2, 1/4] and [-1/4, -1/4+w/2]. (Approx)
    # Actually, let's just use the discrete optimization to find the shape that minimizes the discrete max autoconvolution, assuming the evaluator's metric correlates with this.
    
    # Objective: Minimize max( autoconv(h) ) subject to sum(h) = 1.
    # Or equivalently, minimize max( autoconv(h) / sum(h)^2 ).
    
    def compute_cost_and_grad(x):
        """
        x: log-heights. h = softmax(x).
        Returns cost and gradient w.r.t x.
        """
        # Softmax
        x_shifted = x - np.max(x)
        u = np.exp(x_shifted)
        h = u / np.sum(u)
        
        # Autoconvolution via FFT
        # h is length N. We need circular or linear?
        # Linear autoconvolution of length 2N-1.
        # We only care about the max.
        # Using FFT size 2N is safe for linear conv (no wrap-around for the first N lags, and the rest is symmetric).
        # Actually, autoconv of real signal is symmetric.
        # c_k = sum_{j} h_j h_{j+k}.
        # This is circular if we use FFT without zero padding.
        # We MUST use zero padding to get linear convolution.
        size = 2 * N
        H_fft = np.fft.rfft(h, n=size)
        C = np.fft.irfft(H_fft * H_fft, n=size)
        # C[0] is the sum of squares. C[1] is sum h_j h_{j+1}, etc.
        # The max of the autoconvolution is max(C).
        # Note: C is symmetric, C[k] = C[-k].
        # We just need the max of the first N (or all, it's the same).
        
        # Cost is max(C) / (sum(h))^2. Since sum(h)=1, cost is max(C).
        # But we want to be smooth. Use LogSumExp.
        beta = 50.0
        max_C = np.max(C)
        # If max_C is large, exp(beta*C) overflows.
        # C is roughly 1/N = 1/600 = 0.0016 for uniform.
        # beta * max_C = 50 * 0.0016 = 0.08. exp(0.08) is fine.
        # If we concentrate mass, max_C can be large.
        # If h is delta, max_C ~ 1. beta * 1 = 50. exp(50) ~ 5e21. Safe.
        
        exp_C = np.exp(beta * (C - max_C))
        sum_exp = np.sum(exp_C)
        cost = max_C + np.log(sum_exp) / beta
        
        # Gradient of cost w.r.t h
        # P_k = exp_C_k / sum_exp. This is a probability distribution over lags.
        P = exp_C / sum_exp
        # G_h = 2 * P * h (convolved? No.)
        # Let's derive d/max_C / dh.
        # If cost = max(C), and C_k = sum_j h_j h_{j+k}.
        # dC_k/dh_i = h_{i-k} + h_{i+k} (with wraparound if circular, but we used linear so boundaries are 0).
        # Wait, the FFT gave us linear convolution.
        # The gradient of the max is tricky.
        # Approximation: The gradient of LogSumExp is the expectation of the "one-hot" vector of the max, weighted by softmax.
        # So the effective gradient of the cost w.r.t C is P.
        # G_h = sum_k P_k * (dC_k/dh).
        # dC_k/dh = h_{-k} * 1? No.
        # C_k = sum_j h_j h_{j+k}.
        # This is the circular cross-correlation if we used circular FFT.
        # We used linear FFT (zero padded).
        # In linear convolution, C_k = sum_{j: j, j+k in [0, N-1]} h_j h_{j+k}.
        # The gradient dC_k/dh_i is h_{i-k} (if i-k valid) + h_{i+k} (if i+k valid).
        # This is equivalent to the circular convolution of P with h, IF we treat the arrays as circular of size 2N?
        # No, the zero-padding breaks circular symmetry for the gradient in the simple FFT sense unless we are careful.
        
        # Alternative: Compute gradient numerically? Too slow for 600 dims.
        # Let's use the property that for circular convolution, grad is 2 * P * h (circular).
        # Since we zero-padded to 2N, the "circular" convolution of P (length N, padded to 2N) and h (length N, padded to 2N) will give the correct gradient for the linear convolution?
        # Let's verify.
        # C_k = sum_{j} h_j h_{j+k} (linear).
        # This is the same as circular conv if we pad with 0s and the support doesn't wrap.
        # The gradient dC_k/dh_i = h_{i-k} + h_{i+k}.
        # Let's compute G = 2 * (P * h) where * is circular conv of size 2N.
        # (P * h)_i = sum_k P_k h_{i-k (mod 2N)}.
        # If we pad P and h to 2N, this should work.
        
        P_padded = np.zeros(size)
        P_padded[:N] = P
        # P is symmetric? C is symmetric, so P is symmetric.
        # Actually, C_k = C_{-k}. So P_k = P_{-k}.
        # In the FFT domain, C is real.
        
        H_fft_full = np.fft.fft(h, n=size)
        P_fft = np.fft.fft(P_padded, n=size)
        G_h = 2.0 * np.fft.ifft(P_fft * H_fft_full).real[:N]
        
        # Gradient w.r.t x
        # h_i = exp(x_i) / Z.
        # G_x = h * (G_h - dot(G_h, h))
        dot_Gh_h = np.dot(G_h, h)
        G_x = h * (G_h - dot_Gh_h)
        
        return cost, G_x

    # Initialization strategies
    idx = np.arange(N)
    center = (N - 1) / 2.0
    
    # 1. Uniform (x = 0)
    x0_uniform = np.zeros(N)
    
    # 2. U-shape (bimodal)
    # We want high at ends.
    # h_i ~ exp(c * (i - center)^2).
    # This is a U-shape.
    x0_u = 1.0 * (idx - center)**2 / N
    
    # 3. Cosine (1 + c cos)
    # h_i = 1 + 0.9 cos(2 pi k i / N).
    # Log space: log(1 + 0.9 cos(...))
    # Need to ensure positivity. 1 - 0.9 = 0.1 > 0.
    x0_cos1 = np.log(1.0 + 0.9 * np.cos(2 * np.pi * 1 * idx / N))
    
    best_cost = float('inf')
    best_x = x0_uniform.copy()
    
    # Run L-BFGS-B for each initialization
    initializations = [x0_uniform, x0_u, x0_cos1]
    
    # Add a few random initializations
    for _ in range(3):
        # Smooth random
        noise = np.random.normal(0, 1, N)
        # Smooth it with a kernel?
        # Just use it, L-BFGS should smooth it out.
        # Or use a low-pass filter.
        x0_rand = noise / np.std(noise) * 2.0
        initializations.append(x0_rand)
        
    for x0 in initializations:
        if time.time() > deadline - 10:
            break
        try:
            res = scipy.optimize.minimize(
                compute_cost_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 100}
            )
            if res.fun < best_cost:
                best_cost = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Convert best_x to h
    x_shifted = best_x - np.max(best_x)
    u = np.exp(x_shifted)
    h = u / np.sum(u)
    
    # Return list of heights
    return list(h)

if __name__ == "__main__":
    h = construct()
    # Estimate score
    h_arr = np.array(h)
    size = 2 * N
    H_fft = np.fft.rfft(h_arr, n=size)
    C = np.fft.irfft(H_fft * H_fft, n=size)
    max_C = np.max(C)
    sum_h = np.sum(h_arr)
    C1 = max_C / (sum_h * sum_h)
    print("Estimated C1:", C1)
    print("Sample heights:", h[:5])
