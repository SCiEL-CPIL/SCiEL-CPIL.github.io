import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin for 150s limit
    
    # The objective is to minimize the maximum absolute value of the autocorrelation sidelobes.
    # Let x be a vector of length N with entries in {-1, 1}.
    # Autocorrelation at lag k is R(k) = sum_{i=0}^{N-k-1} x[i] x[i+k] for k > 0.
    # We want to minimize max_{k=1..N-1} |R(k)|.
    
    # A continuous relaxation approach:
    # Let y be a real vector. We want to minimize f(y) = max_{k=1..N-1} |R_y(k)|^2
    # where R_y(k) is computed with y. Then project y to the nearest sign vector.
    # However, the objective is non-convex and non-smooth (due to max and abs).
    
    # Alternative: Use a differentiable proxy.
    # Minimize the sum of squares of the sidelobes: g(y) = sum_{k=1}^{N-1} R_y(k)^2.
    # This is equivalent to minimizing the total energy in the sidelobes.
    # Note that sum_{k=0}^{N-1} R_y(k)^2 = sum_{i=0}^{N-1} y[i]^4 = N if y[i] in {-1,1}.
    # So minimizing the sum of squares of sidelobes is equivalent to maximizing R_y(0)^2 - sum_{k=1}^{N-1} R_y(k)^2? No.
    # Actually, for a sign sequence, the total energy is fixed. Minimizing sidelobe energy is a good proxy for reducing the peak sidelobe.
    
    # Let's use the sum of squares of the sidelobes as the objective for a continuous optimization.
    # Let z be a vector of real numbers. We define the objective function h(z) = sum_{k=1}^{N-1} R_z(k)^2.
    # We can compute R_z(k) efficiently using FFT.
    # R_z(k) = sum_{i=0}^{N-k-1} z[i] z[i+k].
    # This is the aperiodic autocorrelation.
    
    # A better proxy: Minimize the L-infinity norm of the sidelobes using a smooth approximation like the log-sum-exp function.
    # h(z) = log(sum_{k=1}^{N-1} exp(beta * R_z(k)^2)) / beta
    # As beta -> infinity, this approximates max(R_z(k)^2).
    
    # Let's try to minimize the sum of squares first, as it is smoother and easier to optimize.
    # Then we can refine with local search on the sign sequence.
    
    def compute_sidelobe_energy(z):
        # Compute aperiodic autocorrelation
        # Use FFT for efficiency
        n_fft = 1 << (2 * N - 1).bit_length()
        Z = np.fft.rfft(z, n=n_fft)
        acf = np.fft.irfft(Z * np.conj(Z), n=n_fft)[:N]
        # acf[0] is the main lobe, acf[1:] are the sidelobes
        # Note: irfft gives the first N points of the linear autocorrelation
        # But we need to be careful about the definition.
        # The standard autocorrelation via FFT gives circular autocorrelation if not padded.
        # With padding to 2N-1, we get the linear (aperiodic) autocorrelation.
        # The values are acf[k] = sum_{i=0}^{N-k-1} z[i] z[i+k] for k=0..N-1.
        # So acf[1:] are the sidelobes we care about.
        return np.sum(np.abs(acf[1:]) ** 2)
    
    def compute_sidelobe_energy_grad(z):
        # Gradient of sum_{k=1}^{N-1} R_z(k)^2 with respect to z.
        # This is complex to derive analytically. Let's use numerical gradient or a simpler approach.
        # Actually, we can use the fact that the objective is a quadratic form in z.
        # R_z(k) = z^T A_k z, where A_k is a matrix with 1s on the k-th diagonal.
        # This is getting complicated. Let's use a different approach.
        
        # Alternative: Use the fact that for large N, the aperiodic autocorrelation is close to the periodic one.
        # Or, we can just use a simple gradient descent with numerical gradients if needed.
        
        # Let's try a different proxy: Minimize the variance of the sidelobes.
        # Or, let's use the method of alternating projections.
        
        # Actually, a simpler and effective approach is to use the "Chirp Z-transform" or similar, but let's stick to FFT.
        
        # Let's use a gradient-based method with a smooth objective.
        # We'll use the sum of squares of the sidelobes.
        # We can compute the gradient using the chain rule and FFT.
        
        # Let's define the objective as f(z) = ||P z||^2, where P extracts the sidelobes from the autocorrelation.
        # This is still complex. Let's use a simple heuristic:
        # 1. Initialize z with random values in [-1, 1].
        # 2. Use scipy.optimize.minimize with L-BFGS-B to minimize the sum of squares of the sidelobes.
        # 3. Project to the nearest sign sequence.
        # 4. Refine with local search.
        
        # For the gradient, we can use the fact that the autocorrelation is a convolution.
        # Let's implement the gradient numerically if it's too complex.
        
        # Actually, let's use a different strategy: use the "Goertzel" algorithm or similar to compute the objective and gradient efficiently.
        # But for N=400, even a naive O(N^2) computation is fast enough for a few iterations.
        
        # Let's implement the objective and gradient naively for clarity and correctness.
        pass
    
    # Since deriving the gradient is complex, let's use a different approach:
    # Use a continuous relaxation with a differentiable objective that is easier to handle.
    
    # Objective: Minimize the maximum sidelobe using a log-sum-exp approximation.
    def objective(z):
        # Compute aperiodic autocorrelation
        n_fft = 1 << (2 * N - 1).bit_length()
        Z = np.fft.rfft(z, n=n_fft)
        acf = np.fft.irfft(Z * np.conj(Z), n=n_fft)[:N]
        sidelobes = acf[1:]
        # Use log-sum-exp for smooth max approximation
        beta = 10.0
        # To avoid overflow, subtract the max
        max_val = np.max(np.abs(sidelobes))
        if max_val < 1e-10:
            return 0.0
        # We want to minimize max |sidelobes|, so we use exp(beta * |sidelobes|)
        # But |x| is not differentiable at 0. Let's use x^2 instead.
        # Minimize log(sum exp(beta * sidelobes^2))
        vals = beta * sidelobes ** 2
        max_vals = np.max(vals)
        return (np.log(np.sum(np.exp(vals - max_vals))) + max_vals) / beta
    
    # Initialize z with random values
    rng = np.random.default_rng(42)
    z = rng.uniform(-1, 1, size=N)
    
    # Use L-BFGS-B to minimize the objective
    # Note: This might be slow. Let's use a limited number of iterations.
    
    # Actually, let's use a simpler approach: use the fact that the optimal solution is a sign sequence.
    # We can use the "sign flip" local search but with a better initial guess.
    
    # Let's combine:
    # 1. Use a spectral initialization: z = IFFT(1) = delta, but that's not good.
    # 2. Use z = random signs.
    # 3. Use a "soft" local search: instead of flipping signs, move z in the direction of the negative gradient of the sum of squares of the sidelobes.
    
    # Let's implement the gradient of the sum of squares of the sidelobes.
    # f(z) = sum_{k=1}^{N-1} (sum_{i=0}^{N-k-1} z[i] z[i+k])^2
    # df/dz[j] = sum_{k=1}^{N-1} 2 * R_z(k) * dR_z(k)/dz[j]
    # R_z(k) = sum_{i=0}^{N-k-1} z[i] z[i+k]
    # dR_z(k)/dz[j] = z[j+k] if 0 <= j < N-k, else 0 + z[j-k] if k <= j < N, else 0
    # So df/dz[j] = 2 * sum_{k=1}^{N-1} R_z(k) * (z[j+k] * I(0<=j<N-k) + z[j-k] * I(k<=j<N))
    
    def compute_grad(z):
        # Compute autocorrelation
        n_fft = 1 << (2 * N - 1).bit_length()
        Z = np.fft.rfft(z, n=n_fft)
        acf = np.fft.irfft(Z * np.conj(Z), n=n_fft)[:N]
        R = acf[1:]  # Sidelobes
        
        grad = np.zeros(N)
        # For each j, compute the gradient
        # This is O(N^2), which is acceptable for N=400.
        for j in range(N):
            grad[j] = 0.0
            for k in range(1, N):
                if j < N - k:
                    grad[j] += R[k-1] * z[j+k]
                if j >= k:
                    grad[j] += R[k-1] * z[j-k]
            grad[j] *= 2.0
        return grad
    
    # Use gradient descent
    z = rng.uniform(-1, 1, size=N)
    lr = 0.01
    for i in range(100):
        if time.time() > deadline:
            break
        grad = compute_grad(z)
        z -= lr * grad
        # Project to the box [-1, 1]
        z = np.clip(z, -1, 1)
    
    # Project to the nearest sign sequence
    x = np.sign(z)
    x[x == 0] = 1  # Handle zeros
    
    # Refine with local search
    def max_sidelobe(x):
        n_fft = 1 << (2 * N - 1).bit_length()
        X = np.fft.rfft(x, n=n_fft)
        acf = np.fft.irfft(X * np.conj(X), n=n_fft)[:N]
        return np.max(np.abs(acf[1:]))
    
    best_x = x.copy()
    best_val = max_sidelobe(best_x)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            old_val = best_x[i]
            best_x[i] = -old_val
            new_val = max_sidelobe(best_x)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                best_x[i] = old_val
    
    return best_x
