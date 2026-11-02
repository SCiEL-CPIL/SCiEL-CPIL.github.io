"""Program using spectral amplitude shaping to minimize autoconvolution peak."""
import time
import numpy as np
from scipy import optimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    if len(h) == 0:
        return 0.0
    s = np.sum(h)
    if s == 0:
        return 0.0
    # Autoconvolution (linear)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Spectral parameterization:
    # We represent the step function h as the inverse DFT of a vector Z.
    # To ensure h is non-negative and real, we constrain the structure of Z.
    # Let Z[k] = A[k] * exp(i * phi[k]).
    # For h to be real, Z must have Hermitian symmetry: Z[N-k] = conj(Z[k]).
    # We can parameterize A and phi for k=0..N/2.
    # To ensure non-negativity, we can optimize the amplitudes A[k] directly.
    # However, arbitrary A[k] do not guarantee h >= 0.
    # A robust way to ensure non-negativity is to let h = |IDFT(Z)|^2? No, that's positive semidefinite but might not be flexible enough.
    # Alternatively, let h be the IDFT of a vector with non-negative real part? No.
    
    # Let's try a different spectral approach:
    # Minimize the max of h*h.
    # In frequency domain, h*h corresponds to |H|^2 (circular conv) or similar.
    # Actually, linear conv h*h in time corresponds to H(f)^2 in freq (with scaling).
    # The peak of h*h in time is related to the L1 norm of H(f)^2? No.
    # Parseval: sum |h*h|^2 = sum |H|^4.
    # We want to minimize max(h*h).
    
    # Let's stick to the time-domain but use a better initialization and optimizer.
    # The previous attempt used L-BFGS-B on a smooth approximation of max.
    # Let's try a "flatness" approach in the frequency domain.
    # If H(f) is constant, h is a delta (in continuous) or wide (in discrete).
    # If h is uniform, H is a sinc.
    
    # Let's try optimizing the spectral amplitudes directly with a penalty for non-negativity.
    # Variables: real and imaginary parts of Z for k=0..N/2.
    # h = IDFT(Z).
    # Penalty: sum(max(0, -h))^2.
    # Objective: max(conv(h,h)) + lambda * penalty.
    
    # This is a non-smooth objective due to max.
    # We can use a soft-max for the conv peak and a smooth penalty for non-negativity.
    
    # However, 600 variables is small.
    # Let's try a simpler structural change: 
    # The "Two-Block" inspiration suggests that sparse or blocky structures are good.
    # What if we optimize a "comb" or "periodic" structure?
    
    # Let's try a grid search over a few specific parametric families that are known to be good for correlation problems,
    # and then refine with a local search.
    
    # Family 1: Uniform (score ~ 1/N)
    # Family 2: Cosine bell (score ~ ?)
    # Family 3: Two blocks (inspiration)
    # Family 4: "Tent" function (triangular)
    # Family 5: Gaussian
    
    # Let's implement a fast local search using coordinate descent or a simple gradient step
    # on the raw max function, but with random restarts from diverse parametric seeds.
    
    # Actually, the spectral idea in the prompt is: "IDFT of complex vector with fixed unit-modulus phases and optimized amplitudes".
    # If phases are fixed (e.g., 0), then h = IDFT(A) where A is real and symmetric?
    # If A is real and symmetric, h is real and even.
    # If A >= 0, h is not necessarily >= 0.
    # But if we optimize A to make h flat, it might work.
    
    # Let's try: h = IDFT(A) where A is a real, symmetric vector.
    # We optimize A to minimize max(h*h) subject to h >= 0.
    # We can enforce h >= 0 by projecting h to non-negative after each IDFT?
    # If we project h, the spectral representation is lost.
    
    # Let's try a different angle: 
    # The problem is equivalent to finding a function h with L1 norm 1 (normalized) that minimizes ||h*h||_inf.
    # This is related to the "flatness" of the autocorrelation.
    # Random phases in frequency domain give a "white noise" like autocorrelation (low peaks).
    # So, let's generate h by taking the IDFT of a vector with random phases and optimized amplitudes.
    # To ensure h >= 0, we can take h = |IDFT(Z)|. This h is non-negative.
    # Then we optimize the amplitudes of Z to minimize max(h*h).
    # Note: h = |IDFT(Z)| is not linear in Z, so gradients are tricky.
    # But we can use gradient-free methods or subgradients.
    
    # Let's try a simple random search with hill climbing on the amplitudes.
    # Or better: use scipy.optimize.differential_evolution on the amplitudes?
    # 600 dimensions is too high for DE.
    
    # Let's go back to the L-BFGS-B on time-domain h, but with a better smooth objective.
    # The previous program did this. Why did it stagnate? Maybe it got stuck in local minima.
    # Let's try more diverse initializations and a multi-start strategy.
    
    # New Strategy: Multi-start L-BFGS-B with diverse parametric seeds.
    
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Cosine window
    x = np.linspace(-0.25, 0.25, N)
    candidates.append(0.5 * (1 + np.cos(4 * np.pi * x)))
    
    # 3. Gaussian
    for sigma in [0.05, 0.1, 0.15, 0.2]:
        candidates.append(np.exp(-x**2 / (2*sigma**2)))
        
    # 4. Two-humped (mass at edges)
    for center in [0.1, 0.15, 0.2]:
        for w in [0.01, 0.02, 0.05]:
            candidates.append(np.exp(-(x-center)**2/w) + np.exp(-(x+center)**2/w))
            
    # 5. Polynomial bumps
    for p in [0.5, 1, 2, 3, 4, 6, 8]:
        t = np.abs(x) / 0.25
        h_p = np.maximum(0, 1 - t**2) ** p
        candidates.append(h_p)
        
    # 6. Random phases (white noise in freq)
    # h = |IDFT(Z)| where Z has random phases and flat amplitude
    for _ in range(5):
        Z = np.ones(N, dtype=complex)
        Z *= np.exp(1j * np.random.uniform(0, 2*np.pi, N))
        # Ensure Hermitian symmetry for real h
        Z[0] = np.abs(Z[0])
        if N % 2 == 0:
            Z[N//2] = np.abs(Z[N//2])
        for k in range(1, N//2):
            Z[N-k] = np.conj(Z[k])
        h_rand = np.abs(np.fft.ifft(Z))
        candidates.append(h_rand)
        
    # 7. Two blocks (analytic structure)
    # h = A1 * rect(x-c1, w1) + A2 * rect(x-c2, w2)
    # We can sample a few configurations
    for c1 in [-0.1, 0, 0.1]:
        for w1 in [0.05, 0.1, 0.2]:
            for c2 in [0.1, 0.2]:
                for w2 in [0.05, 0.1, 0.2]:
                    h_block = np.zeros(N)
                    mask1 = np.abs(x - c1) < w1/2
                    mask2 = np.abs(x - c2) < w2/2
                    h_block[mask1] = 1.0
                    h_block[mask2] = 1.0
                    if np.sum(h_block) > 0:
                        candidates.append(h_block)

    best_h = candidates[0].copy()
    best_score = compute_score(best_h)
    
    for h in candidates[1:]:
        if time.time() > deadline:
            break
        score = compute_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Optimization
    def objective_smooth(h_flat):
        h = np.maximum(0, h_flat)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        conv = np.convolve(h, h, mode='full')
        norm_conv = conv / (s * s)
        # Soft max with temperature T
        T = 0.1
        max_val = np.max(norm_conv)
        shifted = norm_conv - max_val
        log_sum_exp = max_val + np.log(np.sum(np.exp(shifted / T)) / len(norm_conv)) * T
        return log_sum_exp

    h0 = best_h.copy()
    s0 = np.sum(h0)
    if s0 > 0:
        h0 = h0 / s0
        
    bounds = [(0, None)] * N
    
    # Run optimization from best candidate
    try:
        res = optimize.minimize(
            objective_smooth, 
            h0, 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() < deadline:
            h_opt = np.maximum(0, res.x)
            score_opt = compute_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
    except Exception:
        pass

    # Multi-start from random candidates
    if time.time() < deadline - 20:
        for i in range(10):
            if time.time() > deadline - 5:
                break
            # Pick a random candidate or generate a new one
            if i < len(candidates):
                h_rand = candidates[i].copy()
            else:
                h_rand = np.random.uniform(0, 1, N)
            
            s_rand = np.sum(h_rand)
            if s_rand > 0:
                h_rand = h_rand / s_rand
            
            try:
                res = optimize.minimize(
                    objective_smooth, 
                    h_rand, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                h_opt = np.maximum(0, res.x)
                score_opt = compute_score(h_opt)
                if score_opt < best_score:
                    best_score = score_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    return best_h.tolist()
# EVOLVE-BLOCK-END
