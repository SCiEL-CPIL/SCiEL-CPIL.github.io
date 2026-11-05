import numpy as np
import time

N = 400

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin before 150s kill

    # Precompute FFT lengths
    n_fft = 2 * N - 1
    freqs = np.fft.fftfreq(N)
    
    # We want to minimize || conv(h,h) - C * 1 ||_1 subject to sum(h) = 1
    # This is equivalent to minimizing || H^2 - C * W ||_1 in frequency domain?
    # No, L1 norm in time domain is hard.
    # However, we can use a trick: 
    # Minimizing || x - C ||_1 is equivalent to minimizing || y ||_1 where y = x - C.
    # ADMM for: min_h || conv(h,h) - C*1 ||_1 s.t. sum(h)=1
    # Let z = conv(h,h). min_z || z - C*1 ||_1 s.t. z = conv(h,h), sum(h)=1.
    # This is still non-convex because z = conv(h,h) is a quadratic constraint.
    
    # Alternative Interpretation:
    # The "Idea" suggests minimizing || conv(h,h) - C*1 ||_1.
    # If we fix C, this is a convex problem? No, conv(h,h) is quadratic.
    # BUT, if we relax the problem to a convex one, we might find a better structure.
    
    # Let's try a different convex proxy:
    # Minimize || h * h - C * 1 ||_2^2 (least squares flatness)
    # This is quartic in h, non-convex.
    
    # Let's look at the "ADMM" hint again.
    # Maybe the "convex optimization" refers to the dual or a specific relaxation?
    # Or perhaps it means: Minimize || u ||_1 where u is the autoconvolution error,
    # but we treat the autoconvolution as a linear operator on the *squared* signal?
    
    # Actually, there is a known result that for binary or sparse signals,
    # flat autocorrelation is related to "complementary sequences" or "Zadoff-Chu".
    # But we have real values.
    
    # Let's try a direct ADMM-like approach on the quadratic form.
    # Minimize (1/2) || conv(h,h) - C*1 ||_2^2 + rho/2 || h - h_old ||_2^2 + dual_update
    # This is a quartic minimization, but we can solve it via gradient descent on h.
    
    # The prompt specifically asks for "convex optimization problem that minimizes the l1-norm... solved via ADMM".
    # This implies the problem *is* convex. 
    # Is it possible the problem is:
    # Minimize || v ||_1 subject to v = conv(h,h) - C*1, sum(h)=1?
    # If we consider h as the variable, it's non-convex.
    
    # Perhaps the "convex" part is the subproblem in ADMM?
    # Or maybe the strategy is to optimize the *spectrum*?
    # Let H = FFT(h). conv(h,h) <-> H * H (circular) or H^2 (linear).
    # If we minimize || ifft(H^2) - C ||_1, it's still non-convex in H.
    
    # Let's pivot to a strategy that *uses* ADMM structure for a *convex* relaxation
    # that might yield a better local optimum than the current smooth-max.
    # 
    # Strategy: 
    # 1. Use ADMM to solve: min_h || conv(h,h) - C*1 ||_2^2 s.t. sum(h)=1
    #    This is not convex, but we can use ADMM to split the quadratic constraint.
    #    Actually, let's stick to the prompt's "convex" claim by assuming 
    #    we are optimizing a proxy that is convex.
    #    
    #    Wait, if we fix the magnitude of the spectrum |H|^2, the phase determines the autocorrelation.
    #    Minimizing peak sidelobe is a classic problem.
    #    
    #    Let's try a "Convex Relaxation" approach:
    #    Minimize t subject to:
    #    conv(h,h) <= C*1 + t*1  (pointwise)
    #    sum(h) = 1
    #    This is still non-convex due to conv(h,h).
    #    
    #    Let's try the "ADMM for Quadratic Programming" idea.
    #    We minimize f(h) = || conv(h,h) - C*1 ||_1.
    #    We can use a soft-thresholding operator for the L1 part.
    #    The update for h would require solving:
    #    min_h || conv(h,h) - (z - u) ||_2^2 + (rho/2)||h - h_old||_2^2
    #    This is a quartic minimization. We can solve this with gradient descent.
    
    # Implementation Plan:
    # 1. Initialize h.
    # 2. ADMM loop:
    #    a. h-update: Minimize || conv(h,h) - z_tilde ||_2^2 + (rho/2)||h - h_prev||_2^2
    #       where z_tilde = z - u/rho.
    #       Use gradient descent with FFT-based convolution.
    #    b. z-update: Soft-thresholding of conv(h,h) + u/rho towards C*1.
    #       Actually, the objective is || conv(h,h) - C*1 ||_1.
    #       Let y = conv(h,h). Minimize || y - C*1 ||_1 s.t. y = conv(h,h).
    #       Lagrangian: || y - C*1 ||_1 + (rho/2)|| y - conv(h,h) ||_2^2 + <u, y - conv(h,h)>
    #       z-update (y-update): y = soft_threshold(conv(h,h) - u/rho, 1/rho) + C*1?
    #       Wait, prox of || . - C*1 ||_1 at x is C*1 + soft_threshold(x - C*1, 1/rho).
    #       So y_new = C*1 + soft_threshold(conv(h,h) - u/rho - C*1, 1/rho).
    #       y_new = C*1 + soft_threshold(conv(h,h) - u/rho - C*1, 1/rho).
    #       Let w = conv(h,h) - u/rho.
    #       y_new = C*1 + sign(w - C*1) * max(0, |w - C*1| - 1/rho).
    #    c. u-update: u = u + rho * (y_new - conv(h,h)).
    
    # This seems viable. The h-update is the expensive part (quartic minimization).
    # We can approximate it with a few steps of gradient descent.

    def fft_convolve(h):
        return np.fft.irfft(np.fft.rfft(h, n_fft) ** 2, n_fft)[:N]

    def grad_conv(h):
        # Gradient of || conv(h,h) - target ||_2^2 w.r.t h
        # conv(h,h) = ifft(fft(h)^2)
        # d/dh ifft(fft(h)^2) = 2 * ifft(fft(h) * fft(target_grad)) ?
        # Let J = conv(h,h).
        # <delta_h, J> = <delta_h, ifft(fft(h)^2)>
        # = <fft(delta_h), fft(h)^2>
        # = <2*fft(h)*delta_h, fft(h)> ... no.
        # Gradient of (1/2)||J - T||^2 is ifft(fft(J-T) * 2*fft(h)).
        # Let's verify.
        # L = || ifft(H^2) - T ||^2 = || H^2 - FFT(T) ||^2
        # dL/dH = 2 (H^2 - F(T)) * 2H = 4 H (H^2 - F(T))
        # dL/dh = ifft(4 H (H^2 - F(T)))
        # Note: H is complex. H^2 - F(T) is complex.
        # Since h is real, we take the real part.
        H = np.fft.rfft(h, n_fft)
        F_T = np.fft.rfft(target, n_fft)
        grad_H = 4.0 * H * (H**2 - F_T)
        return np.fft.irfft(grad_H, n_fft)[:N]

    def solve_h_update(z_tilde, h_init, rho, max_iters=50):
        # Minimize || conv(h,h) - z_tilde ||_2^2 + (rho/2)||h - h_init||_2^2
        # Gradient: 2 * ifft(fft(conv(h,h) - z_tilde) * 2*fft(h)) + rho*(h - h_init)
        # = 4 * ifft(fft(conv(h,h) - z_tilde) * fft(h)) + rho*(h - h_init)
        
        h = h_init.copy()
        # Simple gradient descent with backtracking or fixed step
        # Step size needs to be small.
        # Lipschitz constant of gradient?
        # For now, use a fixed step size and check for divergence.
        step = 1.0 / (rho + 10.0 * np.max(np.abs(h_init))) # Heuristic
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            conv_h = fft_convolve(h)
            grad = 4.0 * np.fft.irfft(np.fft.rfft(conv_h - z_tilde, n_fft) * np.fft.rfft(h, n_fft), n_fft)[:N]
            grad += rho * (h - h_init)
            
            # Backtracking line search
            f_old = np.sum((conv_h - z_tilde)**2) + 0.5 * rho * np.sum((h - h_init)**2)
            
            for k in range(20):
                h_new = h - step * grad
                conv_new = fft_convolve(h_new)
                f_new = np.sum((conv_new - z_tilde)**2) + 0.5 * rho * np.sum((h_new - h_init)**2)
                if f_new < f_old:
                    h = h_new
                    break
                step *= 0.5
            else:
                # If no step works, break
                break
                
        # Normalize sum to 1? No, the constraint is handled by the objective or separate.
        # In ADMM, we usually handle constraints via the h-update if it's a simple projection.
        # Here sum(h)=1 is a linear constraint.
        # We can include it in the h-update:
        # min || conv(h,h) - z_tilde ||^2 + (rho/2)||h - h_init||^2 s.t. sum(h)=1
        # This is harder. 
        # Let's enforce sum(h)=1 by projecting h onto the hyperplane sum(h)=1 after each update.
        # This is not strictly correct for the Lagrangian but often works.
        # Better: Add a penalty (mu/2)(sum(h)-1)^2 to the h-update objective.
        
        return h

    def admm_solve(h_init, C, rho=1.0, max_admm_iters=50):
        h = h_init.copy()
        z = fft_convolve(h)
        u = np.zeros(N)
        
        for it in range(max_admm_iters):
            if time.time() > deadline - 10:
                break
            
            # h-update
            z_tilde = z - u / rho
            # We want min || conv(h,h) - z_tilde ||^2 + (rho/2)||h - h_prev||^2 + penalty(sum(h)-1)^2
            # Let's just use the gradient descent solver and project
            h_new = solve_h_update(z_tilde, h, rho, max_iters=20)
            
            # Project to sum(h)=1
            s = np.sum(h_new)
            if abs(s) > 1e-10:
                h_new = h_new / s
            
            h = h_new
            
            # z-update
            conv_h = fft_convolve(h)
            # Prox of || . - C*1 ||_1
            # y = C*1 + soft_threshold(conv_h - u/rho - C*1, 1/rho)
            w = conv_h - u / rho - C * np.ones(N)
            z = C * np.ones(N) + np.sign(w) * np.maximum(np.abs(w) - 1.0/rho, 0)
            
            # u-update
            u = u + rho * (z - conv_h)
            
            # Check convergence
            if np.max(np.abs(z - conv_h)) < 1e-6:
                break
                
        return h

    # Initial guess: Square wave or random
    # Let's try a few initializations
    candidates = []
    
    # 1. Square wave
    sq = np.ones(N)
    sq[N//2:] = -1.0
    sq = sq / np.sum(sq) # Sum is 0, so this fails. 
    # Square wave has sum 0. We need sum != 0.
    # Add a small constant to make sum non-zero.
    sq_offset = sq + 0.1
    sq_offset = sq_offset / np.sum(sq_offset)
    candidates.append(sq_offset)
    
    # 2. Random
    np.random.seed(42)
    h_rand = np.random.randn(N)
    h_rand = h_rand / np.sum(h_rand)
    candidates.append(h_rand)
    
    # 3. Cosine window
    t = np.linspace(0, 1, N)
    h_cos = np.cos(np.pi * t)
    h_cos = h_cos / np.sum(h_cos)
    candidates.append(h_cos)

    best_h = candidates[0]
    best_val = float('inf')
    
    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return float('inf')
        conv = np.convolve(h, h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    # Evaluate initial candidates
    for h in candidates:
        v = compute_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Run ADMM for each candidate with different C and rho
    # C is the target constant for the autoconvolution.
    # Ideally, C should be close to the average autoconvolution value.
    # avg_conv = sum(conv(h,h)) / N = (sum(h))^2 / N = 1/N if sum(h)=1.
    
    C_values = [1.0/N, 2.0/N, 0.5/N]
    rho_values = [1.0, 5.0, 10.0]
    
    for h_init in candidates:
        for C in C_values:
            for rho in rho_values:
                if time.time() > deadline - 20:
                    break
                try:
                    h_admm = admm_solve(h_init, C, rho, max_admm_iters=30)
                    v = compute_value(h_admm)
                    if v < best_val:
                        best_val = v
                        best_h = h_admm.copy()
                except Exception:
                    pass

    # Final polish with L-BFGS on smooth max
    if time.time() < deadline - 5:
        def smooth_max(x, beta=100.0):
            m = np.max(x)
            return m + np.log(np.sum(np.exp((x - m) * beta))) / beta

        def obj(x):
            conv = np.convolve(x, x)
            return smooth_max(conv) + 100.0 * (np.sum(x) - 1.0)**2

        from scipy.optimize import minimize
        try:
            res = minimize(obj, best_h, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 200})
            h_polished = res.x
            v = compute_value(h_polished)
            if v < best_val:
                best_val = v
                best_h = h_polished.copy()
        except Exception:
            pass

    return best_h

if __name__ == "__main__":
    h = construct()
    print(compute_value(h))
