import numpy as np
from scipy.optimize import minimize

def construct():
    N = 600
    L = 0.5
    # Grid for f
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    
    # Number of harmonics
    M = 15
    
    # Precompute basis matrix B (N x M+1)
    # f(x) = sum_{k=0}^M u_k * phi_k(x)
    # phi_0 = 1
    # phi_k = cos(2*pi*k*x/L)
    # Note: We don't need normalization for the shape, but it helps for optimization stability.
    # Let's use orthonormal basis to make energy calculation simple: sum(u^2) * norm_factor?
    # Actually, let's just use raw cosines.
    # B[i, 0] = 1
    # B[i, k] = cos(2*pi*k*x[i]/L)
    
    B = np.zeros((N, M+1))
    B[:, 0] = 1.0
    for k in range(1, M+1):
        B[:, k] = np.cos(2 * np.pi * k * x / L)
    
    # Objective function
    def objective(u):
        f = B @ u
        # Penalty for negative values
        neg = np.maximum(0, -f)
        penalty = np.sum(neg**2)
        
        if penalty > 1e-6:
            # If significantly negative, return a large score
            # But we need to guide the optimizer, so return penalty weighted
            pass
            
        # Compute autoconvolution
        # Zero pad to 2*N to avoid circular convolution artifacts
        # f is defined on [-L/2, L/2]. Support of f*f is [-L, L].
        # Length of f is N. Length of f*f is 2N-1.
        # Use FFT size 2*N or 4*N for safety.
        N_fft = 2 * N
        F = np.fft.fft(f, n=N_fft)
        G = np.fft.ifft(F * F).real
        
        # G is defined on a grid corresponding to 2N points.
        # The peak of G is what we care about.
        # Energy of f: sum(f^2) * dx
        # dx = L / N
        dx = L / N
        energy = np.sum(f**2) * dx
        
        if energy < 1e-12:
            return 1e10
            
        # Normalized max
        score = np.max(G) / energy
        
        # Add penalty for non-negativity
        # Scale penalty to be comparable to score
        score += 10.0 * (penalty / energy)
        
        return score

    # Initial guess: DC component only (uniform)
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    # Optimize
    # L-BFGS-B is good for this.
    # We need to provide gradients for speed? 
    # Gradient of max(G) is hard. 
    # Let's try Nelder-Mead first as it's robust to non-differentiability, 
    # but it might be slow. 
    # With M=15, Nelder-Mead might take too long.
    
    # Alternative: Use the fact that the optimal shape is likely a "bell curve".
    # We can parameterize u to ensure f >= 0?
    # f(x) = (sum u_k phi_k(x))^2 ? No, that changes the function class.
    
    # Let's use L-BFGS-B with a smooth approximation of max.
    # Or, simply use a few random restarts with a simple gradient step on the coefficients?
    
    # Better idea: The function that minimizes the peak of autoconvolution for a given L2 norm 
    # among non-negative functions is related to the eigenfunction of the convolution operator?
    # Actually, for a fixed support, the "best" function is often a raised cosine or similar.
    
    # Let's try to optimize u using scipy minimize with L-BFGS-B.
    # To handle the non-differentiable max, we can use a soft-max approximation or just rely on 
    # the optimizer navigating the landscape.
    
    # However, a simpler and very effective heuristic for this specific problem (minimizing peak 
    # autoconvolution of a non-negative bump) is to use a super-Gaussian or a polynomial bump 
    # that is zero at the boundaries.
    # f(x) = (1 - (x/(L/2))^p)^q for |x| < L/2, 0 otherwise.
    # We can optimize p and q.
    # But the directive says "optimize these coefficients".
    
    # Let's stick to optimizing u.
    # I will use a small number of iterations of a custom gradient descent with 
    # numerical gradient estimation if L-BFGS-B fails or is too slow.
    
    # Actually, let's just run minimize with L-BFGS-B and hope the max is smooth enough 
    # or that the penalty guides it well.
    # To improve convergence, I'll use a slightly larger M and a good initial guess.
    
    # A good initial guess is a cosine bell:
    # f(x) = cos(pi * x / L) for |x| < L/2? 
    # cos(pi * x / L) is zero at +/- L/2.
    # This corresponds to u_0 = 0.5, u_1 = 0.5 (if basis is 1, cos(2pi x/L)?)
    # Wait, period is L. 
    # cos(2*pi*x/L) has period L. 
    # At x=L/2, cos(pi) = -1.
    # We want f(L/2)=0.
    # f(x) = 1 + cos(2*pi*x/L) is zero at L/2? 
    # 1 + cos(pi) = 0. Yes.
    # So u_0 = 1, u_1 = 1 gives a non-negative bump that is zero at the edges.
    
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    u0[1] = 1.0
    
    res = minimize(objective, u0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
    
    best_u = res.x
    best_score = res.fun
    
    # Check if we can do better with a different start
    # Try a wider bell
    u0_2 = np.zeros(M+1)
    u0_2[0] = 1.0
    # Just in case
    
    # Evaluate best_u
    f = B @ best_u
    # Ensure non-negativity by clipping (should be close to 0 if penalty worked)
    f = np.maximum(0, f)
    
    # Return the heights
    return list(f)
