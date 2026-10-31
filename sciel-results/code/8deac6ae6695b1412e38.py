import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Strategy:
    # Represent h as a superposition of non-negative basis functions.
    # h(x) = sum_{k=0}^{M-1} c_k * phi_k(x)
    # where phi_k are band-limited trigonometric polynomials.
    # To ensure h >= 0, we can use squared magnitudes of complex basis functions
    # or simply optimize c_k and project to non-negative orthant, or use a softplus transform.
    # However, the task requires h_i to be the heights on the grid.
    # Let's use a linear combination of a few smooth basis vectors that span a subspace.
    # A better approach for "spectral design":
    # We want h such that |H(f)|^2 is as flat as possible (to minimize peak of h*h).
    # h*h in time domain corresponds to |H(f)|^2 in frequency domain (scaled).
    # Minimizing ||h*h||_inf is equivalent to minimizing the peak of the autocorrelation.
    
    # Let's parameterize h using the first M Discrete Cosine Transform (DCT) or FFT coefficients.
    # Since h is real and non-negative, we can work with a low-dimensional parameter space.
    
    # Basis: Use the first M eigenvectors of the identity? No, that's just coordinates.
    # Use smooth basis: Low frequency sine/cosine waves.
    # Let's define a basis matrix B of shape (N, M).
    # Rows of B are basis functions evaluated at the N grid points.
    
    M = 8  # Number of basis functions
    # Grid points
    x = np.linspace(-0.25, 0.25, N)
    
    # Construct basis: constant, and low freq cos/sin
    # We want basis functions that can form "flat" spectra.
    # Actually, let's just use a random smooth basis or DCT basis.
    # DCT basis is orthogonal and smooth.
    # Let's use the first M columns of the DCT-II matrix (normalized).
    
    # scipy.fft.dct
    try:
        from scipy.fft import dct
        # DCT-II
        # Note: scipy.fft.dct uses different normalizations. We want orthonormal.
        # Let's construct it manually to be safe or use scipy.fft.dct with norm='ortho'
        B_dct = dct(np.eye(N), type=2, norm='ortho')
        B = B_dct[:, :M]
    except ImportError:
        # Fallback to simple trigonometric basis
        B = np.zeros((N, M))
        B[:, 0] = 1.0 / np.sqrt(N)
        for k in range(1, M):
            # k-th harmonic
            B[:, k] = np.cos(2 * np.pi * k * x) / np.sqrt(N)
            if k < M:
                # Add sine components if M allows, or just use cos for simplicity
                pass # Let's stick to cos for now, or mix. 
                # Actually, for real signals, cos and sin are independent.
                # Let's use a mixed basis: 
                # idx 0: const
                # idx 1: cos(2pi x)
                # idx 2: sin(2pi x)
                # idx 3: cos(4pi x)
                # ...
                # This requires M to cover both.
                # Let's stick to DCT if possible, it's better.
                # If no DCT, let's use random smooth basis.
                pass

    # If DCT failed, generate random smooth basis
    if 'B_dct' not in locals():
        rng = np.random.default_rng(42)
        B = rng.normal(size=(N, M))
        # Smooth it by convolving with a small gaussian kernel
        kernel = np.array([1, 2, 1]) / 4.0
        for j in range(M):
            B[:, j] = np.convolve(B[:, j], kernel, mode='same')
        # Normalize columns
        norms = np.linalg.norm(B, axis=0)
        B = B / norms

    # We want to find coefficients c such that h = B @ c is non-negative and maximizes the ratio.
    # Ratio R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    # Since the ratio is scale invariant, we can fix ||h||_1 = 1.
    # Then maximize ||h||_2^2 / ||h*h||_inf.
    
    # Let's optimize c directly.
    # Constraint: h = B @ c >= 0.
    # This is a non-convex constraint if we just project.
    
    # Alternative: Use a soft parameterization.
    # Let c be unconstrained.
    # Define h = B @ (softplus(c) * scaling_factor)
    # Or simply optimize c with a penalty for negative h.
    
    # Let's try optimizing c with L-BFGS-B on the objective -log(R(h)) or -R(h).
    # We need to handle the non-negativity.
    # We can use a transformation: h_i = exp(u_i) ? No, h is linear combination.
    
    # Let's use the fact that B has smooth columns.
    # We will optimize the vector c.
    # To ensure h >= 0, we can add a large constant offset to c?
    # If we add a large positive constant to the first component (which is roughly constant),
    # h will likely be positive.
    
    # Let's define the objective function for scipy.optimize.minimize.
    # We want to maximize R(h).
    # We will minimize -R(h).
    
    def objective(c):
        h = B @ c
        # Enforce non-negativity via penalty or projection?
        # Projection is discontinuous. Penalty is smooth.
        # Let's use a penalty for negative values.
        neg_part = np.maximum(-h, 0)
        penalty = 100.0 * np.sum(neg_part**2)
        
        # Compute ratio
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 1e10 + penalty
        
        # If h has significant negative parts, ratio is ill-defined or bad.
        # Let's clamp h to 0 for the ratio calculation, but keep penalty high.
        h_clamped = np.maximum(h, 0)
        
        l1_c = np.sum(h_clamped)
        if l1_c < 1e-10:
            return 1e10 + penalty
            
        l2_sq = np.sum(h_clamped * h_clamped)
        
        n_fft = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h_clamped, n=n_fft)
        hh = np.fft.irfft(H * H, n=n_fft)
        hh_max = np.max(hh)
        
        if hh_max < 1e-10:
            return 1e10 + penalty
            
        ratio = l2_sq / (l1_c * hh_max)
        
        return -ratio + penalty

    # Initial guess:
    # Uniform distribution corresponds to c having a large component in the constant direction.
    # The first column of DCT is constant.
    # Let's initialize c with a large value in the first component and small random noise.
    
    c0 = np.zeros(M)
    c0[0] = 1.0  # Scale to make h roughly 1
    
    # Run optimization
    best_c = c0.copy()
    best_score = -1e10
    
    # Try a few different initializations
    rng = np.random.default_rng(42)
    
    for init_idx in range(5):
        if time.time() > deadline:
            break
            
        if init_idx == 0:
            c_init = c0.copy()
        else:
            # Random initialization with bias towards constant
            c_init = rng.normal(scale=0.1, size=M)
            c_init[0] = rng.uniform(0.5, 2.0)
            
        try:
            res = minimize(objective, c_init, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success or res.fun < best_score:
                best_c = res.x
                # Calculate actual score with clamped h
                h_test = np.maximum(B @ best_c, 0)
                score_test = compute_ratio(h_test)
                if score_test > best_score:
                    best_score = score_test
        except Exception as e:
            pass

    # Also try Differential Evolution for global search in low-dim space
    # DE is robust but slower. Given M=8, it might be feasible if bounded.
    # We need bounds for c.
    # Let's estimate bounds. If c is too large, penalty kills it.
    # Let's try DE with bounds [-5, 5] for each component.
    
    if time.time() < deadline - 30.0:
        bounds = [(-5, 5) for _ in range(M)]
        # DE requires a positive objective (minimization).
        # Our objective returns -ratio + penalty.
        # We want to minimize this.
        
        # We need to be careful with the penalty in DE context.
        # DE might get stuck in high penalty regions.
        
        def de_objective(c):
            h = B @ c
            neg_part = np.maximum(-h, 0)
            penalty = 100.0 * np.sum(neg_part**2)
            h_clamped = np.maximum(h, 0)
            l1_c = np.sum(h_clamped)
            if l1_c < 1e-10:
                return 1e10 + penalty
            l2_sq = np.sum(h_clamped * h_clamped)
            n_fft = 1 << (2 * N - 1).bit_length()
            H = np.fft.rfft(h_clamped, n=n_fft)
            hh = np.fft.irfft(H * H, n=n_fft)
            hh_max = np.max(hh)
            if hh_max < 1e-10:
                return 1e10 + penalty
            ratio = l2_sq / (l1_c * hh_max)
            return -ratio + penalty

        try:
            # DE is stochastic, use seed
            de_res = differential_evolution(de_objective, bounds, seed=42, maxiter=100, popsize=10, tol=1e-8)
            if de_res.fun < best_score: # Note: best_score is positive ratio, de_res.fun is negative ratio + penalty
                # Check if it's a valid improvement
                h_de = np.maximum(B @ de_res.x, 0)
                score_de = compute_ratio(h_de)
                if score_de > best_score:
                    best_score = score_de
                    best_c = de_res.x
        except Exception as e:
            pass

    # Final refinement:
    # Take the best c found, compute h, and do a local polish on h directly?
    # Or just return h from best_c.
    
    h_final = np.maximum(B @ best_c, 0.0)
    
    # One last check: if the score is bad, fall back to a simple heuristic?
    # The spectral approach should be decent.
    
    # Normalize h? The ratio is scale invariant, so normalization doesn't change score,
    # but returning small numbers might be fine. The task asks for "heights".
    # Let's return the h as is.
    
    return list(h_final)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
