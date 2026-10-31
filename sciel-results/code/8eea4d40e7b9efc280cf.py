import numpy as np
import time

N = 50

def _autoconv_max(h):
    """Compute supremum norm of autoconvolution of h."""
    h = np.asarray(h, dtype=float)
    # Linear convolution
    conv = np.convolve(h, h, mode='full')
    return np.max(np.abs(conv))

def _score(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-30:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    sup_norm = np.max(np.abs(conv))
    if sup_norm < 1e-30:
        return 0.0
    return l2_sq / (l1 * sup_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Strategy: Parametric optimization over smooth families
    # The 1.014 solution used triangular/ramp/Gaussian families with coordinate descent.
    # We will try a combination of:
    # 1. Gaussian-like shapes
    # 2. Triangular (tent) shapes
    # 3. Cosine-bell shapes
    # 4. Optimized linear combinations of basis functions
    
    # Basis functions (all non-negative)
    x = np.linspace(-1, 1, N)
    
    # Gaussian with varying sigma
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    basis = []
    for sigma in sigmas:
        g = np.exp(-x**2 / (2*sigma**2))
        basis.append(g)
    
    # Triangular (tent) functions
    for center in np.linspace(-0.8, 0.8, 5):
        width = 0.5
        t = np.maximum(0, 1 - np.abs(x - center)/width)
        basis.append(t)
    
    # Cosine bell
    for width in [0.3, 0.5, 0.7, 1.0]:
        c = np.maximum(0, np.cos(np.pi * x / (2*width)))
        c[x**2 > (2*width)**2] = 0  # Ensure compact support
        basis.append(c)
    
    # Uniform
    basis.append(np.ones(N))
    
    # Ramp (asymmetric)
    for slope in [0.5, 1.0, 2.0]:
        r = np.clip(slope * (x + 1), 0, 1)
        basis.append(r)
    
    basis = np.array(basis)
    n_basis = len(basis)
    
    # Coordinate descent on non-negative linear combination
    # h = sum(alpha_i * basis_i), alpha_i >= 0
    # We optimize to maximize score(h)
    
    # Initialize with uniform
    alpha = np.zeros(n_basis)
    alpha[-1] = 1.0  # Uniform is last basis function
    
    current_h = alpha @ basis
    current_score = _score(current_h)
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(n_basis):
            # Try adjusting alpha[i]
            # Simple coordinate descent: try a few step sizes
            for delta in [0.01, 0.05, 0.1, 0.2, 0.5]:
                for sign in [1, -1]:
                    new_alpha = alpha.copy()
                    new_alpha[i] = max(0, alpha[i] + sign * delta)
                    
                    # Normalize to avoid trivial scaling issues
                    # But the score is scale-invariant, so normalization doesn't matter
                    new_h = new_alpha @ basis
                    
                    if np.all(new_h < 0):
                        continue
                    
                    # Ensure non-negative
                    new_h = np.maximum(new_h, 0)
                    
                    s = _score(new_h)
                    if s > current_score:
                        current_score = s
                        current_h = new_h
                        alpha = new_alpha
                        improved = True
        
        if not improved:
            # Try finer steps
            for i in range(n_basis):
                for delta in [0.001, 0.005, 0.01]:
                    for sign in [1, -1]:
                        new_alpha = alpha.copy()
                        new_alpha[i] = max(0, alpha[i] + sign * delta)
                        new_h = np.maximum(new_alpha @ basis, 0)
                        s = _score(new_h)
                        if s > current_score:
                            current_score = s
                            current_h = new_h
                            alpha = new_alpha
                            improved = True
        
        if not improved:
            break
    
    # Also try direct optimization of individual parametric families
    families = []
    
    # Gaussian family: h = exp(-a*(x-c)^2), a>0, c in [-0.5, 0.5]
    for a in np.linspace(0.5, 5.0, 10):
        for c in np.linspace(-0.5, 0.5, 5):
            h = np.exp(-a * (x - c)**2)
            s = _score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
    
    # Triangular family: h = max(0, 1 - |x-c|/w)
    for c in np.linspace(-0.5, 0.5, 5):
        for w in np.linspace(0.2, 1.0, 10):
            h = np.maximum(0, 1 - np.abs(x - c)/w)
            s = _score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
    
    # Cosine bell: h = max(0, cos(pi*x/(2w)))
    for w in np.linspace(0.3, 1.0, 10):
        h = np.maximum(0, np.cos(np.pi * x / (2*w)))
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Use the best from coordinate descent
    if current_score > best_score:
        best_score = current_score
        best_h = current_h.copy()
    
    # Final polish: local search around best_h
    if time.time() < deadline - 20:
        for _ in range(100):
            if time.time() > deadline:
                break
            # Perturb best_h
            h_pert = best_h + 0.05 * np.random.randn(N)
            h_pert = np.maximum(h_pert, 0)
            s = _score(h_pert)
            if s > best_score:
                best_score = s
                best_h = h_pert.copy()
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    # Normalize to avoid numerical issues (score is scale-invariant, but keep reasonable magnitude)
    if np.max(best_h) > 0:
        best_h = best_h / np.max(best_h) * 1.0
    
    return list(best_h)
