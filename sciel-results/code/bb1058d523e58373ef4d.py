"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _deadline_guard(timeout=140.0):
    """Return a function that checks if we've exceeded the deadline."""
    start = time.time()
    def check():
        return (time.time() - start) < timeout
    return check

def _objective_and_grad(x):
    """
    Objective: value = abs(2 * N * max(conv(h,h)) / sum(h)^2)
    We minimize this. Since we want lower value, and the expression is positive,
    we minimize the raw value.
    
    x: array of 400 heights
    Returns: (value, gradient)
    """
    h = x
    S = np.sum(h)
    if abs(S) < 1e-12:
        return 1e10, np.zeros_like(h)
    
    # Full autoconvolution
    conv = np.convolve(h, h, mode='full')
    M = np.max(conv)
    
    value = 2.0 * N * M / (S * S)
    
    # Gradient computation
    # value = 2*N * M / S^2
    # d(value)/d(h_i) = 2*N * (dM/dh_i * S^2 - M * 2*S * dS/dh_i) / S^4
    #                  = 2*N * (dM/dh_i * S - 2*M * dS/dh_i) / S^3
    # dS/dh_i = 1 for all i
    # dM/dh_i: M = max(conv), so dM/dh_i = sum over j where conv[j] is max of d(conv[j])/dh_i
    
    # conv[j] = sum_k h[k] * h[j-k] (for valid j in 0..2N-2)
    # d(conv[j])/dh_i = h[j-i] + h[i-j] if indices valid, else 0
    
    # Find indices where conv == M (with tolerance for numerical stability)
    max_idx = np.where(np.abs(conv - M) < 1e-10 * max(1.0, abs(M)))[0]
    
    grad_M = np.zeros(N)
    for j in max_idx:
        # conv[j] = sum_{k=0}^{N-1} h[k]*h[j-k]
        # d(conv[j])/dh_i = h[j-i] (if 0<=j-i<N) + h[i-j] (if 0<=i-j<N)
        # Note: if i == j-i, i.e., j=2i, then it's 2*h[i]
        for i in range(N):
            a = j - i
            b = i - j
            val = 0.0
            if 0 <= a < N:
                val += h[a]
            if 0 <= b < N:
                val += h[b]
            grad_M[i] += val
    
    dS = np.ones(N)
    grad = 2.0 * N * (grad_M * S - 2.0 * M * dS) / (S ** 3)
    
    return value, grad

def _fast_objective(x):
    """Fast objective for initial searches without gradient."""
    h = x
    S = np.sum(h)
    if abs(S) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    M = np.max(conv)
    return 2.0 * N * M / (S * S)

def construct():
    start_time = time.time()
    timeout = 140.0  # Leave 10s margin
    
    best_val = 2.0
    best_h = np.ones(N)
    
    def check_time():
        return (time.time() - start_time) < timeout
    
    # Strategy: Use a low-dimensional parameterization via smooth basis functions
    # h[i] = sum_k c_k * phi_k(i/N)
    # where phi_k are smooth basis functions (e.g., Gaussian bumps, polynomials)
    
    # Generate diverse initial candidates using different smooth profiles
    candidates = []
    
    # 1. Constant (baseline)
    candidates.append(np.ones(N))
    
    # 2. Smooth bumps of various widths and positions
    n_basis = 8
    for width in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for center in [0.0, 0.1, -0.1, 0.2, -0.2]:
            x = np.linspace(-0.25, 0.25, N)
            bump = np.exp(-((x - center)**2) / (2 * width**2))
            # Add a small constant to avoid zero sum
            h = 0.5 + 0.5 * bump
            candidates.append(h)
    
    # 3. Polynomial basis
    for deg in [1, 2, 3]:
        x = np.linspace(-0.25, 0.25, N)
        for coeff in np.random.randn(deg + 1) * 0.1 + 0.5:
            poly = np.zeros(N)
            for d in range(deg + 1):
                poly += coeff * x**d if d < len(coeff) else 0
            # Actually let's do this properly
            poly = np.zeros(N)
            for d in range(min(deg + 1, len(coeff))):
                poly += coeff[d] * x**d
            if np.sum(poly) > 0.1:
                candidates.append(poly)
    
    # 4. Chebyshev-like basis
    x = np.linspace(-1, 1, N)
    for k in range(1, 6):
        cheb = np.cos(k * np.arccos(x))
        h = 0.5 + 0.3 * cheb
        if np.sum(h) > 0.1:
            candidates.append(h)
    
    # Evaluate all candidates and pick top ones
    scored = []
    for h in candidates:
        if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-12:
            continue
        val = _fast_objective(h)
        scored.append((val, h))
        if not check_time():
            break
    
    scored.sort(key=lambda x: x[0])
    
    # Keep top candidates
    top_candidates = [h for _, h in scored[:min(20, len(scored))]]
    if not top_candidates:
        top_candidates = [np.ones(N)]
    
    # Update best
    for h in top_candidates:
        val = _fast_objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Now refine top candidates using L-BFGS-B with analytic gradient
    for h0 in top_candidates[:10]:
        if not check_time():
            break
        
        try:
            result = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_val:
                val = result.fun
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            continue
    
    # If we have time, try a few more random smooth starts
    if check_time():
        for trial in range(15):
            if not check_time():
                break
            
            # Random smooth profile
            x = np.linspace(-0.25, 0.25, N)
            # Use a random combination of Gaussians
            n_gauss = np.random.randint(2, 5)
            h = np.zeros(N)
            for _ in range(n_gauss):
                center = np.random.uniform(-0.25, 0.25)
                width = np.random.uniform(0.05, 0.3)
                amp = np.random.uniform(-1.0, 1.0)
                h += amp * np.exp(-((x - center)**2) / (2 * width**2))
            
            # Ensure positive sum
            if np.sum(h) < 0:
                h = -h
            
            if abs(np.sum(h)) < 0.1:
                h += 0.5
            
            val = _fast_objective(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
            
            # Refine
            if check_time():
                try:
                    result = minimize(
                        _objective_and_grad,
                        h,
                        method='L-BFGS-B',
                        jac=True,
                        options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if result.fun < best_val:
                        best_val = result.fun
                        best_h = result.x.copy()
                except Exception:
                    pass
    
    return best_h
# EVOLVE-BLOCK-END
