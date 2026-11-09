"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # The grid: 50 points on [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, N)
    dx = x[1] - x[0]
    
    # Autoconvolution grid: 99 points on [-1/2, 1/2]
    # h*h at point x_i + x_j
    # We compute the full convolution
    def autoconv(h):
        # h is length N, autoconv is length 2N-1 = 99
        return np.convolve(h, h)
    
    def compute_ratio(h):
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)  # enforce nonnegativity
        norm2_sq = np.sum(h**2)
        if norm2_sq < 1e-15:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-15:
            return 0.0
        ac = np.convolve(h, h)
        sup_ac = np.max(np.abs(ac))
        if sup_ac < 1e-15:
            return 0.0
        return norm2_sq / (norm1 * sup_ac)
    
    # Objective to minimize: -log(ratio) = -log(norm2_sq) + log(norm1) + log(sup_ac)
    # Since we fix norm2_sq = 1, minimize: log(norm1) + log(sup_ac)
    # Or directly minimize norm1 * sup_ac subject to norm2_sq = 1, h >= 0
    
    def objective(h):
        """Minimize ||h||_1 * ||h*h||_inf subject to ||h||_2 = 1, h >= 0"""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h**2)
        if norm2_sq < 1e-15:
            return 1e10
        # Normalize to unit L2
        h_norm = h / np.sqrt(norm2_sq)
        norm1 = np.sum(h_norm)
        ac = np.convolve(h_norm, h_norm)
        sup_ac = np.max(np.abs(ac))
        if sup_ac < 1e-15:
            return 1e10
        return norm1 * sup_ac
    
    def gradient(h):
        """Gradient of ||h||_1 * ||h*h||_inf subject to ||h||_2 = 1, h >= 0"""
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h**2)
        if norm2_sq < 1e-15:
            return np.zeros(N)
        h_norm = h / np.sqrt(norm2_sq)
        
        norm1 = np.sum(h_norm)
        ac = np.convolve(h_norm, h_norm)
        sup_ac = np.max(np.abs(ac))
        
        if sup_ac < 1e-15 or norm1 < 1e-15:
            return np.zeros(N)
        
        # Gradient of norm1 * sup_ac
        # d(norm1)/dh_i = 1/sqrt(norm2_sq) for h_i > 0
        # d(sup_ac)/dh_i: need to find where max occurs
        idx_max = np.argmax(np.abs(ac))
        # ac[k] = sum_j h[j] * h[k-j] (with appropriate indices)
        # For full convolution, ac[k] = sum_{j} h_norm[j] * h_norm[k-j] where valid indices
        # d(ac[k])/dh_norm[i] = h_norm[k-i] + h_norm[i] (by symmetry, but let's be careful)
        # Actually for autoconvolution: ac[k] = sum_j h[j]*h[k-j]
        # d(ac[k])/dh[i] = h[k-i] + h[i] if both indices valid, else one term
        
        # Let's compute the gradient numerically for reliability
        eps = 1e-7
        grad = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_plus[i] += eps
            h_minus = h.copy()
            h_minus[i] -= eps
            
            def val(hp):
                hp = np.maximum(hp, 0.0)
                n2 = np.sum(hp**2)
                if n2 < 1e-15:
                    return 1e10
                hn = hp / np.sqrt(n2)
                n1 = np.sum(hn)
                acv = np.convolve(hn, hn)
                sac = np.max(np.abs(acv))
                return n1 * sac
            
            grad[i] = (val(h_plus) - val(h_minus)) / (2 * eps)
        
        return grad
    
    # Multi-start optimization
    best_score = 0.0
    best_h = None
    
    # Generate diverse initial points
    rng = np.random.RandomState(42)
    n_starts = 20
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if s == 0:
            # Uniform
            h0 = np.ones(N)
        elif s == 1:
            # Center concentrated
            h0 = np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/4)**2))
        elif s == 2:
            # Edge concentrated
            h0 = np.zeros(N)
            h0[0] = 1.0
            h0[-1] = 1.0
        elif s == 3:
            # Sparse random
            h0 = np.zeros(N)
            idx = rng.choice(N, size=N//5, replace=False)
            h0[idx] = 1.0
        elif s == 4:
            # Triangular
            h0 = 1.0 - np.abs(np.arange(N) - N/2) / (N/2)
            h0 = np.maximum(h0, 0)
        elif s == 5:
            # Cosine bump
            h0 = np.cos(np.pi * np.arange(N) / N) ** 2
            h0 = np.maximum(h0, 0)
        elif s == 6:
            # Random positive
            h0 = np.abs(rng.randn(N)) + 0.1
        elif s == 7:
            # Two peaks
            h0 = np.zeros(N)
            h0[N//4] = 1.0
            h0[3*N//4] = 1.0
        elif s == 8:
            # Gaussian narrow
            h0 = np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/10)**2))
        elif s == 9:
            # Gaussian wide
            h0 = np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/2)**2))
        else:
            # Random mix
            h0 = np.abs(rng.randn(N)) * np.exp(-((np.arange(N) - N/2)**2) / (2 * (N/3)**2))
        
        h0 = np.maximum(h0, 0.0)
        
        # Normalize to unit L2
        n2 = np.sum(h0**2)
        if n2 < 1e-15:
            continue
        h0 = h0 / np.sqrt(n2)
        
        # Optimize
        res = minimize(
            objective,
            h0,
            jac=gradient,
            method='L-BFGS-B',
            bounds=[(0.0, None)] * N,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() > deadline:
            break
        
        h_opt = np.maximum(res.x, 0.0)
        n2 = np.sum(h_opt**2)
        if n2 < 1e-15:
            continue
        h_opt = h_opt / np.sqrt(n2)
        
        score = compute_ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Polish with more iterations on best
    if best_h is not None and time.time() < deadline - 5.0:
        res = minimize(
            objective,
            best_h,
            jac=gradient,
            method='L-BFGS-B',
            bounds=[(0.0, None)] * N,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        h_polish = np.maximum(res.x, 0.0)
        n2 = np.sum(h_polish**2)
        if n2 > 1e-15:
            h_polish = h_polish / np.sqrt(n2)
            score = compute_ratio(h_polish)
            if score > best_score:
                best_score = score
                best_h = h_polish
    
    if best_h is None:
        return [1.0] * N
    
    return list(best_h)
# EVOLVE-BLOCK-END
