# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _area_fast(x, n=64, M=20000):
    """Compute union area via quadrature at M heights."""
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    # L shape: (M, n)
    L = x[None, :] + a[None, :] * y[:, None]
    Ls = np.sort(L, axis=1)
    gaps = Ls[:, 1:] - Ls[:, :-1]
    union_len = w + np.sum(np.minimum(gaps, w[:, None]), axis=1)
    return float(np.mean(union_len))

def _area_coarse(x, n=64, M=2000):
    """Coarse area for fast screening."""
    return _area_fast(x, n, M)

def _aligned_positions(n=64):
    """
    Initialize such that all triangle apices converge to x=0.5 at y=1.
    Vertex at y=1 is (x_i + a_i, 1).
    We want x_i + a_i = 0.5 => x_i = 0.5 - a_i.
    """
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    return 0.5 - a

def _gradient(x, n=64, M=1000, h=1e-5):
    """
    Compute gradient of area with respect to x using central differences.
    Optimized to use vectorization where possible, though loop over n is likely faster 
    than creating huge temporary arrays for each coordinate if not careful.
    Here we use a loop for clarity and memory safety, but vectorized area calc.
    """
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (_area_fast(x_plus, n, M) - _area_fast(x_minus, n, M)) / (2 * h)
    return grad

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 600.0
    safety = 30.0
    
    # 1. Generate diverse candidates
    candidates = []
    
    # Primary: Aligned positions (converging to a point)
    aligned = _aligned_positions(n)
    candidates.append(aligned)
    
    # Variants of aligned
    for shift in np.linspace(-0.1, 0.1, 10):
        candidates.append(aligned + shift)
    
    # Random candidates for robustness
    rng = np.random.default_rng(42)
    for _ in range(20):
        candidates.append(rng.uniform(0.0, 1.0, size=n))
    
    # Screen with coarse area
    best_x = None
    best_coarse_area = np.inf
    fine_candidates = []
    
    for i, x in enumerate(candidates):
        if time.time() > deadline - safety - 60.0:
            break
        area_c = _area_coarse(x, n, 2000)
        if area_c < best_coarse_area:
            best_coarse_area = area_c
            best_x = x.copy()
        fine_candidates.append((area_c, x))
    
    # Fine-evaluate top 5 coarse candidates
    fine_candidates.sort(key=lambda t: t[0])
    top_n = min(5, len(fine_candidates))
    
    best_fine_area = np.inf
    best_fine_x = None
    
    for i in range(top_n):
        if time.time() > deadline - safety - 60.0:
            break
        x = fine_candidates[i][1]
        area_f = _area_fast(x, n, 20000)
        if area_f < best_fine_area:
            best_fine_area = area_f
            best_fine_x = x.copy()
    
    if best_fine_x is not None:
        best_x = best_fine_x
        best_area = best_fine_area
    else:
        best_area = best_coarse_area
    
    # 2. Gradient-based optimization (L-BFGS-B)
    # We use a coarse M for the gradient to keep it fast, but evaluate final with fine M.
    # M=1000 is a good balance.
    M_opt = 1000
    
    # Define objective and gradient functions for SciPy
    def objective(x_opt):
        return _area_fast(x_opt, n, M_opt)
    
    def gradient(x_opt):
        return _gradient(x_opt, n, M_opt)
    
    # Try optimization from the best coarse candidate and the aligned candidate
    start_points = [best_x, aligned]
    
    # Remove duplicates if any
    unique_starts = []
    for sp in start_points:
        if not any(np.allclose(sp, u) for u in unique_starts):
            unique_starts.append(sp)
    
    for x_start in unique_starts:
        if time.time() > deadline - safety - 30.0:
            break
        
        try:
            res = minimize(
                objective, 
                x_start, 
                method='L-BFGS-B', 
                jac=gradient,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if res.success or res.fun < best_area:
                # Fine evaluate the result
                area_fine = _area_fast(res.x, n, 20000)
                if area_fine < best_area:
                    best_x = res.x
                    best_area = area_fine
        except Exception:
            pass
            
    # 3. Final local polish with coordinate descent if time permits
    if best_x is not None and time.time() < deadline - safety - 10.0:
        remaining = deadline - safety - time.time() - 5.0
        if remaining > 10.0:
            # Simple coordinate descent polish
            x = best_x.copy()
            current_area = best_area
            step_size = 0.01
            rng = np.random.default_rng(123)
            
            # Use a tighter M for polishing
            M_polish = 5000
            
            start_time = time.time()
            while time.time() - start_time < remaining * 0.8:
                improved = False
                idx = rng.integers(n)
                for sign in [1.0, -1.0]:
                    x_new = x.copy()
                    x_new[idx] += sign * step_size * rng.uniform(0.5, 1.5)
                    area_new = _area_fast(x_new, n, M_polish)
                    if area_new < current_area:
                        x = x_new
                        current_area = area_new
                        improved = True
                        break
                if not improved:
                    step_size *= 0.9
                    if step_size < 1e-6:
                        break
            
            if current_area < best_area:
                best_x = x
                best_area = current_area

    if best_x is None:
        best_x = aligned
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
