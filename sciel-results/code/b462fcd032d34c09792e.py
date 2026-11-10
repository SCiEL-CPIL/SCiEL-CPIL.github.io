# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

def rudin_problem10() -> np.ndarray:
    k = 10
    n_coeffs = 2 * k + 1
    
    # Deadline guard
    t_start = time.time()
    deadline = 620.0  # 10s margin
    
    # Quadrature setup
    # 70 Gauss-Legendre points in cos(theta)
    x, w_theta = np.polynomial.legendre.leggauss(70)
    
    N_phi = 130
    phi = np.arange(N_phi) * (2 * np.pi / N_phi)
    
    # Weights for the full sphere integral
    w_sphere = w_theta[:, None] * (2 * np.pi / N_phi)  # Shape (70, 130)
    
    # Precompute Basis Functions Y_m(x_j, phi_i)
    Y_vals = np.zeros((n_coeffs, 70, 130), dtype=np.float64)
    
    for idx, m in enumerate(range(-k, k + 1)):
        m_abs = abs(m)
        num = (k - m_abs)
        den = (k + m_abs)
        ratio = 1.0
        if k + m_abs != k - m_abs:
            for i in range(k - m_abs + 1, k + m_abs + 1):
                ratio /= i
        norm_const = np.sqrt((2 * k + 1) / (4 * np.pi) * ratio)
        
        P = lpmv(m_abs, k, x)
        
        if m == 0:
            Y_vals[idx, :, :] = norm_const * P[:, None] * np.ones((1, 130))
        elif m > 0:
            cos_m_phi = np.cos(m * phi)
            Y_vals[idx, :, :] = (np.sqrt(2) * norm_const * P[:, None]) * cos_m_phi[None, :]
        else:
            sin_m_phi = np.sin(m_abs * phi)
            Y_vals[idx, :, :] = (np.sqrt(2) * norm_const * P[:, None]) * sin_m_phi[None, :]
            
    Y_flat = Y_vals.reshape(n_coeffs, -1)
    w_flat = w_sphere.flatten()
    
    # Objective and Gradient
    def objective(c):
        f = Y_flat @ c
        U = np.dot(w_flat, f**4)
        V = np.dot(c, c)
        if V == 0:
            return 1e10, np.zeros_like(c)
        val = U / (V**2)
        
        f_cubed = f**3
        grad_U = 4.0 * (Y_flat.T @ (w_flat * f_cubed))
        
        grad_F = grad_U / (V**2) - (4.0 * U / (V**3)) * c
        
        return val, grad_F

    best_c = np.ones(n_coeffs) / np.sqrt(n_coeffs)
    best_score = np.inf
    
    # Generate diverse initializations
    def generate_initializations():
        inits = []
        
        # 1. Equal amplitude (flat spectrum)
        inits.append(np.ones(n_coeffs) / np.sqrt(n_coeffs))
        
        # 2. Random uniform
        for _ in range(100):
            c0 = np.random.uniform(-1, 1, n_coeffs)
            norm = np.linalg.norm(c0)
            if norm > 1e-6:
                inits.append(c0 / norm)
        
        # 3. Random normal
        for _ in range(100):
            c0 = np.random.randn(n_coeffs)
            norm = np.linalg.norm(c0)
            if norm > 1e-6:
                inits.append(c0 / norm)
        
        # 4. Sparse high-m emphasis
        for _ in range(50):
            c0 = np.zeros(n_coeffs)
            m_target = np.random.randint(5, k + 1)
            idx_pos = m_target + k
            idx_neg = -m_target + k
            c0[idx_pos] = np.random.randn()
            c0[idx_neg] = np.random.randn()
            norm = np.linalg.norm(c0)
            if norm > 1e-6:
                inits.append(c0 / norm)
        
        # 5. Sparse low-m emphasis
        for _ in range(50):
            c0 = np.zeros(n_coeffs)
            m_target = np.random.randint(0, 5)
            idx_pos = m_target + k
            idx_neg = -m_target + k
            if m_target == 0:
                c0[idx_pos] = np.random.randn()
            else:
                c0[idx_pos] = np.random.randn()
                c0[idx_neg] = np.random.randn()
            norm = np.linalg.norm(c0)
            if norm > 1e-6:
                inits.append(c0 / norm)
        
        # 6. Alternating sign patterns
        for _ in range(50):
            c0 = np.ones(n_coeffs)
            c0[::2] = -1
            c0 = c0 + np.random.randn(n_coeffs) * 0.1
            norm = np.linalg.norm(c0)
            if norm > 1e-6:
                inits.append(c0 / norm)
        
        return inits

    # Two-stage optimization: coarse then fine
    def optimize_from(c0, stage=0):
        if stage == 0:
            # Coarse stage
            options = {'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        else:
            # Fine stage
            options = {'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
        
        try:
            res = minimize(objective, c0, method='L-BFGS-B', jac=True, options=options)
            if res.success or res.status == 0:
                return res.x
        except Exception:
            pass
        return c0

    # Phase 1: Coarse optimization from many initializations
    inits = generate_initializations()
    coarse_results = []
    
    for i, c0 in enumerate(inits):
        if time.time() > deadline - 120:  # Leave 120s for fine-tuning
            break
        
        c_opt = optimize_from(c0, stage=0)
        score, _ = objective(c_opt)
        if score < best_score:
            best_score = score
            best_c = c_opt
        coarse_results.append(c_opt)
        
        if (i + 1) % 20 == 0:
            print(f"Coarse: {i+1}/{len(inits)}, best={best_score:.10f}")

    # Phase 2: Fine optimization from best coarse results
    if time.time() < deadline - 30:
        # Sort by score and take top candidates
        scored = [(objective(c)[0], c) for c in coarse_results]
        scored.sort(key=lambda x: x[0])
        top_candidates = [c for _, c in scored[:20]]
        
        # Add current best
        if best_c not in top_candidates:
            top_candidates.append(best_c)
        
        for i, c0 in enumerate(top_candidates):
            if time.time() > deadline - 10:
                break
            
            c_opt = optimize_from(c0, stage=1)
            score, _ = objective(c_opt)
            if score < best_score:
                best_score = score
                best_c = c_opt

    # Final cleanup
    if not isinstance(best_c, np.ndarray) or best_c.shape != (n_coeffs,):
        return np.ones(n_coeffs)
        
    max_c = np.max(np.abs(best_c))
    if max_c < 1e-3:
        best_c = best_c * (1e-2 / max_c)
    if max_c > 1e5:
        best_c = best_c * (1e4 / max_c)
        
    return best_c

# EVOLVE-BLOCK-END
