# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.integrate import gauss_legendre


def _build_basis():
    """Build real orthonormal degree-10 spherical harmonics on a quadrature grid."""
    k = 10
    n_theta = 70
    n_phi = 130
    
    mu, w = gauss_legendre(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    basis = np.zeros((21, n_theta, n_phi), dtype=np.float64)
    
    for m_idx in range(21):
        m = m_idx - 10
        abs_m = abs(m)
        
        # N_m = sqrt(21/(4*pi) * (10-abs_m)!/(10+abs_m)!)
        num = 1.0
        for i in range(10 - abs_m + 1, 11):
            num *= i
        den = 1.0
        for i in range(10 - abs_m + 1, 10 + abs_m + 1):
            den *= i
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * num / den)
        
        P = lpmv(abs_m, 10, mu)
        
        if m == 0:
            basis[m_idx] = N_m * P[:, np.newaxis] * np.ones((1, n_phi))
        elif m > 0:
            basis[m_idx] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            basis[m_idx] = np.sqrt(2.0) * N_m * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return basis, mu, w, phi, dphi


def _compute_R(c, basis, mu, w, phi, dphi):
    """Compute R = ||f||_4 / ||f||_2."""
    f = np.tensordot(c, basis, axes=(0, 0))
    weights = np.outer(w, np.full(len(phi), dphi))
    
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    
    if norm2_sq <= 0:
        return np.inf
    
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2


def _compute_grad(c, basis, mu, w, phi, dphi):
    """Compute gradient of R with respect to c."""
    f = np.tensordot(c, basis, axes=(0, 0))
    weights = np.outer(w, np.full(len(phi), dphi))
    
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    
    if norm2_sq <= 0:
        return np.zeros_like(c)
    
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    f3 = f * f2
    int_f3_Y = np.array([np.sum(f3 * basis[j] * weights) for j in range(21)])
    int_f_Y = np.array([np.sum(f * basis[j] * weights) for j in range(21)])
    
    d_norm4_4_dc = 4.0 * int_f3_Y
    d_norm4_dc = 0.25 * norm4_4 ** (-0.75) * d_norm4_4_dc
    
    d_norm2_sq_dc = 2.0 * int_f_Y
    d_norm2_dc = d_norm2_sq_dc / (2.0 * norm2)
    
    grad = d_norm4_dc / norm2 - (norm4 / norm2 ** 2) * d_norm2_dc
    
    return grad


def rudin_problem10() -> np.ndarray:
    """Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonics."""
    start_time = time.time()
    deadline = start_time + 615.0
    
    basis, mu, w, phi, dphi = _build_basis()
    
    # Try multiple initializations
    initializations = [
        np.ones(21, dtype=np.float64),
        np.zeros(21, dtype=np.float64),
        np.random.randn(21, dtype=np.float64),
    ]
    
    best_c = None
    best_R = np.inf
    
    for c0 in initializations:
        if time.time() > deadline - 60:
            break
        
        c = c0.copy()
        
        # If all zeros, use a small random perturbation
        if np.all(c == 0):
            c = np.random.randn(21) * 0.1
        
        R = _compute_R(c, basis, mu, w, phi, dphi)
        
        if not np.isfinite(R):
            continue
        
        local_best_R = R
        local_best_c = c.copy()
        
        lr = 0.05
        momentum = np.zeros(21, dtype=np.float64)
        beta = 0.9
        step = 0
        stagnant = 0
        
        while time.time() < deadline - 30:
            R = _compute_R(c, basis, mu, w, phi, dphi)
            
            if R < local_best_R:
                local_best_R = R
                local_best_c = c.copy()
                stagnant = 0
            else:
                stagnant += 1
            
            if stagnant > 200:
                break
            
            grad = _compute_grad(c, basis, mu, w, phi, dphi)
            
            if not np.all(np.isfinite(grad)):
                lr *= 0.5
                continue
            
            momentum = beta * momentum - lr * grad
            c_new = c + momentum
            
            if not np.all(np.isfinite(c_new)):
                lr *= 0.5
                momentum *= 0.5
                continue
            
            R_new = _compute_R(c_new, basis, mu, w, phi, dphi)
            
            if R_new < R:
                c = c_new
                stagnant = 0
                if step % 50 == 0:
                    lr = min(lr * 1.02, 0.5)
            else:
                lr *= 0.95
                momentum *= 0.5
            
            step += 1
        
        if local_best_R < best_R:
            best_R = local_best_R
            best_c = local_best_c.copy()
    
    if best_c is None:
        return np.ones(21, dtype=np.float64)
    
    # Validate and rescale if needed
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3:
        best_c = best_c * (1e-2 / max_abs) if max_abs > 0 else np.ones(21)
    elif max_abs > 1e5:
        best_c = best_c * (1e4 / max_abs)
    
    return best_c


# EVOLVE-BLOCK-END
