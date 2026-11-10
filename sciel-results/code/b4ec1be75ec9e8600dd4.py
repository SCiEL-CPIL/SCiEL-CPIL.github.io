import numpy as np
from scipy.special import lpmv, factorial
from scipy.optimize import minimize
import time

def rudin_problem10() -> np.ndarray:
    K = 10
    N_THETA = 70
    N_PHI = 130
    
    # Deadline guard
    start_time = time.time()
    DEADLINE = 600.0
    SAFETY_MARGIN = 10.0
    
    # 1. Setup Quadrature
    x, w_theta = np.polynomial.legendre.leggauss(N_THETA)
    phi = np.linspace(0, 2 * np.pi, N_PHI, endpoint=False)
    
    N_POINTS = N_THETA * N_PHI
    
    # Weights for the double sum
    # Sum_i,j w_theta[i] * (2*pi/N_PHI) * f(x_i, phi_j)
    w_phi = 2 * np.pi / N_PHI
    # weights shape (N_THETA, N_PHI)
    weights_grid = np.outer(w_theta, np.ones(N_PHI) * w_phi)
    weights = weights_grid.flatten() # (N_POINTS,)

    # 2. Precompute Spherical Harmonics Y_m at quadrature points
    # Y_0 = N_0 P_10^0(cos theta)
    # Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    # Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    # N_m = sqrt( (2K+1)/(4 pi) * (K-m)! / (K+m)! )
    
    Y = np.zeros((N_POINTS, 2 * K + 1))
    
    # Compute N_m
    N_m = np.zeros(K + 1)
    for m in range(K + 1):
        num = (2 * K + 1) / (4 * np.pi) * factorial(K - m) / factorial(K + m)
        N_m[m] = np.sqrt(num)
        
    # Evaluate P_10^m(x)
    P = np.zeros((K + 1, N_THETA))
    for m in range(K + 1):
        P[m] = lpmv(m, K, x)
        
    # Fill Y matrix
    # Index mapping: m = -10..10 -> index 0..20
    # m_idx = m + 10
    
    # Reshape for broadcasting
    x_grid = x[:, np.newaxis] # (N_THETA, 1)
    phi_grid = phi[np.newaxis, :] # (1, N_PHI)
    
    for m in range(K + 1):
        idx_pos = m + K
        idx_neg = -m + K
        
        N_val = N_m[m]
        P_val = P[m, :, np.newaxis] # (N_THETA, 1)
        
        if m == 0:
            Y_m = N_val * P_val * np.ones((1, N_PHI))
            Y[:, idx_pos] = Y_m.flatten()
        else:
            cos_m_phi = np.cos(m * phi_grid)
            sin_m_phi = np.sin(m * phi_grid)
            
            Y_pos = np.sqrt(2.0) * N_val * P_val * cos_m_phi
            Y_neg = np.sqrt(2.0) * N_val * P_val * sin_m_phi
            
            Y[:, idx_pos] = Y_pos.flatten()
            Y[:, idx_neg] = Y_neg.flatten()
            
    # 3. Compute the 4-index integral tensor T
    # T[i, j, k, l] = sum_n w_n Y[i, n] Y[j, n] Y[k, n] Y[l, n]
    # This allows fast evaluation of J(c) = sum T[i,j,k,l] c[i] c[j] c[k] c[l]
    
    # To compute T efficiently, we can use einsum on the weighted Y matrix
    # Y_w = Y * weights[:, np.newaxis]  # (N_POINTS, 21)
    # T = np.einsum('ni,nj,nk,nl->ijkl', Y_w, Y, Y, Y)
    # This might be slow if done naively. Let's try.
    # 9100 * 21^4 is roughly 1.7 billion operations. Might be too slow in Python/NumPy loop.
    # Better:
    # T[i,j,k,l] = sum_n (w_n Y[i,n] Y[j,n]) (Y[k,n] Y[l,n])
    # Let A = Y * sqrt(w)[:, None]  # (N_POINTS, 21)
    # Then T = (A^T A) ... no, that's 2 indices.
    # T is a 4-tensor.
    
    # We can compute T by iterating over pairs of indices?
    # Or just use einsum. NumPy einsum is optimized.
    # Let's check memory. 21^4 * 8 bytes = ~1.5 MB. Fine.
    # Time: 9100 points.
    # einsum 'ni,nj,nk,nl->ijkl' with n=9100, i,j,k,l=21.
    # This is essentially 4 matrix multiplications or similar.
    
    Y_weighted = Y * np.sqrt(weights)[:, np.newaxis]
    # T[i,j,k,l] = sum_n Y_w[i,n] Y_w[j,n] Y_w[k,n] Y_w[l,n]
    # We can do:
    # M1 = Y_w.T @ Y_w  # (21, 21) -- this is 2-index
    # We need 4-index.
    
    # Let's just use einsum.
    T = np.einsum('ni,nj,nk,nl->ijkl', Y_weighted, Y_weighted, Y_weighted, Y_weighted, optimize=True)
    
    # 4. Define Objective and Gradient using T
    def objective_and_grad(c):
        # J(c) = sum T[i,j,k,l] c[i] c[j] c[k] c[l]
        # Using einsum for speed
        val = np.einsum('ijkl,i,j,k,l->', T, c, c, c, c, optimize=True)
        
        # Gradient
        # dJ/dc[i] = 4 * sum T[i,j,k,l] c[j] c[k] c[l]
        # Due to symmetry, T[i,j,k,l] = T[j,i,k,l] etc.
        # The factor 4 comes from 4 terms in the derivative.
        grad = 4.0 * np.einsum('ijkl,j,k,l->i', T, c, c, c, optimize=True)
        
        return val, grad

    def objective_val(c):
        return objective_and_grad(c)[0]
        
    def gradient_val(c):
        return objective_and_grad(c)[1]
        
    def constraint_fun(c):
        return np.dot(c, c) - 1.0
        
    def constraint_jac(c):
        return 2.0 * c
        
    constraints = [{'type': 'eq', 'fun': constraint_fun, 'jac': constraint_jac}]
    
    # 5. Optimization
    # Start with m=0 (index K)
    c_init = np.zeros(2 * K + 1, dtype=np.float64)
    c_init[K] = 1.0
    
    best_c = c_init.copy()
    best_val = objective_and_grad(best_c)[0]
    
    rng = np.random.default_rng(42)
    
    # Try a few random starts
    starts = [c_init]
    for _ in range(20):
        c_rand = rng.normal(size=2 * K + 1)
        c_rand /= np.linalg.norm(c_rand)
        starts.append(c_rand)
        
    for c0 in starts:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
            
        if np.allclose(c0, 0):
            continue
            
        try:
            res = minimize(
                objective_val,
                c0,
                jac=gradient_val,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            
            c_opt = res.x
            # Ensure c_opt is normalized
            norm = np.linalg.norm(c_opt)
            if norm > 0:
                c_opt /= norm
            val_opt = objective_and_grad(c_opt)[0]
            
            if val_opt < best_val:
                best_val = val_opt
                best_c = c_opt
        except Exception:
            continue
            
    # Final check
    if not np.all(np.isfinite(best_c)):
        return np.ones(2 * K + 1, dtype=np.float64)
        
    max_abs_c = np.max(np.abs(best_c))
    if max_abs_c < 1e-3:
        best_c = best_c * (1.0 / max_abs_c)
    elif max_abs_c > 1e5:
        best_c = best_c * (1e5 / max_abs_c)
        
    return best_c
