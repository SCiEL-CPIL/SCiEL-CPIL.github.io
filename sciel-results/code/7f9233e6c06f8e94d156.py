import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s limit
    
    # Factorials needed: 0! = 1, 4! = 24, 8! = 40320, 12! = 479001600
    fact = {0: 1.0, 4: 24.0, 8: 40320.0, 12: 479001600.0}
    
    def compute_exact_integrals(c0, c4, c8):
        """Compute I0, Ix2, Ideriv exactly using Hermite orthogonality."""
        c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
        c = {0: c0, 4: c4, 8: c8, 12: c12}
        degrees = [0, 4, 8, 12]
        
        # I0 = sum_k c_k^2 * k!
        I0 = 0.0
        for k in degrees:
            I0 += c[k]**2 * fact[k]
        
        # Ix2 = sum over k,l of c_k c_l * <H_k x^2 H_l>
        # x^2 H_k = 0.5*H_{k+2} + k*H_k + 0.5*k*(k-1)*H_{k-2}
        Ix2 = 0.0
        for k in degrees:
            ck = c[k]
            if abs(ck) < 1e-30:
                continue
            # Contribution from 0.5*H_{k+2}
            if (k+2) in fact:
                Ix2 += ck * c[k+2] * 0.5 * fact[k+2]
            # Contribution from k*H_k
            Ix2 += ck * c[k] * k * fact[k]
            # Contribution from 0.5*k*(k-1)*H_{k-2}
            if k >= 2 and (k-2) in fact:
                Ix2 += ck * c[k-2] * 0.5 * k * (k-1) * fact[k-2]
        
        # Ideriv = sum over k,l of c_k c_l * <H_k' H_l'>
        # H_k' = 2k H_{k-1}, so H_k' H_l' = 4kl H_{k-1} H_{l-1}
        # <H_{k-1} H_{l-1}> = (k-1)! delta_{k-1,l-1} = (k-1)! delta_{k,l}
        Ideriv = 0.0
        for k in degrees:
            if k == 0:
                continue  # H_0' = 0
            ck = c[k]
            Ideriv += ck**2 * 4.0 * k * k * fact[k-1]
        
        return I0, Ix2, Ideriv
    
    def objective(coeffs):
        """Compute the uncertainty product A(f)*A(Fhat(f))."""
        c0, c4, c8 = coeffs
        try:
            I0, Ix2, Ideriv = compute_exact_integrals(c0, c4, c8)
            if I0 <= 0 or not np.isfinite(I0):
                return 1e10
            if Ix2 < 0 or not np.isfinite(Ix2):
                return 1e10
            if Ideriv < 0 or not np.isfinite(Ideriv):
                return 1e10
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * (Ideriv / I0)
            
            if A_f_sq <= 0 or A_gf_sq <= 0:
                return 1e10
            
            return np.sqrt(A_f_sq * A_gf_sq)
        except Exception:
            return 1e10
    
    # Initial guess from seed (Goncalves et al. 2017)
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score = objective(x0)
    best_coeffs = x0.copy()
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Local optimization from seed using L-BFGS-B (gradient-based, faster)
    if time.time() < deadline - 30:
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish
    if time.time() < deadline - 20:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 500, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Random restarts with multiple strategies
    num_restarts = 40
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if i % 3 == 0:
            # Perturb around best
            scale = 0.05 * (1.0 + i/20.0)
            x_start = best_coeffs + rng.normal(0, scale, 3)
        elif i % 3 == 1:
            # Perturb around seed
            scale = 0.1
            x_start = x0 + rng.normal(0, scale, 3)
        else:
            # Wider exploration
            scale = 0.3
            x_start = x0 + rng.normal(0, scale, 3)
        
        try:
            # Use L-BFGS-B for speed
            res = opt.minimize(objective, x_start, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 40})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-14, 'fatol': 1e-16, 'maxiter': 2000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Even tighter polish if time remains
    if time.time() < deadline - 2:
        try:
            res = opt.minimize(objective, best_coeffs, method='L-BFGS-B', 
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
