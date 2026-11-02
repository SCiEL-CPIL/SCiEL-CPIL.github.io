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
    
    def objective_and_grad(coeffs):
        """Compute the uncertainty product and its gradient analytically."""
        c0, c4, c8 = coeffs
        try:
            I0, Ix2, Ideriv = compute_exact_integrals(c0, c4, c8)
            if I0 <= 0 or not np.isfinite(I0):
                return 1e10, np.zeros(3)
            if Ix2 < 0 or not np.isfinite(Ix2):
                return 1e10, np.zeros(3)
            if Ideriv < 0 or not np.isfinite(Ideriv):
                return 1e10, np.zeros(3)
            
            A_f_sq = Ix2 / I0
            A_gf_sq = (1.0 / (4.0 * np.pi**2)) * (Ideriv / I0)
            
            if A_f_sq <= 0 or A_gf_sq <= 0:
                return 1e10, np.zeros(3)
            
            val = np.sqrt(A_f_sq * A_gf_sq)
            
            # Compute gradient analytically
            # f = sqrt(A_f_sq * A_gf_sq) = sqrt((Ix2/I0) * (Ideriv/(4*pi^2*I0)))
            # f = sqrt(Ix2 * Ideriv) / (2*pi*I0)
            # df/dc = (1/(2*pi*I0)) * d/dc[sqrt(Ix2*Ideriv)]
            #       = (1/(2*pi*I0)) * (1/(2*sqrt(Ix2*Ideriv))) * (Ideriv*dIx2/dc + Ix2*dIderiv/dc)
            #       = (1/(4*pi*sqrt(Ix2*Ideriv)*I0)) * (Ideriv*dIx2/dc + Ix2*dIderiv/dc)
            
            # We need dI0/dc, dIx2/dc, dIderiv/dc
            c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
            c = {0: c0, 4: c4, 8: c8, 12: c12}
            degrees = [0, 4, 8, 12]
            
            # dI0/dc_i = 2*c_i*fact[i] for i in {0,4,8}, and for c12: 2*c12*fact[12]*dc12/dc_i
            # dc12/dc0 = -1/10395, dc12/dc4 = -3/10395, dc12/dc8 = -105/10395
            dc12_dc = np.array([-1.0/10395.0, -3.0/10395.0, -105.0/10395.0])
            
            dI0_dc = np.zeros(3)
            for i, k in enumerate([0, 4, 8]):
                dI0_dc[i] = 2.0 * c[k] * fact[k] + 2.0 * c[12] * fact[12] * dc12_dc[i]
            
            # dIx2/dc: Ix2 = sum_{k,l} c_k c_l M_{kl} where M_{kl} = <H_k x^2 H_l>
            # dIx2/dc_i = sum_l c_l M_{il} + sum_k c_k M_{ki}
            # Since M is symmetric, dIx2/dc_i = 2*sum_l c_l M_{il}
            
            # Build M matrix
            M = np.zeros((4, 4))
            for idx_k, k in enumerate(degrees):
                for idx_l, l in enumerate(degrees):
                    # <H_k x^2 H_l>
                    # x^2 H_k = 0.5*H_{k+2} + k*H_k + 0.5*k*(k-1)*H_{k-2}
                    # <H_k x^2 H_l> = <(0.5*H_{k+2} + k*H_k + 0.5*k*(k-1)*H_{k-2}) H_l>
                    # = 0.5*fact[k+2]*delta_{k+2,l} + k*fact[k]*delta_{k,l} + 0.5*k*(k-1)*fact[k-2]*delta_{k-2,l}
                    M_val = 0.0
                    if k + 2 == l:
                        M_val += 0.5 * fact[k+2]
                    if k == l:
                        M_val += k * fact[k]
                    if k >= 2 and k - 2 == l:
                        M_val += 0.5 * k * (k-1) * fact[k-2]
                    M[idx_k, idx_l] = M_val
            
            # Now compute dIx2/dc_i for i in {0,4,8}
            # Ix2 = sum_{a,b} c_a c_b M_{ab} where a,b in {0,4,8,12}
            # c_12 depends on c_0, c_4, c_8
            # dIx2/dc_i = sum_b c_b M_{i,b} + sum_a c_a M_{a,i} + sum_b c_b M_{12,b} * dc12/dc_i + sum_a c_a M_{a,12} * dc12/dc_i
            # Since M is symmetric: = 2*sum_b c_b M_{i,b} + 2*sum_a c_a M_{12,a} * dc12/dc_i
            
            dIx2_dc = np.zeros(3)
            c_arr = np.array([c[0], c[4], c[8], c[12]])
            for i in range(3):
                # 2*sum_b c_b M_{i,b}
                s1 = 0.0
                for b in range(4):
                    s1 += c_arr[b] * M[i, b]
                # 2*sum_a c_a M_{12,a} * dc12/dc_i
                s2 = 0.0
                for a in range(4):
                    s2 += c_arr[a] * M[3, a]  # M_{12,a}
                dIx2_dc[i] = 2.0 * s1 + 2.0 * s2 * dc12_dc[i]
            
            # dIderiv/dc: Ideriv = sum_k c_k^2 * 4*k^2*fact[k-1] for k in {4,8,12}
            # dIderiv/dc_i = 2*c_i*4*i^2*fact[i-1] + 2*c_12*4*12^2*fact[11]*dc12/dc_i
            dIderiv_dc = np.zeros(3)
            for i, k in enumerate([0, 4, 8]):
                term1 = 0.0
                if k > 0:
                    term1 = 2.0 * c[k] * 4.0 * k * k * fact[k-1]
                term2 = 2.0 * c[12] * 4.0 * 12 * 12 * fact[11] * dc12_dc[i]
                dIderiv_dc[i] = term1 + term2
            
            # Combine
            sqrt_term = np.sqrt(Ix2 * Ideriv)
            if sqrt_term <= 0 or not np.isfinite(sqrt_term):
                return val, np.zeros(3)
            
            grad = np.zeros(3)
            for i in range(3):
                grad[i] = (1.0 / (4.0 * np.pi * sqrt_term * I0)) * (Ideriv * dIx2_dc[i] + Ix2 * dIderiv_dc[i])
            
            # Also account for d/dI0 of the 1/I0 factor
            # f = sqrt(Ix2*Ideriv)/(2*pi*I0)
            # df/dc = (1/(2*pi)) * [d(sqrt(Ix2*Ideriv))/dc * 1/I0 + sqrt(Ix2*Ideriv) * d(1/I0)/dc]
            # d(1/I0)/dc = -dI0/dc / I0^2
            # So: df/dc = (1/(2*pi*I0)) * d(sqrt(Ix2*Ideriv))/dc - sqrt(Ix2*Ideriv)/(2*pi*I0^2) * dI0/dc
            #       = (1/(2*pi*I0)) * [1/(2*sqrt(Ix2*Ideriv)) * (Ideriv*dIx2/dc + Ix2*dIderiv/dc)] - sqrt(Ix2*Ideriv)/(2*pi*I0^2) * dI0/dc
            
            grad_corrected = np.zeros(3)
            for i in range(3):
                term1 = (1.0 / (2.0 * np.pi * I0 * 2.0 * sqrt_term)) * (Ideriv * dIx2_dc[i] + Ix2 * dIderiv_dc[i])
                term2 = -sqrt_term / (2.0 * np.pi * I0**2) * dI0_dc[i]
                grad_corrected[i] = term1 + term2
            
            return val, grad_corrected
        except Exception:
            return 1e10, np.zeros(3)
    
    def objective(coeffs):
        """Compute the uncertainty product A(f)*A(Fhat(f))."""
        val, _ = objective_and_grad(coeffs)
        return val
    
    # Initial guess from seed (Goncalves et al. 2017)
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    
    best_score, _ = objective_and_grad(x0)
    best_coeffs = x0.copy()
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Local optimization from seed using L-BFGS-B with analytic gradient
    if time.time() < deadline - 30:
        try:
            res = opt.minimize(objective, x0, jac=lambda x: objective_and_grad(x)[1],
                              method='L-BFGS-B', 
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
            # Use L-BFGS-B with analytic gradient for speed
            res = opt.minimize(objective, x_start, jac=lambda x: objective_and_grad(x)[1],
                              method='L-BFGS-B', 
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
            res = opt.minimize(objective, best_coeffs, jac=lambda x: objective_and_grad(x)[1],
                              method='L-BFGS-B', 
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
