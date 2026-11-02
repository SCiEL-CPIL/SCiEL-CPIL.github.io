import numpy as np
import scipy.optimize as opt
import time

def construct():
    start_time = time.time()
    deadline = 180.0  # Safety margin before 210s limit
    
    # Factorials needed: 0! = 1, 4! = 24, 8! = 40320, 12! = 479001600
    fact = {0: 1.0, 4: 24.0, 8: 40320.0, 12: 479001600.0}
    fact11 = 39916800.0 # 11! for derivative term of H12
    
    def compute_exact_integrals(c0, c4, c8):
        """Compute I0, Ix2, Ideriv exactly using Hermite orthogonality."""
        # Constraint P(0)=0: H_n(0) is 0 for odd n, (-1)^{n/2} (n-1)!! / 2^{n/2} for even n.
        # H_0(0) = 1
        # H_4(0) = 3
        # H_8(0) = 105
        # H_12(0) = 10395
        # c0*1 + c4*3 + c8*105 + c12*10395 = 0
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
        # H_k' = 2k H_{k-1}
        # <H_{k-1} H_{l-1}> = (k-1)! delta_{k,l}
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
            c12 = -(c0 + 3.0*c4 + 105.0*c8) / 10395.0
            c = {0: c0, 4: c4, 8: c8, 12: c12}
            degrees = [0, 4, 8, 12]
            
            dc12_dc = np.array([-1.0/10395.0, -3.0/10395.0, -105.0/10395.0])
            
            dI0_dc = np.zeros(3)
            for i, k in enumerate([0, 4, 8]):
                dI0_dc[i] = 2.0 * c[k] * fact[k] + 2.0 * c[12] * fact[12] * dc12_dc[i]
            
            M = np.zeros((4, 4))
            for idx_k, k in enumerate(degrees):
                for idx_l, l in enumerate(degrees):
                    M_val = 0.0
                    if k + 2 == l:
                        M_val += 0.5 * fact[k+2]
                    if k == l:
                        M_val += k * fact[k]
                    if k >= 2 and k - 2 == l:
                        M_val += 0.5 * k * (k-1) * fact[k-2]
                    M[idx_k, idx_l] = M_val
            
            dIx2_dc = np.zeros(3)
            c_arr = np.array([c[0], c[4], c[8], c[12]])
            for i in range(3):
                s1 = 0.0
                for b in range(4):
                    s1 += c_arr[b] * M[i, b]
                s2 = 0.0
                for a in range(4):
                    s2 += c_arr[a] * M[3, a]
                dIx2_dc[i] = 2.0 * s1 + 2.0 * s2 * dc12_dc[i]
            
            dIderiv_dc = np.zeros(3)
            for i, k in enumerate([0, 4, 8]):
                term1 = 0.0
                if k > 0:
                    term1 = 2.0 * c[k] * 4.0 * k * k * fact[k-1]
                term2 = 2.0 * c[12] * 4.0 * 12 * 12 * fact11 * dc12_dc[i]
                dIderiv_dc[i] = term1 + term2
            
            sqrt_term = np.sqrt(Ix2 * Ideriv)
            if sqrt_term <= 0 or not np.isfinite(sqrt_term):
                return val, np.zeros(3)
            
            grad_corrected = np.zeros(3)
            for i in range(3):
                term1 = (1.0 / (2.0 * np.pi * I0 * 2.0 * sqrt_term)) * (Ideriv * dIx2_dc[i] + Ix2 * dIderiv_dc[i])
                term2 = -sqrt_term / (2.0 * np.pi * I0**2) * dI0_dc[i]
                grad_corrected[i] = term1 + term2
            
            return val, grad_corrected
        except Exception:
            return 1e10, np.zeros(3)
    
    def objective(coeffs):
        val, _ = objective_and_grad(coeffs)
        return val

    # Initial guess from seed (Goncalves et al. 2017)
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    best_score, _ = objective_and_grad(x0)
    best_coeffs = x0.copy()
    
    # Phase 1: Global Search using Differential Evolution
    # The current program uses a grid search in a narrow box. 
    # We will use DE to explore a much wider space.
    # The coefficients can be large in magnitude, but the ratio matters.
    # Let's bound them to a reasonable range where the Gaussian decay and polynomial growth balance.
    # If coefficients are too large, I0 and Ix2 grow, but the ratio might stabilize.
    # However, the constraint P(0)=0 links them.
    # Let's try bounds [-5, 5] for all three.
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    # DE is stochastic, so we set a seed for reproducibility? No, let it explore.
    # But we have a time limit.
    # popsize=20, 3 variables -> 60 initial points.
    # maxiter=100 -> 6000 evaluations.
    # Each eval is ~1ms -> 6s. This is safe.
    
    try:
        res_de = opt.differential_evolution(objective, bounds, 
                                            maxiter=200, 
                                            popsize=25, 
                                            tol=1e-10, 
                                            mutation=(0.5, 1.5), 
                                            recombination=0.9,
                                            seed=42,
                                            polish=False, # We will polish manually
                                            init='sobol')
        
        if res_de.success and np.isfinite(res_de.fun) and res_de.fun < best_score:
            best_score = res_de.fun
            best_coeffs = res_de.x.copy()
            
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 20:
        return np.array(best_coeffs)
    
    # Phase 2: Refine the best found by DE (and the seed)
    # We will refine the best candidate from DE and the original seed to ensure we don't lose the seed if DE failed.
    candidates_to_refine = [best_coeffs, x0]
    
    # Also, let's add a few random points from the DE population if we had access? 
    # We don't have access to the population directly in the result, but we can generate some.
    # Actually, let's just refine the best found so far.
    
    for coeffs in candidates_to_refine:
        if time.time() > deadline - 10:
            break
        
        # Use L-BFGS-B for high precision
        try:
            res = opt.minimize(objective, coeffs, jac=lambda x: objective_and_grad(x)[1],
                              method='L-BFGS-B', 
                              options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
        
        # Follow up with Nelder-Mead for any remaining flatness
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                              options={'xatol': 1e-15, 'fatol': 1e-16, 'maxiter': 1000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Phase 3: Final Polish on the absolute best
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(objective, best_coeffs, jac=lambda x: objective_and_grad(x)[1],
                              method='L-BFGS-B', 
                              options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 200})
            if np.isfinite(res.fun) and res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return np.array(best_coeffs)
