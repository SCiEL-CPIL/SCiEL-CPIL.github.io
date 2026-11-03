# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin

    # Precompute constants for analytic integration
    # Integral of x^(2k) * exp(-x^2) dx from -inf to inf
    # = 0.5 * Gamma(k + 0.5)
    # We need this for k up to 12 (since P is degree 12, P^2 is degree 24, max power x^24 -> k=12)
    # For P', max power is x^11, squared is x^22 -> k=11
    # Let's precompute I_k = 0.5 * Gamma(k + 0.5) for k=0..12
    
    I_k = np.zeros(13)
    I_k[0] = np.sqrt(np.pi) / 2.0
    for k in range(1, 13):
        I_k[k] = I_k[k-1] * (k - 0.5)

    # Probabilists' Hermite polynomials in monomial form
    # He_0(x) = 1
    # He_4(x) = x^4 - 6x^2 + 3
    # He_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # He_12(x) = x^12 - 42x^10 + 1605x^8 - 15120x^6 + 30240x^4 - 12096x^2 + 9240
    
    # Coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    # Index 0: x^0, Index 1: x^2, ..., Index 6: x^12
    
    He0_coeffs = np.zeros(7)
    He0_coeffs[0] = 1.0
    
    He4_coeffs = np.zeros(7)
    He4_coeffs[0] = 3.0
    He4_coeffs[1] = -6.0
    He4_coeffs[2] = 1.0
    
    He8_coeffs = np.zeros(7)
    He8_coeffs[0] = 105.0
    He8_coeffs[1] = -420.0
    He8_coeffs[2] = 210.0
    He8_coeffs[3] = -28.0
    He8_coeffs[4] = 1.0
    
    He12_coeffs = np.zeros(7)
    He12_coeffs[0] = 9240.0
    He12_coeffs[1] = -12096.0
    He12_coeffs[2] = 30240.0
    He12_coeffs[3] = -15120.0
    He12_coeffs[4] = 1605.0
    He12_coeffs[5] = -42.0
    He12_coeffs[6] = 1.0
    
    def get_P_coeffs(c0, c4, c8):
        """
        Returns the monomial coefficients of P(x) = c0*He0 + c4*He4 + c8*He8 + c12*He12
        where c12 is determined by P(0)=0.
        """
        # P(0) = c0*1 + c4*3 + c8*105 + c12*9240 = 0
        c12 = -(c0 * 1.0 + c4 * 3.0 + c8 * 105.0) / 9240.0
        
        P_coeffs = c0 * He0_coeffs + c4 * He4_coeffs + c8 * He8_coeffs + c12 * He12_coeffs
        return P_coeffs

    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        c0, c4, c8 = params
        a = get_P_coeffs(c0, c4, c8)
        
        # P(x) = sum_{j=0}^6 a_j x^(2j)
        
        # 1. Compute I0 = int P(x)^2 exp(-x^2) dx
        # P(x)^2 = sum_{i,j} a_i a_j x^(2(i+j))
        # Integral is sum_{i,j} a_i a_j I_{i+j}
        I0 = 0.0
        for i in range(7):
            if a[i] == 0: continue
            for j in range(i, 7):
                if a[j] == 0: continue
                term = a[i] * a[j] * I_k[i+j]
                if i != j:
                    term *= 2.0
                I0 += term
        
        if I0 < 1e-12:
            return 1e10
            
        # 2. Compute I1 = int x^2 P(x)^2 exp(-x^2) dx
        # x^2 P(x)^2 = sum_{i,j} a_i a_j x^(2(i+j)+2)
        # Integral is sum_{i,j} a_i a_j I_{i+j+1}
        I1 = 0.0
        for i in range(7):
            if a[i] == 0: continue
            for j in range(i, 7):
                if a[j] == 0: continue
                term = a[i] * a[j] * I_k[i+j+1]
                if i != j:
                    term *= 2.0
                I1 += term
                
        # 3. Compute I2 = int (P'(x))^2 exp(-x^2) dx
        # P'(x) = sum_{j=1}^6 2j a_j x^(2j-1)
        # Let b_k be coefficients of P'(x) for x^(2k-1) where k=1..6
        # b_k = 2k a_k
        # (P')^2 = sum_{i,j=1}^6 b_i b_j x^(2(i+j-1))
        # Let m = i+j-1. Power is 2m.
        # Integral is sum_{i,j} b_i b_j I_{i+j-1}
        
        I2 = 0.0
        for i in range(1, 7):
            if a[i] == 0: continue
            bi = 2.0 * i * a[i]
            for j in range(i, 7):
                if a[j] == 0: continue
                bj = 2.0 * j * a[j]
                term = bi * bj * I_k[i+j-1]
                if i != j:
                    term *= 2.0
                I2 += term
                
        return (I1 * I2) / (I0 * I0)

    # Seed values from current program
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    seed = np.array([c0_init, c4_init, c8_init])
    
    best_score = objective(seed)
    best_coeffs = seed.copy()

    def time_ok(margin=10.0):
        return (time.time() - start_time) < (deadline - margin)

    # Optimization
    # Since the objective is analytic and fast, we can afford more iterations or global search
    rng = np.random.default_rng(42)
    
    # Try the seed first
    if time_ok(5.0):
        res = minimize(
            objective,
            seed,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_coeffs = res.x.copy()

    # Try a few random restarts to ensure global minimum
    # The landscape might have local minima, but for 3 variables it's likely smooth
    for _ in range(10):
        if not time_ok(5.0):
            break
        
        # Perturb seed
        pert = seed + rng.normal(0, 0.1, 3)
        
        try:
            res = minimize(
                objective,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Final polish with L-BFGS-B if possible
    if time_ok(5.0):
        try:
            res = minimize(
                objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass

    return best_coeffs

# EVOLVE-BLOCK-END
