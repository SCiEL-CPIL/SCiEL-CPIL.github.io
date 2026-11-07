import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, differential_evolution
import sys

# Precomputed constants
H0_0 = 1.0
H4_0 = 3.0
H8_0 = 105.0
H12_0 = 10395.0

# Coefficients for H_n(x) expressed in powers of y=x^2
# H0(y) = 1
# H4(y) = y^2 - 6y + 3
# H8(y) = y^4 - 28y^3 + 280y^2 - 840y + 105
# H12(y) = y^6 - 60y^5 + 1260y^4 - 12600y^3 + 45360y^2 - 30240y + 10395

# We want P(y) = g(sqrt(y))/y
# g(y) = c0*1 + c1*(y^2 - 6y + 3) + c2*(y^4 - 28y^3 + 280y^2 - 840y + 105) + c3*(y^6 - 60y^5 + 1260y^4 - 12600y^3 + 45360y^2 - 30240y + 10395)
# g(0) = c0 + 3c1 + 105c2 + 10395c3 = 0 => c3 = -(c0 + 3c1 + 105c2)/10395

# P(y) = g(y)/y = c3*y^5 + (-60c3)*y^4 + (1260c3+c2)*y^3 + (-12600c3-28c2)*y^2 + (45360c3+280c2-6c1)*y + (-30240c3-840c2-6c1)
# Note: The constant term of g(y) is 0, so division by y is exact.
# Let's verify the constant term of P(y) is the coeff of y^1 in g(y).
# g(y) = ... + A1*y + A0. A0=0.
# P(y) = ... + A2*y + A1.
# So P(y) is a degree 5 polynomial.

def _compute_value(c0, c1, c2):
    c3 = -(c0 + 3.0 * c1 + 105.0 * c2) / 10395.0
    
    # Coefficients of P(y) from highest power to lowest
    # P(y) = a5*y^5 + a4*y^4 + a3*y^3 + a2*y^2 + a1*y + a0
    a5 = c3
    a4 = -60.0 * c3
    a3 = 1260.0 * c3 + c2
    a2 = -12600.0 * c3 - 28.0 * c2
    a1 = 45360.0 * c3 + 280.0 * c2 - 6.0 * c1
    a0 = -30240.0 * c3 - 840.0 * c2 - 6.0 * c1
    
    # Strip leading zeros to handle degree drop
    coeffs = [a5, a4, a3, a2, a1, a0]
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-15:
        coeffs = coeffs[1:]
    
    if len(coeffs) <= 1:
        # Constant or zero polynomial
        return 1e100
    
    # Find roots
    try:
        roots = np.roots(coeffs)
    except Exception:
        return 1e100
        
    # Filter real positive roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            r_real = r.real
            if r_real > 1e-10: # Must be positive
                real_roots.append(r_real)
                
    if not real_roots:
        return 1e100

    # Check for sign change (odd multiplicity)
    max_y = 0.0
    eps_factor = 1e-8
    for r in real_roots:
        eps = eps_factor * max(1.0, r)
        y_left = r - eps
        y_right = r + eps
        
        # Evaluate P(y) using Horner's method
        val_left = coeffs[0]
        val_right = coeffs[0]
        for i in range(1, len(coeffs)):
            val_left = val_left * y_left + coeffs[i]
            val_right = val_right * y_right + coeffs[i]
        
        # Check sign change
        if val_left * val_right < 0:
            if r > max_y:
                max_y = r
                
    if max_y == 0.0:
        return 1e100
        
    return max_y / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill
    
    # Seed
    seed_coeffs = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    seed_value = _compute_value(*seed_coeffs)
    
    best_coeffs = seed_coeffs.copy()
    best_value = seed_value
    
    # Local refinement around seed with tighter bounds
    # The seed is likely close to optimal, so focus search there
    local_bounds = [
        (-1.5, -0.5),  # c0 around -1.13
        (0.0, 0.5),    # c1 around 0.04
        (-0.01, 0.01)  # c2 around 0.0003
    ]
    
    # Penalty-based objective: minimize largest sign-changing root, penalize negativity
    def penalty_objective(x):
        c0, c1, c2 = x
        c3 = -(c0 + 3.0 * c1 + 105.0 * c2) / 10395.0
        
        a5 = c3
        a4 = -60.0 * c3
        a3 = 1260.0 * c3 + c2
        a2 = -12600.0 * c3 - 28.0 * c2
        a1 = 45360.0 * c3 + 280.0 * c2 - 6.0 * c1
        a0 = -30240.0 * c3 - 840.0 * c2 - 6.0 * c1
        
        coeffs = [a5, a4, a3, a2, a1, a0]
        while len(coeffs) > 1 and abs(coeffs[0]) < 1e-15:
            coeffs = coeffs[1:]
        
        if len(coeffs) <= 1:
            return 1e100
        
        # Find roots
        try:
            roots = np.roots(coeffs)
        except Exception:
            return 1e100
        
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-10:
                r_real = r.real
                if r_real > 1e-10:
                    real_roots.append(r_real)
        
        if not real_roots:
            return 1e100
        
        # Find largest sign-changing root
        max_y = 0.0
        eps_factor = 1e-8
        for r in real_roots:
            eps = eps_factor * max(1.0, r)
            y_left = r - eps
            y_right = r + eps
            
            val_left = coeffs[0]
            val_right = coeffs[0]
            for i in range(1, len(coeffs)):
                val_left = val_left * y_left + coeffs[i]
                val_right = val_right * y_right + coeffs[i]
            
            if val_left * val_right < 0:
                if r > max_y:
                    max_y = r
        
        if max_y == 0.0:
            return 1e100
        
        # Penalty for P(y) < 0 on a grid
        penalty = 0.0
        y_grid = np.linspace(1e-6, max_y * 1.5 + 1.0, 200)
        for y in y_grid:
            val = coeffs[0]
            for i in range(1, len(coeffs)):
                val = val * y + coeffs[i]
            if val < 0:
                penalty += -val
                if penalty > 1e10:
                    break
        
        return max_y / (2.0 * np.pi) + penalty * 100.0
    
    # Phase 1: Focused local search around seed using Nelder-Mead with penalty
    if time.time() - start_time < 80.0:
        try:
            res = minimize(
                penalty_objective, seed_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Grid search in tight neighborhood for fine-tuning
    if time.time() - start_time < 140.0:
        # Fine grid around best found
        c0_center, c1_center, c2_center = best_coeffs
        c0_range = 0.05
        c1_range = 0.02
        c2_range = 0.002
        n_steps = 20
        
        c0_vals = np.linspace(c0_center - c0_range, c0_center + c0_range, n_steps)
        c1_vals = np.linspace(c1_center - c1_range, c1_center + c1_range, n_steps)
        c2_vals = np.linspace(c2_center - c2_range, c2_center + c2_range, n_steps)
        
        for c0 in c0_vals:
            if time.time() - start_time > 150.0:
                break
            for c1 in c1_vals:
                if time.time() - start_time > 155.0:
                    break
                for c2 in c2_vals:
                    if time.time() - start_time > 160.0:
                        break
                    val = _compute_value(c0, c1, c2)
                    if val < best_value:
                        best_value = val
                        best_coeffs = np.array([c0, c1, c2])
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < 185.0:
        try:
            res = minimize(
                penalty_objective, best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 30000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
            )
            if res.fun < best_value:
                best_value = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return best_coeffs
