"""Improved growth-index fit with iterative reweighting and polish pass."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """Fit six coefficients to growth index via iterative least-squares with reweighting."""
    t_start = time.time()
    deadline = t_start + 85.0  # 5s safety margin before 90s kill

    # Grid of CPL parameters (Om0, w0, wa) - focus on physically reasonable range
    Om0_vals = np.array([0.20, 0.24, 0.28, 0.30, 0.32, 0.35])
    w0_vals = np.array([-1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7])
    wa_vals = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])

    # Scale factor grid for fitting
    a_vals = np.linspace(0.05, 0.99, 60)

    # Build design matrix and target
    rows = []
    targets = []
    weights = []

    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    a_cubed = a**3
                    de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                    E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                    
                    if E2 <= 0 or E2 < 1e-12:
                        continue
                    Om_a = Om0 * a_cubed / E2
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    
                    if u <= 0.01:
                        continue
                    
                    # Linder-Van de Stark: gamma = (5/7) * (Om(a)/u)^(4/7)
                    gamma_target = (5.0/7.0) * (Om_a / u)**(4.0/7.0)
                    
                    # Design matrix row: [1, q, q^2, u, wa*a, u*q]
                    row = [1.0, q, q*q, u, wa*a, u*q]
                    rows.append(row)
                    targets.append(gamma_target)
                    weights.append(1.0)
                    
                    if time.time() > deadline - 10.0:
                        break
                if time.time() > deadline - 10.0:
                    break
            if time.time() > deadline - 10.0:
                break
        if time.time() > deadline - 10.0:
            break

    if len(rows) < 50:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    A = np.array(rows)
    b = np.array(targets)
    w = np.array(weights)

    # Iterative weighted least squares with reweighting (3 iterations)
    coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    for iteration in range(4):
        # Compute current errors
        pred = A @ coeffs
        errors = np.abs(pred - b)
        
        # Reweight: give more weight to points with larger errors (inverse error weighting)
        # Use a soft reweighting to avoid division by zero
        w = 1.0 / (errors + 0.01)
        
        # Weighted least squares
        Aw = A * w[:, None]
        bw = b * w
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
        except Exception:
            break
        
        # Clip to [-10, 10]
        coeffs = np.clip(coeffs, -10.0, 10.0)
        
        if time.time() > deadline - 5.0:
            break

    # Verify gamma stays in [0.1, 1.2] on a test grid
    def check_gamma_range(c):
        for Om0 in Om0_vals[:4]:
            for w0 in w0_vals[:4]:
                for wa in wa_vals[:4]:
                    for a in np.linspace(0.05, 0.99, 20):
                        a_cubed = a**3
                        de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                        E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                        if E2 <= 0:
                            continue
                        Om_a = Om0 * a_cubed / E2
                        q = 1.0 - Om_a
                        u = 1.0 + w0 + wa * (1.0 - a)
                        gamma = (c[0] + c[1]*q + c[2]*q*q + 
                                 c[3]*u + c[4]*wa*a + c[5]*u*q)
                        if gamma < 0.1 or gamma > 1.2:
                            return False
        return True

    if not check_gamma_range(coeffs):
        # Try to fix with a constrained optimization
        def obj(c):
            return np.sum((A @ c - b)**2)
        
        constraints = []
        # Add soft constraints via penalty in objective
        def penalty(c):
            pen = 0.0
            for Om0 in Om0_vals[:3]:
                for w0 in w0_vals[:3]:
                    for wa in wa_vals[:3]:
                        for a in np.linspace(0.1, 0.99, 10):
                            a_cubed = a**3
                            de_factor = a**(-3.0*(1.0+w0)) * np.exp(-3.0*wa*(1.0-a))
                            E2 = Om0 * a_cubed + (1.0 - Om0) * de_factor
                            if E2 <= 0:
                                continue
                            Om_a = Om0 * a_cubed / E2
                            q = 1.0 - Om_a
                            u = 1.0 + w0 + wa * (1.0 - a)
                            gamma = (c[0] + c[1]*q + c[2]*q*q + 
                                     c[3]*u + c[4]*wa*a + c[5]*u*q)
                            if gamma < 0.1:
                                pen += (0.1 - gamma)**2
                            elif gamma > 1.2:
                                pen += (gamma - 1.2)**2
            return pen
        
        def obj_constrained(c):
            return obj(c) + 100.0 * penalty(c)
        
        bounds = [(-10, 10)] * 6
        try:
            result = minimize(obj_constrained, coeffs, method='L-BFGS-B', bounds=bounds)
            if result.success:
                coeffs = np.clip(result.x, -10.0, 10.0)
        except Exception:
            pass
        
        if not check_gamma_range(coeffs):
            return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Final polish: high-precision local solve with tighter tolerance
    if time.time() < deadline - 2.0:
        Aw = A * w[:, None]
        bw = b * w
        try:
            coeffs_polish, residuals, rank, sv = np.linalg.lstsq(Aw, bw, rcond=1e-12)
            if check_gamma_range(coeffs_polish):
                coeffs = np.clip(coeffs_polish, -10.0, 10.0)
        except Exception:
            pass

    return [float(c) for c in coeffs]
# EVOLVE-BLOCK-END
