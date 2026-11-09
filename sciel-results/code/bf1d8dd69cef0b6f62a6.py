"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _fit_coefficients():
    """Fit the six growth-index coefficients via least squares to the Linder-Waugh formula."""
    t_start = time.time()
    
    # Grid of CPL parameters and scale factors
    w0_vals = np.linspace(-1.2, -0.8, 25)
    wa_vals = np.linspace(-0.5, 0.5, 25)
    a_vals = np.linspace(0.3, 1.0, 50)
    
    Omega_m0 = 0.3
    Omega_de0 = 1.0 - Omega_m0
    
    # Build the design matrix and target vector
    rows = []
    targets = []
    
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                # Compute u = 1 + w0 + wa*(1-a)
                u = 1.0 + w0 + wa * (1.0 - a)
                if u <= 0.05:  # Avoid singularities
                    continue
                
                # Compute Omega_m(a) for flat CPL universe
                denom = Omega_m0 * a**3 + Omega_de0 / u
                Om_a = Omega_m0 * a**3 / denom
                
                if Om_a <= 0 or Om_a >= 1:
                    continue
                
                q = 1.0 - Om_a
                target_gamma = (5.0/7.0) * (Om_a / u) ** 0.38
                
                # Design matrix row: [1, q, q^2, u, wa*a, u*q]
                row = np.array([1.0, q, q**2, u, wa * a, u * q])
                rows.append(row)
                targets.append(target_gamma)
                
                if time.time() - t_start > 45:  # Safety margin
                    break
            if time.time() - t_start > 45:
                break
        if time.time() - t_start > 45:
            break
    
    if len(rows) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(rows)
    b = np.array(targets)
    
    # Solve least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    
    # Clip to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # Verify gamma stays in [0.1, 1.2] on the training grid
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                u = 1.0 + w0 + wa * (1.0 - a)
                if u <= 0.05:
                    continue
                denom = Omega_m0 * a**3 + Omega_de0 / u
                Om_a = Omega_m0 * a**3 / denom
                if Om_a <= 0 or Om_a >= 1:
                    continue
                q = 1.0 - Om_a
                gamma = coeffs[0] + coeffs[1]*q + coeffs[2]*q**2 + coeffs[3]*u + coeffs[4]*wa*a + coeffs[5]*u*q
                if gamma < 0.1 or gamma > 1.2:
                    return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return coeffs.tolist()

def construct():
    """Construct the six growth-index coefficients."""
    t_start = time.time()
    
    try:
        coeffs = _fit_coefficients()
    except Exception:
        coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Ensure finite and in range
    for i in range(6):
        if not np.isfinite(coeffs[i]) or coeffs[i] < -10 or coeffs[i] > 10:
            coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            break
    
    return coeffs
# EVOLVE-BLOCK-END
