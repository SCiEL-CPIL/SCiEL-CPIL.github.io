"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Derive six coefficients for the growth-index ansatz:
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    
    where q = 1 - Om(a), u = 1 + w0 + wa*(1-a)
    
    We fit to the Linder approximation:
    gamma_exact = (5/7) * (Om(a) / (1 + w0 + wa*(1-a)))^0.38
                 = (5/7) * ((1-q) / u)^0.38
    
    We perform a least-squares fit over a grid of cosmological parameters
    (Om0, w0, wa) and scale factors a in [0.1, 1.0].
    """
    import numpy as np
    
    # Grid of cosmological parameters
    Om0_vals = np.linspace(0.2, 0.35, 15)
    w0_vals = np.linspace(-1.0, 0.0, 10)
    wa_vals = np.linspace(-0.5, 0.5, 10)
    
    # Scale factor grid
    a_vals = np.linspace(0.1, 1.0, 50)
    
    # Build design matrix and target
    A_rows = []
    y_vals = []
    
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    # Compute Om(a) for flat CPL universe
                    # Om_m(a) = Om0 * a^3 / E(a)^2 where E^2(a) = Om0*a^3 + (1-Om0)*a^2*(1+w0+wa*(1-a))^(-2*wa)
                    # Actually for CPL: E^2(a) = Om0*a^3 + (1-Om0)*exp(3*wa*(1-a)) * a^2 * (1+wa)^(2*wa)
                    # More standard: E^2(a) = Om0*a^3 + (1-Om0)*a^2*(1+wa)^(2*wa)*exp(3*wa*(1-a))
                    # Let's use the standard form:
                    # E^2(a) = Om0 * a^3 + (1 - Om0) * a^2 * (1 + wa)^(2*wa) * exp(3*wa*(1-a))
                    
                    if abs(wa) < 1e-10:
                        # wa = 0 case: E^2(a) = Om0*a^3 + (1-Om0)*a^2
                        E2 = Om0 * a**3 + (1 - Om0) * a**2
                    else:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))
                    
                    if E2 <= 0:
                        continue
                    
                    Om_a = Om0 * a**3 / E2
                    if Om_a <= 0 or Om_a >= 1:
                        continue
                    
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a)
                    
                    if u <= 0:
                        continue
                    
                    # Target gamma from Linder formula
                    gamma_target = (5.0 / 7.0) * (Om_a / u)**0.38
                    # = (5/7) * ((1-q)/u)^0.38
                    
                    if gamma_target < 0.1 or gamma_target > 1.2:
                        continue
                    
                    # Design vector: [1, q, q^2, u, wa*a, u*q]
                    row = np.array([1.0, q, q**2, u, wa * a, u * q])
                    A_rows.append(row)
                    y_vals.append(gamma_target)
    
    if len(A_rows) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(A_rows)
    y = np.array(y_vals)
    
    # Solve least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    
    # Clip coefficients to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # Verify gamma stays in [0.1, 1.2] on a dense grid
    # If not, fall back to constant
    c0, c1, c2, c3, c4, c5 = coeffs
    
    valid = True
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    if abs(wa) < 1e-10:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2
                    else:
                        E2 = Om0 * a**3 + (1 - Om0) * a**2 * (1 + wa)**(2*wa) * np.exp(3*wa*(1-a))
                    
                    if E2 <= 0:
                        continue
                    
                    Om_a = Om0 * a**3 / E2
                    if Om_a <= 0 or Om_a >= 1:
                        continue
                    
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a)
                    
                    if u <= 0:
                        continue
                    
                    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                    
                    if gamma < 0.1 or gamma > 1.2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use constant 5/7
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return [float(c0), float(c1), float(c2), float(c3), float(c4), float(c5)]
# EVOLVE-BLOCK-END
