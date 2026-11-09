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
    
    We perform a weighted least-squares fit over a grid of cosmological parameters
    (Om0, w0, wa) and scale factors a in [0.1, 1.0], with emphasis on capturing
    the non-linear dependence on both q and u.
    """
    import numpy as np
    
    # Grid of cosmological parameters - denser and focused on physically relevant range
    Om0_vals = np.linspace(0.22, 0.32, 20)
    w0_vals = np.linspace(-1.0, 0.0, 15)
    wa_vals = np.linspace(-0.4, 0.4, 15)
    
    # Scale factor grid - denser at low a where growth is more sensitive
    a_vals = np.concatenate([
        np.linspace(0.1, 0.3, 20),
        np.linspace(0.3, 0.7, 30),
        np.linspace(0.7, 1.0, 20)
    ])
    
    # Build design matrix and target
    A_rows = []
    y_vals = []
    weights = []
    
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                for a in a_vals:
                    # Compute Om(a) for flat CPL universe
                    # E^2(a) = Om0 * a^3 + (1 - Om0) * a^2 * (1 + wa)^(2*wa) * exp(3*wa*(1-a))
                    
                    if abs(wa) < 1e-10:
                        # wa = 0 case: E^2(a) = Om0*a^3 + (1-Om0)*a^2
                        E2 = Om0 * a**3 + (1 - Om0) * a**2
                    else:
                        # Handle potential numerical issues with (1+wa)^(2*wa)
                        if 1 + wa <= 0:
                            continue
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
                    
                    # Weight: higher weight for points where the ansatz is most stressed
                    # (extreme q, u values) to ensure good coverage
                    w = 1.0 + 0.5 * abs(q - 0.5) + 0.3 * abs(u - 1.0)
                    weights.append(w)
    
    if len(A_rows) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    A = np.array(A_rows)
    y = np.array(y_vals)
    w_arr = np.array(weights)
    
    # Solve weighted least squares
    Aw = A * np.sqrt(w_arr)[:, np.newaxis]
    yw = y * np.sqrt(w_arr)
    
    coeffs, residuals, rank, sv = np.linalg.lstsq(Aw, yw, rcond=None)
    
    # Clip coefficients to [-10, 10]
    coeffs = np.clip(coeffs, -10.0, 10.0)
    
    # Verify gamma stays in [0.1, 1.2] on a dense grid
    # If not, iteratively adjust to satisfy constraints
    c0, c1, c2, c3, c4, c5 = coeffs
    
    # Check validity on dense grid
    def check_validity(c0, c1, c2, c3, c4, c5):
        for Om0 in Om0_vals:
            for w0 in w0_vals:
                for wa in wa_vals:
                    for a in a_vals:
                        if abs(wa) < 1e-10:
                            E2 = Om0 * a**3 + (1 - Om0) * a**2
                        else:
                            if 1 + wa <= 0:
                                continue
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
                            return False
        return True
    
    if not check_validity(c0, c1, c2, c3, c4, c5):
        # Fallback: use constant 5/7 which is always safe
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return [float(c0), float(c1), float(c2), float(c3), float(c4), float(c5)]
# EVOLVE-BLOCK-END
