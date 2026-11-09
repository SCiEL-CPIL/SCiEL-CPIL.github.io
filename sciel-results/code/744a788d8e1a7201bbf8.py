import numpy as np

def construct():
    # Sample cosmologies
    Om0_vals = [0.25, 0.30, 0.35]
    w0_vals = np.linspace(-1.5, -0.5, 7)
    wa_vals = np.linspace(-0.5, 0.5, 7)
    a_vals = np.logspace(np.log10(0.1), 0, 60)
    
    A = []
    b = []
    
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                Ode0 = 1.0 - Om0
                for a in a_vals:
                    # Compute f_de(a) = exp(-3 * integral from 1 to a of (1+w(a'))/a' da')
                    if abs(a - 1.0) < 1e-10:
                        f_de = 1.0
                    else:
                        n_pts = 80
                        t = np.linspace(1.0, a, n_pts)
                        w_t = w0 + wa * (1.0 - t)
                        integrand = (1.0 + w_t) / t
                        integral = np.trapz(integrand, t)
                        f_de = np.exp(-3.0 * integral)
                    
                    denom = Om0 * a**(-3) + Ode0 * f_de
                    if denom <= 0:
                        continue
                    Om_a = Om0 * a**(-3) / denom
                    Om_a = np.clip(Om_a, 1e-10, 1.0 - 1e-10)
                    
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    
                    if u <= 0.01:
                        continue
                    
                    gamma_target = (5.0 / 7.0) * (Om_a / u)**0.38
                    
                    basis = [1.0, q, q**2, u, wa * a, u * q]
                    A.append(basis)
                    b.append(gamma_target)
    
    A = np.array(A)
    b = np.array(b)
    
    # Least squares fit
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    coeffs = np.clip(coeffs, -10, 10)
    
    return coeffs.tolist()
