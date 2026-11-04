import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import eval_hermite
from scipy.integrate import quad
import time
import math

def construct():
    start_time = time.time()
    deadline = 190.0  # Safety margin

    # Hermite polynomial degrees
    degrees = [0, 4, 8, 12]
    
    # Precompute weights for L2 norm: W_i = sqrt(pi) * 2^i * i!
    W = np.array([math.sqrt(math.pi) * (2**d) * math.factorial(d) for d in degrees])
    
    # Values at 0
    H_vals_0 = np.array([eval_hermite(d, 0) for d in degrees])
    
    # c12 is fixed by g(0) = 0
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # c12 = -(c0*H0(0) + c4*H4(0) + c8*H8(0)) / H12(0)
    def get_c12(c0, c4, c8):
        return -(c0 * H_vals_0[0] + c4 * H_vals_0[1] + c8 * H_vals_0[2]) / H_vals_0[3]

    # Function to evaluate g(x)
    def g(x, c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        c = np.array([c0, c4, c8, c12])
        # g(x) = sum c_i H_i(x) * exp(-x^2/2)
        h_vals = np.array([eval_hermite(d, x) for d in degrees])
        poly = np.dot(c, h_vals)
        return poly * np.exp(-x**2 / 2.0)

    # Function to find roots of g(x)
    # We search on a finite interval [-L, L]. The function decays fast.
    L = 15.0
    grid = np.linspace(-L, L, 1000)
    
    def find_roots(c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        c = np.array([c0, c4, c8, c12])
        
        # Evaluate g on grid
        h_vals_grid = np.array([eval_hermite(d, grid) for d in degrees])
        poly_grid = np.dot(c, h_vals_grid)
        g_grid = poly_grid * np.exp(-grid**2 / 2.0)
        
        roots = []
        # Check for sign changes
        for i in range(len(grid) - 1):
            if g_grid[i] == 0:
                roots.append(grid[i])
            elif g_grid[i] * g_grid[i+1] < 0:
                try:
                    r = brentq(lambda x: g(x, c0, c4, c8), grid[i], grid[i+1], xtol=1e-12)
                    roots.append(r)
                except Exception:
                    pass
        
        # Sort and unique roots
        if not roots:
            return []
        roots = np.sort(roots)
        # Remove duplicates within tolerance
        unique_roots = [roots[0]]
        for r in roots[1:]:
            if r - unique_roots[-1] > 1e-9:
                unique_roots.append(r)
        return unique_roots

    # Precompute integrals of H_j(x) * exp(-x^2/2) from -inf to x
    # Actually, we need integrals over intervals [a, b].
    # I_j(a, b) = integral_a^b H_j(x) exp(-x^2/2) dx
    # We can use quad for each interval.
    
    def compute_L1_and_grad(c0, c4, c8):
        c12 = get_c12(c0, c4, c8)
        c = np.array([c0, c4, c8, c12])
        
        # L2 norm squared
        l2_sq = np.dot(c**2, W)
        if l2_sq < 1e-15:
            return 1e10, np.zeros(3)
        l2 = np.sqrt(l2_sq)
        
        # Find roots
        roots = find_roots(c0, c4, c8)
        
        # Define intervals
        # Intervals are [-L, r1], [r1, r2], ..., [rk, L]
        # We only care about intervals where g is not zero.
        # The sign is determined by g(midpoint).
        
        L = 15.0
        boundaries = np.concatenate(([-L], roots, [L]))
        
        l1 = 0.0
        grad_l1 = np.zeros(3) # Gradient w.r.t c0, c4, c8
        
        for i in range(len(boundaries) - 1):
            a = boundaries[i]
            b = boundaries[i+1]
            if b - a < 1e-10:
                continue
            
            mid = (a + b) / 2.0
            g_mid = g(mid, c0, c4, c8)
            
            if np.abs(g_mid) < 1e-12:
                # This interval might be a root or very small contribution
                # Skip or handle carefully. If it's a root, the integral is 0? No, the sign flips.
                # If g is 0 at mid, it's likely a root was missed or it's a flat region.
                # For safety, if it's very small, skip.
                continue
                
            sign = 1.0 if g_mid > 0 else -1.0
            
            # Compute integral of g(x) over [a, b]
            # integral_a^b sum c_j H_j(x) exp(-x^2/2) dx
            # = sum c_j * integral_a^b H_j(x) exp(-x^2/2) dx
            
            interval_integrals = []
            for j, d in enumerate(degrees):
                # Integrate H_d(x) exp(-x^2/2) from a to b
                val, err = quad(lambda x: eval_hermite(d, x) * np.exp(-x**2 / 2.0), a, b, limit=50)
                interval_integrals.append(val)
            
            interval_integrals = np.array(interval_integrals)
            
            # Contribution to L1
            val_l1 = sign * np.dot(c, interval_integrals)
            l1 += val_l1
            
            # Contribution to Gradient of L1 w.r.t c0, c4, c8
            # d/dc_k (sign * sum c_j I_j) = sign * I_k for k in {0, 4, 8}
            # Note: c12 is a function of c0, c4, c8.
            # L1 = sign * (c0 I_0 + c4 I_4 + c8 I_8 + c12(c) I_12)
            # dL1/dc0 = sign * (I_0 + dc12/dc0 * I_12)
            # dc12/dc0 = -H0(0)/H12(0)
            # dc12/dc4 = -H4(0)/H12(0)
            # dc12/dc8 = -H8(0)/H12(0)
            
            dc12_dc0 = -H_vals_0[0] / H_vals_0[3]
            dc12_dc4 = -H_vals_0[1] / H_vals_0[3]
            dc12_dc8 = -H_vals_0[2] / H_vals_0[3]
            
            grad_l1[0] += sign * (interval_integrals[0] + dc12_dc0 * interval_integrals[3])
            grad_l1[1] += sign * (interval_integrals[1] + dc12_dc4 * interval_integrals[3])
            grad_l1[2] += sign * (interval_integrals[2] + dc12_dc8 * interval_integrals[3])
            
        return l1, grad_l1

    def objective_and_grad(c):
        c0, c4, c8 = c
        l1, grad_l1 = compute_L1_and_grad(c0, c4, c8)
        
        c12 = get_c12(c0, c4, c8)
        c_full = np.array([c0, c4, c8, c12])
        l2_sq = np.dot(c_full**2, W)
        
        if l2_sq < 1e-15:
            return 1e10, np.zeros(3)
        
        l2 = np.sqrt(l2_sq)
        
        # Objective J = l1^2 / l2^2
        J = (l1 * l1) / l2_sq
        
        # Gradient of J
        # J = l1^2 * l2^-2
        # dJ/dc = 2*l1 * dl1/dc * l2^-2 + l1^2 * (-2) * l2^-3 * dl2/dc
        # dl2/dc = 1/(2*l2) * d(l2^2)/dc
        # d(l2^2)/dc_k = 2 * c_k * W_k (for k=0,4,8) + 2 * c12 * W12 * dc12/dc_k
        # Wait, l2^2 = c0^2 W0 + c4^2 W4 + c8^2 W8 + c12^2 W12
        # d(l2^2)/dc0 = 2 c0 W0 + 2 c12 W12 dc12/dc0
        # ...
        
        dc12_dc0 = -H_vals_0[0] / H_vals_0[3]
        dc12_dc4 = -H_vals_0[1] / H_vals_0[3]
        dc12_dc8 = -H_vals_0[2] / H_vals_0[3]
        
        dl2_sq_dc0 = 2 * c0 * W[0] + 2 * c12 * W[3] * dc12_dc0
        dl2_sq_dc4 = 2 * c4 * W[1] + 2 * c12 * W[3] * dc12_dc4
        dl2_sq_dc8 = 2 * c8 * W[2] + 2 * c12 * W[3] * dc12_dc8
        
        dl2_dc0 = dl2_sq_dc0 / (2 * l2)
        dl2_dc4 = dl2_sq_dc4 / (2 * l2)
        dl2_dc8 = dl2_sq_dc8 / (2 * l2)
        
        grad_J = np.zeros(3)
        grad_J[0] = 2 * l1 * grad_l1[0] / l2_sq - 2 * l1**2 * dl2_dc0 / l2**3
        grad_J[1] = 2 * l1 * grad_l1[1] / l2_sq - 2 * l1**2 * dl2_dc4 / l2**3
        grad_J[2] = 2 * l1 * grad_l1[2] / l2_sq - 2 * l1**2 * dl2_dc8 / l2**3
        
        return J, grad_J

    # Seed values
    c0_init = -113.0 / 100.0
    c4_init = 1.0 / 25.0
    c8_init = 1.0 / 3240.0
    best_c = np.array([c0_init, c4_init, c8_init])
    best_score, _ = objective_and_grad(best_c)

    # Multi-start L-BFGS-B
    n_starts = 20
    np.random.seed(42)
    
    for i in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        if i == 0:
            x0 = best_c
        else:
            # Random start on sphere
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            x0 = np.array([
                np.cos(theta) * np.cos(phi),
                np.cos(theta) * np.sin(phi),
                np.sin(theta)
            ])
        
        try:
            res = minimize(
                objective_and_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_c = res.x
        except Exception:
            pass

    # Final polish with even tighter tolerance
    try:
        res = minimize(
            objective_and_grad,
            best_c,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_c = res.x
    except Exception:
        pass

    # Scale to unit L2 norm
    c0_f, c4_f, c8_f = best_c
    c12_f = get_c12(c0_f, c4_f, c8_f)
    norm_sq_f = c0_f**2 * W[0] + c4_f**2 * W[1] + c8_f**2 * W[2] + c12_f**2 * W[3]
    if norm_sq_f > 1e-15:
        scale = 1.0 / np.sqrt(norm_sq_f)
        best_c = best_c * scale

    return np.array(best_c)
