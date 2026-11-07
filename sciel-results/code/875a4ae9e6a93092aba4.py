"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing


def construct():
    """Find optimal coefficients c0, c1, c2 to minimize the uncertainty constant."""
    
    def compute_Q_coefficients(c0, c1, c2):
        """
        Returns coefficients of Q(u) = g(x)/x^2 where u=x^2.
        Q(u) = a5*u^5 + a4*u^4 + a3*u^3 + a2*u^2 + a1*u + a0
        """
        c3 = -(c0 + 12.0*c1 + 1680.0*c2) / 4435200.0
        
        a5 = c3 * 4096.0
        a4 = c3 * (-98304.0)
        a3 = c2*256.0 + c3*1032192.0
        a2 = c2*(-3072.0) + c3*(-5308416.0)
        a1 = c1*16.0 + c2*11520.0 + c3*12096000.0
        a0 = c1*(-48.0) + c2*(-13440.0) + c3*(-12902400.0)
        
        return np.array([a5, a4, a3, a2, a1, a0])

    def evaluate_value(c0, c1, c2):
        """Compute the actual value = r^2/(2*pi) for given coefficients."""
        Q_coeffs = compute_Q_coefficients(c0, c1, c2)
        
        # Check for degenerate case
        if np.all(np.abs(Q_coeffs) < 1e-14):
            return 1e100
            
        # Find all roots
        roots = np.roots(Q_coeffs)
        
        # Filter for real roots with small imaginary part
        real_roots = []
        for root in roots:
            if abs(root.imag) < 1e-10:
                r = root.real
                if r > 1e-10:
                    real_roots.append(r)
        
        if len(real_roots) == 0:
            return 1e100
            
        real_roots.sort(reverse=True)
        
        # Find largest root with sign change
        for r in real_roots:
            eps = 1e-6 * max(1.0, abs(r))
            Q_left = np.polyval(Q_coeffs, r - eps)
            Q_right = np.polyval(Q_coeffs, r + eps)
            
            if Q_left * Q_right < 0:
                return r*r / (2.0 * np.pi)
        
        return 1e100

    def objective(coeffs):
        """Minimize the value (lower is better)."""
        c0, c1, c2 = coeffs
        try:
            val = evaluate_value(c0, c1, c2)
            return val
        except Exception:
            return 1e100

    # Seed
    x0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_value = evaluate_value(*x0)
    
    # Strategy 1: Differential Evolution with wider bounds
    bounds = [
        (-10.0, 10.0),    # c0
        (-2.0, 2.0),      # c1
        (-0.05, 0.05)     # c2
    ]
    
    best_coeffs = x0.copy()
    best_val = seed_value
    
    # Run multiple optimization strategies
    strategies = [
        ('DE', lambda: differential_evolution(
            objective, bounds, seed=42, maxiter=500, popsize=30,
            tol=1e-10, polish=True, init='sobol'
        )),
        ('DualAnneal', lambda: dual_annealing(
            objective, bounds=bounds, seed=42, maxiter=1000,
            restarts=10
        )),
        ('NelderMead_from_seed', lambda: minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )),
        ('NelderMead_from_seed_perturbed', lambda: minimize(
            objective, x0 + np.array([0.01, 0.001, 0.0001]), method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )),
    ]
    
    for name, opt_func in strategies:
        try:
            result = opt_func()
            if hasattr(result, 'x') and hasattr(result, 'fun'):
                if result.fun < best_val:
                    best_val = result.fun
                    best_coeffs = result.x.copy()
                    # Polish with Nelder-Mead
                    polish_result = minimize(
                        objective, best_coeffs, method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
                    )
                    if polish_result.fun < best_val:
                        best_val = polish_result.fun
                        best_coeffs = polish_result.x.copy()
        except Exception:
            pass
    
    # Final verification
    final_val = evaluate_value(*best_coeffs)
    if final_val < seed_value:
        return best_coeffs
    else:
        return x0

# EVOLVE-BLOCK-END
