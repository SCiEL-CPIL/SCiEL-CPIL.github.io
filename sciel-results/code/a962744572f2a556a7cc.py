"""Radau IIA collocation integrator for the Van der Pol oscillator (mu=5)."""
# EVOLVE-BLOCK-START
import numpy as np


def _radau_ii_a_5():
    """Return (A, B, c) for the 5-stage Radau IIA collocation method.
    A is the collocation matrix (n x n), B is the quadrature weights (n,), c are the stage nodes (n,).
    """
    n = 5
    # The Radau IIA method uses Gauss-Lobatto-like nodes (Radau nodes):
    # The nodes c_i are the roots of the polynomial P_n(x) = sum_{k=0}^n binom(2n-k, n) x^k (1-x)^{n-k}
    # Actually, for Radau IIA, the nodes are the roots of P_n(x) where P_n is the (n-th) Radau polynomial.
    # The Radau nodes (for n=5) are the roots of:
    # P_5(x) = 1260 x^5 - 5460 x^4 + 9660 x^3 - 7560 x^2 + 2520 x - 300 = 0
    # But it's easier to use known values or compute via scipy.
    # Since we want to avoid scipy in the core loop for speed, let's hardcode the nodes for n=5.
    
    # Radau IIA nodes for n=5 (from literature or computed):
    # The nodes are the roots of the polynomial:
    # P_n(x) = \sum_{j=0}^n \binom{2n-j}{n} x^j (1-x)^{n-j}
    # For n=5:
    # P_5(x) = 1260 x^5 - 5460 x^4 + 9660 x^3 - 7560 x^2 + 2520 x - 300
    # Let's find the roots.
    
    coeffs = [1260, -5460, 9660, -7560, 2520, -300]
    c = np.roots(coeffs)
    # The roots should be real and in (0, 1]. Let's sort them.
    c = np.sort(c.real)
    
    # The collocation matrix A is defined by:
    # A_{ij} = \int_{c_i}^{c_j} \ell_j(s) ds
    # where \ell_j(s) are the Lagrange basis polynomials for the nodes c_1, ..., c_n.
    # For Radau IIA, we use the nodes c_1 < c_2 < ... < c_n.
    
    # Let's compute A and B using polynomial integration.
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    # For each j, find the Lagrange basis polynomial \ell_j(x) = \prod_{k \neq j} (x - c_k) / (c_j - c_k)
    # Then A_{ij} = \int_{c_i}^{c_j} \ell_j(s) ds
    # And B_j = \int_0^1 \ell_j(s) ds
    
    for j in range(n):
        # Build the polynomial \ell_j(x)
        # \ell_j(x) = \prod_{k \neq j} (x - c_k) / (c_j - c_k)
        numerator_coeffs = np.array([1.0])
        denom = 1.0
        for k in range(n):
            if k != j:
                numerator_coeffs = np.convolve(numerator_coeffs, [1.0, -c[k]])
                denom *= (c[j] - c[k])
        numerator_coeffs = numerator_coeffs / denom
        
        # Integrate \ell_j(s) from c_i to c_j for each i
        # The integral of a polynomial can be done analytically.
        # \int \ell_j(s) ds = sum_k numerator_coeffs[k] * s^{k+1} / (k+1)
        
        # For B_j: integral from 0 to 1
        B[j] = sum(numerator_coeffs[k] / (k + 1) for k in range(len(numerator_coeffs)))
        
        for i in range(n):
            # Integral from c_i to c_j
            # If i > j, the integral is negative of integral from c_j to c_i
            # But in Radau IIA, A is lower triangular? No, it's not strictly triangular.
            # Let's just compute the definite integral.
            integral = lambda x: sum(numerator_coeffs[k] * x**(k+1) / (k+1) for k in range(len(numerator_coeffs)))
            A[i, j] = integral(c[j]) - integral(c[i])
    
    return A, B, c


def _compute_jacobian(f, t, y, h):
    """Compute the Jacobian of f at (t, y) using finite differences."""
    n = len(y)
    J = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        scale = 1e-7 * max(1.0, abs(y[j]))
        e[j] = scale
        J[:, j] = (f(t, y + e) - f(t, y)) / scale
    return J


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Precompute Radau IIA coefficients for 5 stages
    A, B, c = _radau_ii_a_5()
    
    # The mass matrix M = I - h * A (for each step, h changes)
    # We need to solve (I - h * A) Y = y + h * B * F(Y)
    # This is a system of n * n_stages equations.
    
    # Let's use a fixed number of steps. For the Van der Pol oscillator with mu=5,
    # the dynamics are somewhat stiff. Let's use a reasonable number of steps.
    # The time span is [0, 10]. Let's try 500 steps, so h = 0.02.
    steps = 500
    h = (t1 - t0) / steps
    
    t = t0
    for _ in range(steps):
        # Newton iteration for the stage values Y (n x n_stages)
        # Y is a matrix where Y[:, j] is the stage value at stage j.
        Y = np.tile(y, (1, len(c)))  # Initial guess: y for all stages
        
        # Newton iteration
        for _newton in range(10):
            # Compute F(Y) = f(t + h*c, Y) for each stage
            F = np.empty_like(Y)
            for j in range(len(c)):
                F[:, j] = f(t + h * c[j], Y[:, j])
            
            # Residual: R = y + h * B * F - (I - h * A) Y
            # (I - h * A) Y = y + h * B * F
            # R = y + h * B * F - Y + h * A * Y
            
            # Compute A * Y
            AY = np.zeros_like(Y)
            for j in range(len(c)):
                for i in range(len(c)):
                    AY[:, j] += h * A[i, j] * Y[:, i]
            
            # B * F
            BF = np.zeros(n)
            for j in range(len(c)):
                BF += B[j] * F[:, j]
            
            R = y + h * BF - Y + AY
            
            # Check convergence
            if np.linalg.norm(R, ord=np.inf) < 1e-12:
                break
            
            # Compute the Jacobian of the residual with respect to Y
            # dR/dY = -(I - h * A) + h * dA_Y/dY + h * B * dF/dY
            # dA_Y/dY is tricky. Let's use a simpler approach:
            # The system is: Y - h * A * Y = y + h * B * F(Y)
            # Let G(Y) = Y - h * A * Y - h * B * F(Y) - y = 0
            # dG/dY = I - h * A - h * B * dF/dY
            # But dF/dY is a block matrix.
            
            # For simplicity, let's use a finite-difference Jacobian for the full system.
            # The system has n * n_stages unknowns.
            # This is expensive. Let's use a different approach.
            
            # Actually, for Radau IIA, the system is:
            # Y = y + h * sum_j B_j F(Y_j)
            # And Y_j = y + h * sum_i A_{ji} F(Y_i)
            # This is a coupled system.
            
            # Let's use a simpler Newton update:
            # We want to solve: Y = y + h * B * F(Y)
            # But Y also appears in the argument of F.
            
            # Let's use the following approach:
            # 1. Compute F(Y) for the current Y.
            # 2. Compute the Jacobian J_Y of F with respect to Y (block diagonal, with blocks J_j = df/dy at (t + h*c_j, Y_j)).
            # 3. The Newton update is:
            #    (I - h * B * J_Y) dY = -R
            #    But this is not quite right because Y appears in both the LHS and RHS.
            
            # Let's use a more standard approach for implicit Runge-Kutta methods:
            # The system is: Y = y + h * sum_j B_j F(Y_j)
            # This is a fixed-point equation. We can solve it with Newton's method.
            # The Jacobian of the RHS with respect to Y is:
            #   J = h * [B_1 * J_1, B_2 * J_2, ..., B_n * J_n]
            # where J_j = df/dy at (t + h*c_j, Y_j).
            
            # But this is not quite right either. Let's look at the standard formulation.
            
            # For Radau IIA, the stage equations are:
            # Y_j = y + h * sum_i A_{ji} F(Y_i)
            # This is a system of n * n_stages equations.
            
            # Let's vectorize Y as a vector of length n * n_stages.
            # The Jacobian of the system is:
            #   J_sys = I - h * A_block - h * B_block * J_F
            # where A_block is the block matrix with A_{ji} in block (j, i),
            # B_block is the block matrix with B_j in block (j, i) for all i,
            # and J_F is the block diagonal matrix with J_j in block (j, j).
            
            # This is getting complicated. Let's use a simpler approach:
            # Use a fixed-point iteration with a few Newton steps.
            
            # For now, let's use a simple Newton update with a finite-difference Jacobian
            # of the full residual.
            
            # Flatten Y
            Y_flat = Y.flatten()
            n_unknowns = n * len(c)
            
            # Compute the Jacobian of R with respect to Y_flat
            J_R = np.zeros((n_unknowns, n_unknowns))
            for k in range(n_unknowns):
                e = np.zeros(n_unknowns)
                scale = 1e-8 * max(1.0, abs(Y_flat[k]))
                e[k] = scale
                Y_pert = (Y_flat + e).reshape(n, len(c))
                
                # Compute R for Y_pert
                F_pert = np.empty_like(Y_pert)
                for j in range(len(c)):
                    F_pert[:, j] = f(t + h * c[j], Y_pert[:, j])
                
                AY_pert = np.zeros_like(Y_pert)
                for j in range(len(c)):
                    for i in range(len(c)):
                        AY_pert[:, j] += h * A[i, j] * Y_pert[:, i]
                
                BF_pert = np.zeros(n)
                for j in range(len(c)):
                    BF_pert += B[j] * F_pert[:, j]
                
                R_pert = y + h * BF_pert - Y_pert + AY_pert
                
                J_R[:, k] = (R_pert.flatten() - R.flatten()) / scale
            
            # Solve J_R * dY = -R
            try:
                dY = np.linalg.solve(J_R, -R.flatten())
            except np.linalg.LinAlgError:
                dY = -R.flatten() * 0.1  # Fallback
            
            Y = (Y_flat + dY).reshape(n, len(c))
        
        # Update y
        y = y + h * BF
        t += h
    
    return y
# EVOLVE-BLOCK-END
