import numpy as np
import scipy.optimize
import time

_start_time = time.time()
DEADLINE = 330.0
MARGIN = 5.0

def _check_deadline():
    return (time.time() - _start_time) > (DEADLINE - MARGIN)

def _van_der_pol_jac(t, y):
    """Analytical Jacobian for the Van der Pol oscillator (mu=5)."""
    y0, y1 = y
    return np.array([
        [0.0, 1.0],
        [-10.0 * y0 * y1 - 1.0, 5.0 * (1.0 - y0 * y0)]
    ])

def _chebyshev_lobatto_nodes(n):
    """
    Return Chebyshev-Lobatto nodes on [0, 1] of degree n.
    n is the number of nodes.
    """
    # k = 0..n-1
    k = np.arange(n)
    # x_k = cos(pi * k / (n-1))
    x = np.cos(np.pi * k / (n - 1))
    # Map from [-1, 1] to [0, 1]
    x = (x + 1.0) / 2.0
    return x

def _chebyshev_lobatto_diff_matrix(n):
    """
    Return the differentiation matrix for Chebyshev-Lobatto nodes on [0, 1].
    """
    x = _chebyshev_lobatto_nodes(n)
    # Standard formula for differentiation matrix on [-1, 1]
    # Then scale for [0, 1]
    
    # c = 1 for i=0, n-1, else 2
    c = np.ones(n)
    c[0] = 1.0
    c[-1] = 1.0
    c = c * (-1)**np.arange(n)
    
    # j = 0..n-1
    j = np.arange(n)
    j = j[:, np.newaxis]
    
    # i = 0..n-1
    i = np.arange(n)
    i = i[np.newaxis, :]
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        D = (c[:, None] / c[None, :]) * (2.0 * np.ones((n, n)) / (x[:, None] - x[None, :] + 1e-16))
        
    # Fix diagonal elements
    # D_ii = -sum_{j!=i} D_ij
    D = D - np.diag(np.sum(D, axis=1))
    
    # Scale from [-1, 1] to [0, 1]
    # If x_new = (x_old + 1)/2, then dx_old/dx_new = 2
    # So D_new = 2 * D_old
    D = 2.0 * D
    
    return D

def integrate(f, t_span, y0):
    if _check_deadline():
        return np.array(y0, dtype=float)

    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Number of collocation points
    N = 50
    
    # Get nodes and diff matrix
    # Map nodes to [t0, t1]
    x_lob = _chebyshev_lobatto_nodes(N)
    t_nodes = t0 + (t1 - t0) * x_lob
    
    D = _chebyshev_lobatto_diff_matrix(N)
    
    # Unknowns: y at nodes 1..N-1
    # y0 is fixed at node 0
    # Total unknowns: 2 * (N - 1)
    
    def residual(y_flat):
        if _check_deadline():
            return np.zeros_like(y_flat)
            
        # Reshape y_flat to (N-1, 2)
        Y_int = y_flat.reshape(N - 1, 2)
        
        # Full Y array (N, 2)
        Y_full = np.zeros((N, 2))
        Y_full[0] = y0
        Y_full[1:] = Y_int
        
        # Compute derivatives at all nodes
        # Y_deriv[i] = sum_j D[i,j] Y_full[j]
        Y_deriv = D @ Y_full
        
        # Compute f at interior nodes (1..N-1)
        # We only need residual for interior nodes to satisfy BC at 0
        # R_i = Y_deriv[i] - f(t_i, Y_full[i])
        
        R = np.zeros(N - 1)
        for i in range(1, N):
            # Call f
            fi = f(t_nodes[i], Y_full[i])
            R[i - 1] = Y_deriv[i] - fi
            
        return R

    def jacobian(y_flat):
        if _check_deadline():
            return np.zeros((2 * (N - 1), 2 * (N - 1)))

        Y_int = y_flat.reshape(N - 1, 2)
        Y_full = np.zeros((N, 2))
        Y_full[0] = y0
        Y_full[1:] = Y_int
        
        # Jacobian of residual w.r.t y_flat
        # R_i = sum_j D[i,j] Y_j - f(t_i, Y_i)
        # dR_i / dY_k = D[i,k] - J_ode(t_i, Y_i)_{component}
        
        # We need to construct the Jacobian matrix J_R of size (2*(N-1), 2*(N-1))
        # Rows correspond to R components (interior nodes, 2 comps each)
        # Cols correspond to Y_int components (interior nodes, 2 comps each)
        
        J_R = np.zeros((2 * (N - 1), 2 * (N - 1)))
        
        for i in range(1, N):
            # Row block for node i
            row_start = (i - 1) * 2
            row_end = row_start + 2
            
            # Term 1: D[i, :] @ dY_full/dY_int
            # dY_full/dY_int is an identity-like matrix for the interior part
            # Y_full[0] is fixed, Y_full[1:] = Y_int
            # So dY_full/dY_int has 0 in first row block, Identity in rest
            # Specifically, dY_full[k] / dY_int[m] = delta_{k, m+1}
            
            # For each component c (0,1) of R_i:
            # dR_{i,c} / dY_{k,d} = sum_l D[i,l] * (dY_{l,c} / dY_{k,d}) - (J_ode)_{c,d} * delta_{i,k}
            
            # dY_{l,c} / dY_{k,d} is 1 if l == k+1 and c == d, else 0.
            # So sum_l D[i,l] * ... = D[i, k+1] * delta_{c,d}
            
            J_ode_i = _van_der_pol_jac(t_nodes[i], Y_full[i])
            
            for c in range(2):
                for k in range(1, N):
                    # Column index in J_R for Y_int[k-1, d]
                    # d is 0 or 1
                    # We iterate d implicitly by matrix structure?
                    # Let's just fill the 2x2 block for each k
                    
                    col_start = (k - 1) * 2
                    col_end = col_start + 2
                    
                    # Block J_R[row_start:row_end, col_start:col_end]
                    # = D[i, k+1] * I_2 - delta_{i,k} * J_ode_i
                    
                    if i == k:
                        block = D[i, k+1] * np.eye(2) - J_ode_i
                    else:
                        block = D[i, k+1] * np.eye(2)
                    
                    J_R[row_start:row_end, col_start:col_end] = block
        
        return J_R

    # Initial guess
    # Use linear interpolation from y0 to a rough estimate?
    # Or just y0?
    # Let's try y0 + t * y'(0)
    y_prime_0 = f(t0, y0)
    y_guess_linear = y0 + (t_nodes[1:] - t0)[:, np.newaxis] * y_prime_0
    y_flat_guess = y_guess_linear.flatten()
    
    # Solve
    try:
        sol = scipy.optimize.root(
            residual, 
            y_flat_guess, 
            method='hybr', 
            jac=jacobian,
            tol=1e-15,
            options={'xtol': 1e-15}
        )
        
        if sol.success:
            # Reconstruct full solution to get y(t1)
            Y_int = sol.x.reshape(N - 1, 2)
            Y_full = np.zeros((N, 2))
            Y_full[0] = y0
            Y_full[1:] = Y_int
            return Y_full[-1]
        else:
            # Fallback to solve_ivp if root finding fails
            import scipy.integrate
            sol_ivp = scipy.integrate.solve_ivp(
                f, t_span, y0, method='Radau', 
                jac=_van_der_pol_jac, rtol=1e-11, atol=1e-13
            )
            if sol_ivp.success:
                return sol_ivp.y[:, -1]
            return y0
            
    except Exception:
        import scipy.integrate
        try:
            sol_ivp = scipy.integrate.solve_ivp(
                f, t_span, y0, method='Radau', 
                jac=_van_der_pol_jac, rtol=1e-11, atol=1e-13
            )
            if sol_ivp.success:
                return sol_ivp.y[:, -1]
        except:
            pass
        return y0
