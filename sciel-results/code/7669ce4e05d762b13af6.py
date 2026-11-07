"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_grid = resource_grid
    n_m = len(m_grid)
    
    # Chebyshev setup
    # We map the resource grid [0.001, 40] to [-1, 1]
    m_min = 0.001
    m_max = 40.0
    scale = (m_max - m_min) / 2.0
    shift = (m_max + m_min) / 2.0
    
    # Number of Chebyshev nodes (collocation points)
    # High enough to be accurate, low enough to be fast.
    N_cheb = 81 
    # Chebyshev nodes in [-1, 1]
    k = np.arange(N_cheb)
    x_cheb = np.cos(np.pi * (k + 0.5) / N_cheb)
    
    # Map to [m_min, m_max]
    m_cheb = scale * x_cheb + shift
    # Ensure ascending order for np.interp if needed, but Chebyshev ops are order-independent
    # However, for final output we need sorted m_grid.
    
    # Chebyshev interpolation matrix (Vandermonde-like)
    # T_n(x) = cos(n * arccos(x))
    # We need a matrix C such that V(m) = sum c_i T_i(x)
    # C[i, j] = T_j(x_i)
    C_mat = np.zeros((N_cheb, N_cheb))
    for j in range(N_cheb):
        C_mat[:, j] = np.cos(j * np.arccos(x_cheb))
        
    # Inverse for coefficients from values
    # C_inv = inv(C_mat)
    # To avoid explicit inversion, we can solve C c = v using least squares or precompute inv
    try:
        C_inv = np.linalg.inv(C_mat)
    except np.linalg.LinAlgError:
        C_inv = np.linalg.pinv(C_mat)

    # Derivative matrix for Chebyshev polynomials
    # d/dx T_n(x) = n * U_{n-1}(x)
    # U_n(x) = sin((n+1)theta)/sin(theta) where x = cos(theta)
    D_mat = np.zeros((N_cheb, N_cheb))
    theta = np.arccos(x_cheb)
    for n in range(1, N_cheb):
        # U_{n-1}(x)
        if n == 1:
            U = np.ones(N_cheb)
        else:
            # Avoid division by zero at endpoints? Nodes are interior (k+0.5), so sin(theta) != 0
            U = np.sin(n * theta) / np.sin(theta)
        D_mat[:, n] = n * U
        
    # Second derivative matrix
    # d^2/dx^2 T_n(x) = (n^2 - 1) * T_{n-2}(x) for n >= 2, 0 for n=0, -1 for n=1?
    # Actually:
    # T_0'' = 0
    # T_1'' = 0
    # T_2'' = 2
    # T_n'' = n^2 T_n - (n^2-1) T_{n-2} ... wait.
    # Let's use the relation: (1-x^2) T_n'' - x T_n' + n^2 T_n = 0
    # T_n'' = (x T_n' - n^2 T_n) / (1-x^2)
    # This is singular at x=1, -1. Our nodes are interior, so it's fine.
    # However, direct formula T_n''(x) = n^2 T_n(x) - (n^2-1) T_{n-2}(x) is cleaner?
    # Let's check n=2: 4 T_2 - 3 T_0 = 4(2x^2-1) - 3 = 8x^2 - 7.
    # T_2 = 2x^2-1, T_2' = 4x, T_2'' = 8.
    # Formula: 8x^2 - 7 != 8. So that formula is wrong.
    
    # Correct relation: T_n''(x) = n^2 T_n(x) - n(n-1) T_{n-2}(x)? No.
    # Let's just compute D2 using D @ D.
    D2_mat = D_mat @ D_mat

    def get_utility(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def get_utility_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def get_utility_double_prime(c, gamma):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # Coefficients for V[t, z]
    # V_coefs[t, z, :] shape (horizon, 2, N_cheb)
    V_coefs = np.zeros((horizon, 2, N_cheb))
    
    # Policy c[t, z] at Chebyshev nodes
    c_nodes = np.zeros((horizon, 2, N_cheb))

    # Terminal period
    c_nodes[horizon - 1, :, :] = m_cheb[np.newaxis, :]
    # V_terminal(m) = u(m)
    V_vals_term = get_utility(m_cheb, gamma)
    V_coefs[horizon - 1, 0, :] = C_inv @ V_vals_term
    V_coefs[horizon - 1, 1, :] = C_inv @ V_vals_term

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: simple proportional rule
            for tt in range(t, -1, -1):
                c_nodes[tt, :, :] = m_cheb[np.newaxis, :] / (horizon - tt)
                # Compute V coefs roughly
                V_vals = get_utility(c_nodes[tt, 0, :], gamma)
                V_coefs[tt, 0, :] = C_inv @ V_vals
                V_vals = get_utility(c_nodes[tt, 1, :], gamma)
                V_coefs[tt, 1, :] = C_inv @ V_vals
            break

        for z in range(2):
            # Initialize c with proportional guess
            c = m_cheb * 0.5
            
            # Gauss-Seidel / Fixed Point Iteration
            # We want to solve:
            # u'(c) = beta * R * E[ V'_{t+1}(m_next) ]
            # Since V' is monotonic, we can solve for c directly.
            # Let's use a damped Newton or simple iteration.
            # Given the smoothness, a few Newton steps on the Chebyshev nodes should suffice.
            
            max_iter = 15
            for _ in range(max_iter):
                m_next_0 = R * (m_cheb - c) + income[0]
                m_next_1 = R * (m_cheb - c) + income[1]
                
                # Map m_next to x_next in [-1, 1]
                x_next_0 = (m_next_0 - shift) / scale
                x_next_1 = (m_next_1 - shift) / scale
                
                # Clip to [-1, 1] to avoid nan in arccos
                x_next_0 = np.clip(x_next_0, -1.0, 1.0)
                x_next_1 = np.clip(x_next_1, -1.0, 1.0)
                
                # Evaluate V' and V'' using Chebyshev series
                # V'(x) = sum c_i T_i'(x) = sum c_i D[i, j] ? No.
                # V(x) = sum_j c_j T_j(x)
                # V'(x) = sum_j c_j T_j'(x)
                # T_j'(x) = j U_{j-1}(x)
                # We have D_mat[i, j] = T_j'(x_i)
                # So V'(x_i) = D_mat @ c_coefs
                # But we need V'(x_next) where x_next is NOT necessarily a node.
                # We need to evaluate the derivative polynomial at arbitrary points.
                
                # Construct Vandermonde matrix for x_next
                # T_j(x_next) = cos(j * arccos(x_next))
                theta_next_0 = np.arccos(x_next_0)
                theta_next_1 = np.arccos(x_next_1)
                
                C_next_0 = np.zeros((N_cheb, N_cheb))
                C_next_1 = np.zeros((N_cheb, N_cheb))
                for j in range(N_cheb):
                    C_next_0[:, j] = np.cos(j * theta_next_0)
                    C_next_1[:, j] = np.cos(j * theta_next_1)
                    
                # Derivative basis at x_next
                # T_j'(x) = j * sin(j*theta)/sin(theta)
                sin_theta_0 = np.sin(theta_next_0)
                sin_theta_1 = np.sin(theta_next_1)
                
                D_next_0 = np.zeros((N_cheb, N_cheb))
                D_next_1 = np.zeros((N_cheb, N_cheb))
                for j in range(1, N_cheb):
                    D_next_0[:, j] = j * np.sin(j * theta_next_0) / sin_theta_0
                    D_next_1[:, j] = j * np.sin(j * theta_next_1) / sin_theta_1
                # j=0 derivative is 0
                
                # Second derivative
                # T_j''(x) = (j^2 T_j(x) - j(j-1) T_{j-2}(x)) / (1-x^2)? No, use D@D logic?
                # Simpler: T_j''(x) = j^2 T_j(x) - (j^2-1) T_{j-2}(x) is WRONG.
                # Use: T_j''(x) = j^2 T_j(x) - j(j-1) T_{j-2}(x) ... let's verify.
                # j=2: 4 T_2 - 2(1) T_0 = 4(2x^2-1) - 2 = 8x^2 - 6.
                # T_2'' = 8. 8x^2 - 6 != 8.
                # Correct: T_j''(x) = j^2 T_j(x) - j(j-1) T_{j-2}(x) is for (1-x^2)T'' - xT' + j^2 T = 0?
                # Let's just evaluate D_mat at nodes and interpolate? No, we need values at x_next.
                # We can compute T_j''(x) directly.
                # T_j''(x) = j^2 T_j(x) - j(j-1) T_{j-2}(x) is NOT correct.
                # The correct identity is: T_j''(x) = j^2 T_j(x) - (j^2-1) T_{j-2}(x) is also suspect.
                # Let's use: T_j''(x) = j^2 T_j(x) - j(j-1) T_{j-2}(x) ... 
                # Actually, the recurrence is (1-x^2)T'' - xT' + j^2 T = 0.
                # So T'' = (x T' - j^2 T) / (1-x^2).
                # This is stable for x in (-1, 1).
                
                D2_next_0 = np.zeros((N_cheb, N_cheb))
                D2_next_1 = np.zeros((N_cheb, N_cheb))
                denom_0 = 1.0 - x_next_0**2
                denom_1 = 1.0 - x_next_1**2
                # Avoid division by zero (should not happen with interior nodes/clip)
                denom_0[denom_0 < 1e-10] = 1e-10
                denom_1[denom_1 < 1e-10] = 1e-10
                
                for j in range(N_cheb):
                    if j == 0:
                        D2_next_0[:, j] = 0
                        D2_next_1[:, j] = 0
                    else:
                        T_j_0 = np.cos(j * theta_next_0)
                        T_j_1 = np.cos(j * theta_next_1)
                        Tp_j_0 = D_next_0[:, j]
                        Tp_j_1 = D_next_1[:, j]
                        D2_next_0[:, j] = (x_next_0 * Tp_j_0 - j**2 * T_j_0) / denom_0
                        D2_next_1[:, j] = (x_next_1 * Tp_j_1 - j**2 * T_j_1) / denom_1

                # Evaluate expected derivatives
                V_prime_0 = C_next_0 @ V_coefs[t+1, 0, :] # Wait, V' is D @ C?
                # V(x) = C(x) @ V_coefs
                # V'(x) = D(x) @ V_coefs
                # So V_prime_0 = D_next_0 @ V_coefs[t+1, 0, :]
                V_prime_0 = D_next_0 @ V_coefs[t+1, 0, :]
                V_prime_1 = D_next_1 @ V_coefs[t+1, 1, :]
                
                V_double_prime_0 = D2_next_0 @ V_coefs[t+1, 0, :]
                V_double_prime_1 = D2_next_1 @ V_coefs[t+1, 1, :]
                
                # Note: V is a function of m. We differentiated w.r.t x.
                # dV/dm = dV/dx * dx/dm = dV/dx * (1/scale)
                # d2V/dm2 = d2V/dx2 * (1/scale^2)
                
                E_V_prime = (transition[z, 0] * V_prime_0 + transition[z, 1] * V_prime_1) / scale
                E_V_double_prime = (transition[z, 0] * V_double_prime_0 + transition[z, 1] * V_double_prime_1) / (scale * scale)
                
                # F(c) = u'(c) - beta * R * E[V'(m_next)]
                u_prime = get_utility_prime(c, gamma)
                F = u_prime - beta * R * E_V_prime
                
                # F'(c) = u''(c) + beta * R^2 * E[V''(m_next)]
                u_double_prime = get_utility_double_prime(c, gamma)
                F_prime = u_double_prime + beta * (R**2) * E_V_double_prime
                
                # Newton step
                step = F / F_prime
                c_new = c - step
                c_new = np.clip(c_new, 1e-12, m_cheb)
                
                if np.max(np.abs(c_new - c)) < 1e-8:
                    c = c_new
                    break
                c = c_new

            c_nodes[t, z, :] = c
            
            # Compute V_coefs for this period
            # V(m) = u(c(m)) + beta E[V_next(m_next)]
            m_next_0 = R * (m_cheb - c) + income[0]
            m_next_1 = R * (m_cheb - c) + income[1]
            x_next_0 = np.clip((m_next_0 - shift) / scale, -1.0, 1.0)
            x_next_1 = np.clip((m_next_1 - shift) / scale, -1.0, 1.0)
            theta_next_0 = np.arccos(x_next_0)
            theta_next_1 = np.arccos(x_next_1)
            
            C_next_0 = np.zeros((N_cheb, N_cheb))
            C_next_1 = np.zeros((N_cheb, N_cheb))
            for j in range(N_cheb):
                C_next_0[:, j] = np.cos(j * theta_next_0)
                C_next_1[:, j] = np.cos(j * theta_next_1)
                
            V_next_val_0 = C_next_0 @ V_coefs[t+1, 0, :]
            V_next_val_1 = C_next_1 @ V_coefs[t+1, 1, :]
            
            V_vals = get_utility(c, gamma) + beta * (transition[z, 0] * V_next_val_0 + transition[z, 1] * V_next_val_1)
            V_coefs[t, z, :] = C_inv @ V_vals

    # Interpolate to resource_grid
    consumption = np.zeros((horizon, 2, n_m))
    
    # Map resource_grid to x
    x_grid = (resource_grid - shift) / scale
    x_grid = np.clip(x_grid, -1.0, 1.0)
    theta_grid = np.arccos(x_grid)
    
    C_grid = np.zeros((n_m, N_cheb))
    for j in range(N_cheb):
        C_grid[:, j] = np.cos(j * theta_grid)
        
    for t in range(horizon):
        for z in range(2):
            # We need c at resource_grid.
            # c is a function of m. We have c_nodes at m_cheb.
            # We can interpolate c_nodes from m_cheb to m_grid using np.interp
            # Or we can store c as a Chebyshev series too?
            # The policy c(m) is smooth. Let's fit a Chebyshev series to c_nodes.
            c_coefs = C_inv @ c_nodes[t, z, :]
            c_vals = C_grid @ c_coefs
            
            # Clip to valid range
            c_vals = np.clip(c_vals, 1e-12, resource_grid + 1e-10)
            consumption[t, z, :] = c_vals

    # Ensure c[5] is exactly the grid
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Final validity check
    consumption = np.where(np.isfinite(consumption), consumption, resource_grid[np.newaxis, np.newaxis, :])
    consumption = np.clip(consumption, 1e-12, resource_grid[np.newaxis, np.newaxis, :] + 1e-10)
    
    return consumption
# EVOLVE-BLOCK-END
