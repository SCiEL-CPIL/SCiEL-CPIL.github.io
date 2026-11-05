"""Chebyshev Ridge Regression closure with AICc degree selection."""
# EVOLVE-BLOCK-START
import numpy as np


def construct(training_x, training_subgrid_tendency):
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    n = len(x)
    if n < 6:
        return np.zeros(6)

    # Scale x to [-1, 1] for Chebyshev stability
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max - x_min < 1e-12:
        x_min = x_min - 0.5
        x_max = x_max + 0.5
    alpha = 2.0 / (x_max - x_min)
    beta = -1.0 - alpha * x_min
    
    xi = alpha * x + beta
    
    # Precompute Chebyshev polynomials T_0 to T_5 for all xi
    # T_0 = 1
    # T_1 = xi
    # T_k = 2*xi*T_{k-1} - T_{k-2}
    T = np.zeros((n, 6))
    T[:, 0] = 1.0
    T[:, 1] = xi
    for k in range(2, 6):
        T[:, k] = 2.0 * xi * T[:, k-1] - T[:, k-2]
        
    # Ridge regression lambda
    # Heuristic: small relative to data variance
    y_var = np.var(y)
    if y_var < 1e-12:
        y_var = 1.0
    lam = 1e-6 * y_var
    
    best_aicc = np.inf
    best_coeff_x = np.zeros(6)
    
    # Conversion matrix from Chebyshev coeffs (c_T) to Monomial coeffs in xi (m_xi)
    # Based on:
    # T_0 = 1
    # T_1 = xi
    # T_2 = 2xi^2 - 1
    # T_3 = 4xi^3 - 3xi
    # T_4 = 8xi^4 - 8xi^2 + 1
    # T_5 = 16xi^5 - 20xi^3 + 5xi
    #
    # m_0 = c_0 - c_2 + c_4
    # m_1 = c_1 - 3c_3 + 5c_5
    # m_2 = 2c_2 - 8c_4
    # m_3 = 4c_3 - 20c_5
    # m_4 = 8c_4
    # m_5 = 16c_5
    
    # Conversion matrix M such that m_xi = M @ c_T
    M_cheb_to_mono_xi = np.zeros((6, 6))
    M_cheb_to_mono_xi[0, 0] = 1.0
    M_cheb_to_mono_xi[0, 2] = -1.0
    M_cheb_to_mono_xi[0, 4] = 1.0
    
    M_cheb_to_mono_xi[1, 1] = 1.0
    M_cheb_to_mono_xi[1, 3] = -3.0
    M_cheb_to_mono_xi[1, 5] = 5.0
    
    M_cheb_to_mono_xi[2, 2] = 2.0
    M_cheb_to_mono_xi[2, 4] = -8.0
    
    M_cheb_to_mono_xi[3, 3] = 4.0
    M_cheb_to_mono_xi[3, 5] = -20.0
    
    M_cheb_to_mono_xi[4, 4] = 8.0
    
    M_cheb_to_mono_xi[5, 5] = 16.0
    
    for degree in range(1, 6):
        # Design matrix for this degree
        T_d = T[:, :degree+1]
        
        # Ridge Regression: (T^T T + lambda I) c = T^T y
        try:
            A = T_d.T @ T_d + lam * np.eye(degree+1)
            b = T_d.T @ y
            c_cheb = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue
            
        # Convert to monomial in xi
        m_xi = M_cheb_to_mono_xi @ c_cheb
        m_xi = m_xi[:degree+1] # Zero out higher terms if any (should be 0)
        
        # Convert monomial in xi to monomial in x
        # xi = alpha * x + beta
        # We have P(xi) = sum_{k=0}^D m_k xi^k
        # We want Q(x) = sum_{j=0}^D q_j x^j
        # Substitute xi = alpha x + beta
        # Use polynomial multiplication/expansion
        
        # Initialize result coefficients
        q = np.zeros(degree+1)
        
        # For each term m_k * (alpha x + beta)^k, expand and add to q
        for k in range(degree + 1):
            if abs(m_xi[k]) < 1e-15:
                continue
            # Expand (alpha x + beta)^k
            # Coefficients of x^j in (alpha x + beta)^k are C(k, j) alpha^j beta^(k-j)
            for j in range(k + 1):
                # Binomial coefficient
                # Compute C(k, j)
                c_kj = 1.0
                for i in range(j):
                    c_kj = c_kj * (k - i) / (i + 1)
                term_coeff = m_xi[k] * c_kj * (alpha ** j) * (beta ** (k - j))
                q[j] += term_coeff
                
        # Ensure length 6
        coeff_x = np.zeros(6)
        coeff_x[:degree+1] = q
        
        # Calculate AICc
        # Predicted y
        y_pred = T_d @ c_cheb
        rss = np.sum((y - y_pred) ** 2)
        if rss < 1e-12:
            rss = 1e-12
        
        k_params = degree + 1
        # AICc = n * ln(RSS/n) + k * (n + k + 1) / (n - k - 1)
        if n - k_params - 1 <= 0:
            aicc = np.inf
        else:
            aicc = n * np.log(rss / n) + k_params * (n + k_params + 1) / (n - k_params - 1)
            
        if aicc < best_aicc:
            best_aicc = aicc
            best_coeff_x = coeff_x
            
    # Ensure finite
    best_coeff_x = np.where(np.isfinite(best_coeff_x), best_coeff_x, 0.0)
    return best_coeff_x
# EVOLVE-BLOCK-END
