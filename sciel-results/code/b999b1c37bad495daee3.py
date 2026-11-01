import numpy as np
from scipy.optimize import differential_evolution

def construct():
    # Indices of the Hermite polynomials
    indices = [0, 4, 8, 12]
    
    # Precompute factorials and powers of 2
    # G_jj = sqrt(pi) * 2^j * j!
    # X_jk involves j, k
    # K_jj = 4 * j^2 * sqrt(pi) * 2^(j-1) * (j-1)!
    
    # We need to construct 4x4 matrices G, X, K for the basis h_j = H_j exp(-x^2/2)
    # Note: The weight for orthogonality is exp(-x^2).
    
    G = np.zeros((4, 4))
    X = np.zeros((4, 4))
    K = np.zeros((4, 4))
    
    import math
    sqrt_pi = math.sqrt(math.pi)
    
    for i, j in enumerate(indices):
        # G_ii
        G[i, i] = sqrt_pi * (2 ** j) * math.factorial(j)
        
        # K_ii
        if j == 0:
            K[i, i] = 0.0
        else:
            K[i, i] = 4 * (j ** 2) * sqrt_pi * (2 ** (j - 1)) * math.factorial(j - 1)
            
    for i, j in enumerate(indices):
        for k_idx, k in enumerate(indices):
            # X_jk = int x^2 H_j H_k exp(-x^2)
            # Formula: x^2 H_j = 0.25 H_{j+2} + (j + 0.5) H_j + j(j-1) H_{j-2}
            # Integral is sum of terms where H_m matches H_k
            val = 0.0
            # Term 1: 0.25 H_{j+2}. Non-zero if k == j+2
            if k == j + 2:
                val += 0.25 * sqrt_pi * (2 ** (j + 2)) * math.factorial(j + 2)
            # Term 2: (j + 0.5) H_j. Non-zero if k == j
            if k == j:
                val += (j + 0.5) * sqrt_pi * (2 ** j) * math.factorial(j)
            # Term 3: j(j-1) H_{j-2}. Non-zero if k == j-2
            if k == j - 2:
                val += j * (j - 1) * sqrt_pi * (2 ** (j - 2)) * math.factorial(j - 2)
            
            X[i, k_idx] = val
            # Symmetry
            # X is symmetric because x^2 is even and H_j H_k is even?
            # H_j(-x) = (-1)^j H_j(x).
            # x^2 H_j H_k is even if j+k is even.
            # Our indices are 0,4,8,12 (all even). So j+k is even.
            # So X is symmetric.
            
    # Constraint: c_0 - 3c_4 + 105c_8 - 10395 c_12 = 0
    # c_12 = (c_0 - 3c_4 + 105c_8) / 10395
    # Let u = [c_0, c_4, c_8]
    # c = M u
    # M is 4x3
    M = np.zeros((4, 3))
    M[0, 0] = 1.0
    M[1, 1] = 1.0
    M[2, 2] = 1.0
    M[3, 0] = 1.0 / 10395.0
    M[3, 1] = -3.0 / 10395.0
    M[3, 2] = 105.0 / 10395.0
    
    # Project matrices to 3D
    G3 = M.T @ G @ M
    X3 = M.T @ X @ M
    K3 = M.T @ K @ M
    
    def objective(u):
        # u is [c0, c4, c8]
        # Calculate norms
        norm_sq = u @ G3 @ u
        x_norm_sq = u @ X3 @ u
        k_norm_sq = u @ K3 @ u
        
        if norm_sq <= 0:
            return 1e100
            
        sigma_x_sq = x_norm_sq / norm_sq
        sigma_k_sq = k_norm_sq / norm_sq
        
        return sigma_x_sq * sigma_k_sq

    # Use differential evolution for global optimization
    # Bounds for coefficients
    # The seed was [-1.13, 0.04, 0.0003]
    # Let's use a wider range to ensure global minimum
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Run optimization
    result = differential_evolution(objective, bounds, maxiter=1000, popsize=15, tol=1e-12, seed=42)
    
    best_u = result.x
    
    # Return the three coefficients
    return best_u

if __name__ == "__main__":
    coeffs = construct()
    print(coeffs)
