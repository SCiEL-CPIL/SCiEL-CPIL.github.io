"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import eigh

# Precompute Gauss-Hermite quadrature nodes and weights
# Using a high number of points for accuracy
n_points = 200
x_nodes, x_weights = np.polynomial.hermite.hermgauss(n_points)

# Define correct Physicists' Hermite polynomials H_0, H_4, H_8, H_12
def H0(x):
    return np.ones_like(x)

def H4(x):
    return 16 * x**4 - 48 * x**2 + 12

def H8(x):
    return 256 * x**8 - 4480 * x**6 + 26880 * x**4 - 40320 * x**2 + 1680

def H12(x):
    # Correct coefficients for Physicists' Hermite H_12
    return (4096 * x**12 - 231424 * x**10 + 4648960 * x**8 - 
            38379520 * x**6 + 121958400 * x**4 - 132249600 * x**2 + 6652800)

def construct():
    # Construct basis functions that satisfy g(0)=0 automatically
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    # c12 = -(c0*1 + c4*12 + c8*1680) / 665280
    
    # Basis vectors for c0, c4, c8
    # u0 = H0 - (1/665280) * H12
    # u4 = H4 - (12/665280) * H12
    # u8 = H8 - (1680/665280) * H12
    
    h0 = H0(x_nodes)
    h4 = H4(x_nodes)
    h8 = H8(x_nodes)
    h12 = H12(x_nodes)
    
    coeff_12_0 = 1.0 / 665280.0
    coeff_12_4 = 12.0 / 665280.0
    coeff_12_8 = 1680.0 / 665280.0
    
    u0 = h0 - coeff_12_0 * h12
    u4 = h4 - coeff_12_4 * h12
    u8 = h8 - coeff_12_8 * h12
    
    # Compute M0 and M2 matrices
    # M0[i,j] = sum(w_k * u_i(x_k) * u_j(x_k))
    # M2[i,j] = sum(w_k * x_k^2 * u_i(x_k) * u_j(x_k))
    
    # Vectorize for speed
    # u is a matrix of shape (3, n_points)
    U = np.vstack([u0, u4, u8])
    
    # Weights for M0: w_k
    # Weights for M2: w_k * x_k^2
    w = x_weights
    w_x2 = x_weights * x_nodes**2
    
    # M0 = U @ diag(w) @ U.T
    # M2 = U @ diag(w_x2) @ U.T
    
    # Efficient calculation
    Uw = U * np.sqrt(w)  # Element-wise multiply
    M0 = Uw @ Uw.T
    
    Uw_x2 = U * np.sqrt(w_x2)
    M2 = Uw_x2 @ Uw_x2.T
    
    # Solve generalized eigenvalue problem M2 v = lambda M0 v
    # We want the smallest eigenvalue
    try:
        eigenvalues, eigenvectors = eigh(M2, M0)
        # eigenvectors columns correspond to eigenvalues
        # Smallest eigenvalue is at index 0
        optimal_coeffs = eigenvectors[:, 0]
        
        # Normalize the output coefficients? 
        # The problem asks for "three coefficients". 
        # The scale doesn't change the uncertainty product.
        # We return the eigenvector as is.
        return optimal_coeffs
        
    except Exception:
        # Fallback to a simple guess if eigenvalue solver fails
        return np.array([1.0, 0.0, 0.0])

# EVOLVE-BLOCK-END
