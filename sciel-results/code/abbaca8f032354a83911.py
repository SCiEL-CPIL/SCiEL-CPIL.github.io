"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh
from math import factorial

def _precompute_constants():
    """Precompute all constants needed for the bound computation."""
    sqrt_pi = np.sqrt(np.pi)
    degrees = np.array([0, 4, 8, 12], dtype=float)
    
    # Norms: norm_n = 2^n * n! * sqrt(pi)
    norms = np.zeros(4)
    for i, n in enumerate(degrees):
        n_int = int(n)
        norms[i] = (2.0 ** n_int) * factorial(n_int) * sqrt_pi
    
    # Diagonal M coefficients: diag_M_n = (n + 0.5) * norm_n
    diag_M = (degrees + 0.5) * norms
    
    # Off-diagonal integrals for |n_i - n_j| == 2
    # For pair (i, j) with n_i > n_j and n_i = n_j + 2:
    #   integral = 2^{n_j} * (n_j + 2)! * sqrt_pi
    # For pair (i, j) with n_j > n_i and n_j = n_i + 2:
    #   integral = n_i * (n_i - 1) * 2^{n_i - 4} * (n_i - 2)! * sqrt_pi
    # Both formulas give the same value when the degrees differ by 2.
    off_diag = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if i >= j:
                continue
            ni = int(degrees[i])
            nj = int(degrees[j])
            if abs(ni - nj) == 2:
                n_min = min(ni, nj)
                n_max = max(ni, nj)
                # Use the formula: integral = 2^{n_min} * n_max! * sqrt_pi
                # This works because if n_max = n_min + 2:
                #   2^{n_min} * (n_min + 2)! * sqrt_pi
                # And the other formula gives the same result.
                integral = (2.0 ** n_min) * factorial(n_max) * sqrt_pi
                off_diag[i, j] = integral
                off_diag[j, i] = integral
    
    # Hermite polynomial values at 0
    H_vals_0 = np.array([1.0, -12.0, 1680.0, -103950.0])
    
    return norms, diag_M, off_diag, H_vals_0, degrees

def _bound_and_grad(x, norms, diag_M, off_diag, H_vals_0, degrees):
    """
    Compute the bound M/N and its gradient with respect to x = [c0, c4, c8].
    c12 is determined by g(0) = 0: c12 = (c0 + c4*H4_0 + c8*H8_0) / (-H12_0)
    """
    c0, c4, c8 = x[0], x[1], x[2]
    H0_0, H4_0, H8_0, H12_0 = H_vals_0
    
    # Compute c12 from the constraint g(0) = 0
    c12 = (c0 + c4 * H4_0 + c8 * H8_0) / (-H12_0)
    
    # Full coefficient vector
    coeffs = np.array([c0, c4, c8, c12])
    
    # Compute N = sum_i coeffs[i]^2 * norms[i]
    N = np.dot(coeffs ** 2, norms)
    
    # Compute M = sum_i coeffs[i]^2 * diag_M[i] + sum_{i!=j} coeffs[i]*coeffs[j]*off_diag[i,j]
    M = np.dot(coeffs ** 2, diag_M)
    for i in range(4):
        for j in range(4):
            if i < j and off_diag[i, j] != 0:
                M += 2.0 * coeffs[i] * coeffs[j] * off_diag[i, j]
    
    if N <= 0:
        return np.inf, np.zeros(3)
    
    bound = M / N
    
    # Compute gradient of bound = M/N with respect to x = [c0, c4, c8]
    # d(bound)/dx_k = (dM/dx_k * N - M * dN/dx_k) / N^2
    # 
    # dN/dx_k = 2 * coeffs[k] * norms[k] * dc_k/dx_k
    # dM/dx_k = 2 * coeffs[k] * diag_M[k] * dc_k/dx_k + 2 * sum_{j != k} coeffs[j] * off_diag[k,j] * dc_j/dx_k + 2 * sum_{j != k} coeffs[j] * off_diag[j,k] * dc_k/dx_k
    # 
    # Where dc_k/dx_k is the derivative of coeffs[k] with respect to x_k.
    # coeffs[0] = c0 = x[0], so dc0/dx0 = 1
    # coeffs[1] = c4 = x[1], so dc4/dx1 = 1
    # coeffs[2] = c8 = x[2], so dc8/dx2 = 1
    # coeffs[3] = c12 = (c0 + c4*H4_0 + c8*H8_0) / (-H12_0)
    #   dc12/dx0 = 1 / (-H12_0)
    #   dc12/dx1 = H4_0 / (-H12_0)
    #   dc12/dx2 = H8_0 / (-H12_0)
    
    dc_dc12 = np.array([1.0, H4_0, H8_0]) / (-H12_0)
    
    # Derivatives of coeffs with respect to x
    # dcoeffs[i]/dx[k]
    dcoeffs_dx = np.zeros((4, 3))
    dcoeffs_dx[0, 0] = 1.0  # dc0/dx0
    dcoeffs_dx[1, 1] = 1.0  # dc4/dx1
    dcoeffs_dx[2, 2] = 1.0  # dc8/dx2
    dcoeffs_dx[3, :] = dc_dc12  # dc12/dx0, dc12/dx1, dc12/dx2
    
    grad = np.zeros(3)
    for k in range(3):
        # dN/dx_k = 2 * sum_i coeffs[i] * norms[i] * dcoeffs_dx[i, k]
        dN_dk = 2.0 * np.dot(coeffs * norms, dcoeffs_dx[:, k])
        
        # dM/dx_k = 2 * sum_i coeffs[i] * diag_M[i] * dcoeffs_dx[i, k] 
        #          + 2 * sum_{i!=j} coeffs[j] * off_diag[i,j] * (dcoeffs_dx[i,k] * coeffs[j] + coeffs[i] * dcoeffs_dx[j,k])
        # Simplified: dM/dx_k = 2 * sum_i coeffs[i] * (diag_M[i] * dcoeffs_dx[i,k] + sum_j off_diag[i,j] * coeffs[j] * dcoeffs_dx[i,k])
        #            + 2 * sum_j coeffs[j] * (sum_i off_diag[i,j] * coeffs[i] * dcoeffs_dx[j,k])
        # Actually, let's be more careful:
        # M = sum_i c_i^2 * diag_M[i] + sum_{i!=j} c_i * c_j * off_diag[i,j]
        # dM/dx_k = 2 * sum_i c_i * diag_M[i] * dc_i/dx_k + sum_{i!=j} (dc_i/dx_k * c_j + c_i * dc_j/dx_k) * off_diag[i,j]
        
        dM_dk = 0.0
        for i in range(4):
            dM_dk += 2.0 * coeffs[i] * diag_M[i] * dcoeffs_dx[i, k]
            for j in range(4):
                if i != j:
                    dM_dk += (dcoeffs_dx[i, k] * coeffs[j] + coeffs[i] * dcoeffs_dx[j, k]) * off_diag[i, j]
        
        grad[k] = (dM_dk * N - M * dN_dk) / (N * N)
    
    return bound, grad

def construct():
    """
    Tune a Fourier-invariant Hermite-polynomial times Gaussian test function to reduce
    the upper bound on A(f)*A(Fourier(f)).
    
    Returns: three coefficients [c0, c4, c8] of the Hermite combination.
    """
    # Precompute constants
    norms, diag_M, off_diag, H_vals_0, degrees = _precompute_constants()
    
    # Construct the 4x4 matrices D_N and D_M
    D_N = np.diag(norms)
    D_M = np.diag(diag_M) + off_diag
    
    # Construct the transformation matrix S such that c = S * u
    # c = [c0, c4, c8, c12]^T
    # u = [c0, c4, c8]^T
    # c12 = (c0 + c4*H4_0 + c8*H8_0) / (-H12_0)
    H0_0, H4_0, H8_0, H12_0 = H_vals_0
    k0 = 1.0 / (-H12_0)
    k1 = H4_0 / (-H12_0)
    k2 = H8_0 / (-H12_0)
    
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [k0, k1, k2]
    ])
    
    # Compute reduced matrices
    # A_red = S^T D_M S
    # B_red = S^T D_N S
    A_red = S.T @ D_M @ S
    B_red = S.T @ D_N @ S
    
    # Solve generalized eigenvalue problem A_red v = lambda B_red v
    # eigh returns eigenvalues in ascending order
    try:
        eigenvalues, eigenvectors = eigh(A_red, B_red)
        # The minimum bound is the smallest eigenvalue
        # The corresponding eigenvector gives the optimal coefficients
        min_idx = 0
        optimal_u = eigenvectors[:, min_idx]
    except Exception:
        # Fallback to previous method if eigenvalue solver fails
        x0 = np.array([-113/100, 1/25, 1/3240])
        return x0
    
    return optimal_u

# EVOLVE-BLOCK-END
