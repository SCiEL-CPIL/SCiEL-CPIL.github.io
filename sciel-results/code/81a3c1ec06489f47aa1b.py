import numpy as np
import time

def construct():
    # Physicists' Hermite polynomials at 0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # Normalization constants N_n = sqrt(pi) * 2^n * n!
    # We can ignore sqrt(pi) as it cancels
    N0 = 1.0
    N4 = 384.0
    N8 = 10321920.0
    N12 = 1962007275520.0
    
    # Coefficients for <x^2> numerator: (n + 0.5) * N_n
    A0 = 0.5 * N0
    A4 = 4.5 * N4
    A8 = 8.5 * N8
    A12 = 12.5 * N12
    
    # Matrix A (diagonal)
    A = np.diag([A0, A4, A8, A12])
    
    # Matrix B (diagonal)
    B = np.diag([N0, N4, N8, N12])
    
    # Constraint vector h
    h = np.array([H0_0, H4_0, H8_0, H12_0])
    
    # Transform to orthonormal basis for the metric B
    # y = B^{-1/2} c
    # Minimize y^T M y / y^T y subject to v^T y = 0
    # M = B^{-1/2} A B^{-1/2}
    # v = B^{-1/2} h
    
    B_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(B)))
    M = B_inv_sqrt @ A @ B_inv_sqrt
    v = B_inv_sqrt @ h
    
    # Project M onto the subspace orthogonal to v
    # P = I - v v^T / (v^T v)
    v_norm_sq = np.dot(v, v)
    P = np.eye(4) - np.outer(v, v) / v_norm_sq
    
    # The eigenvalues of P M P restricted to the subspace v^T y = 0
    # are the stationary values of the Rayleigh quotient.
    # We can compute the eigenvalues of P M P and take the smallest non-zero one?
    # Actually, P M P has 0 as an eigenvalue with multiplicity 1 (along v).
    # The other 3 eigenvalues are the values of interest.
    
    eigenvalues = np.linalg.eigvalsh(P @ M @ P)
    
    # Sort eigenvalues
    eigenvalues = np.sort(eigenvalues)
    
    # The smallest eigenvalue is 0 (corresponding to v).
    # The next smallest is the minimum of the quotient on the subspace.
    min_val = eigenvalues[1]
    
    # We need the coefficients c corresponding to this minimum.
    # Find eigenvector y_min
    eigenvectors = np.linalg.eigh(P @ M @ P)
    # eigenvectors[1] corresponds to eigenvalues[1]
    y_min = eigenvectors[1]
    
    # Convert back to c
    c_min = B_inv_sqrt.T @ y_min  # B_inv_sqrt is diagonal, so T is same
    
    # Normalize c_min to match the expected scale?
    # The problem asks for "three coefficients".
    # The scale doesn't matter for the ratio, but let's return a reasonable vector.
    # We can normalize such that max |c| = 1 or similar.
    # Or just return c_min.
    
    # The return value is "three coefficients of the Hermite combination (degrees 0, 4, 8; the degree-12 weight is fixed...)"
    # So we return c0, c4, c8.
    # c12 is determined by the constraint.
    
    c0, c4, c8, c12 = c_min
    
    # Check if the constraint is satisfied
    # c0 + 12 c4 + 1680 c8 + 665280 c12 should be 0
    
    return np.array([c0, c4, c8])

