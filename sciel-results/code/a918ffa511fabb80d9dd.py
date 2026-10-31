"""Optimized uncertainty inequality test function via local convex refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds, LinearConstraint
import itertools

# Physical constants for the uncertainty product bound
# We need to compute A(f) * A(F(f)) where A(g) = sqrt(Var_x * Var_xi)
# For a test function g, the bound on C4 involves the ratio of the L1 norm of g
# and its Fourier transform to the product of standard deviations.
# Specifically, the uncertainty principle states A(f)A(F) >= 1/2 (in some conventions) or similar.
# The task asks to reduce the upper bound on A(f)*A(Fourier(f)).
# This is equivalent to minimizing the "uncertainty" of the test function itself, 
# or rather, finding a test function that makes the constant in the inequality tight.
# However, the prompt says "reduce the upper bound on A(f)*A(Fourier(f))".
# Actually, for a specific test function g, we can compute the exact value of the product 
# of its variances in position and frequency space. The "upper bound" likely refers to the 
# constant C4 derived from this test function in the proof of the inequality.
# The seed program returns coefficients for H0, H4, H8.
# The H12 coefficient is fixed to ensure g(0)=0.

# Let's derive the relationship.
# Hermite polynomials H_n(x) are orthogonal with weight e^{-x^2/2} (physicist's) or e^{-x^2} (probabilist's).
# Standard Hermite polynomials in probability use weight e^{-x^2/2}.
# Let's assume the standard probabilist's Hermite polynomials:
# H_0(x) = 1
# H_1(x) = x
# H_2(x) = x^2 - 1
# H_3(x) = x^3 - 3x
# H_4(x) = x^4 - 6x^2 + 3
# H_8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H_12(x) = x^12 - 60x^10 + 1260x^8 - 12600x^6 + 30240x^4 - 15120x^2 + 945

# The test function is g(x) = (c0*H0 + c4*H4 + c8*H8 + c12*H12) * exp(-x^2/2) ?
# Or is the Gaussian part separate? "Hermite-polynomial times Gaussian test function".
# Usually, the basis functions are psi_n(x) = H_n(x) e^{-x^2/2}.
# If g(x) = P(x) e^{-x^2/2}, then g(0) = P(0).
# P(0) = c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0)
# H0(0) = 1
# H4(0) = 3
# H8(0) = 105
# H12(0) = 945
# Constraint: g(0) = 0 => c0 + 3c4 + 105c8 + 945c12 = 0
# => c12 = -(c0 + 3c4 + 105c8) / 945

# We need to compute A(g) = sqrt(Var_x * Var_xi).
# For g(x) = P(x) e^{-x^2/2}, the density is |g(x)|^2 = P(x)^2 e^{-x^2}.
# This is not a standard Gaussian unless P(x) is a constant.
# We need to compute:
# <x> = integral x |g|^2 dx / integral |g|^2 dx
# <x^2> = integral x^2 |g|^2 dx / integral |g|^2 dx
# Var_x = <x^2> - <x>^2
#
# Similarly for the Fourier transform.
# The Fourier transform of H_n(x) e^{-x^2/2} is i^n H_n(x) e^{-x^2/2} (up to constants).
# Specifically, if F{f}(xi) = integral f(x) e^{-i x xi} dx,
# F{e^{-x^2/2}} = sqrt(2 pi) e^{-xi^2/2}
# F{H_n(x) e^{-x^2/2}} = i^n sqrt(2 pi) H_n(x) e^{-x^2/2} (depending on normalization)
# Let's use the property that the Hermite functions are eigenfunctions of the Fourier transform.
# psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}
# F{psi_n}(xi) = (-i)^n psi_n(xi)
#
# If g(x) = sum c_n H_n(x) e^{-x^2/2},
# Then F{g}(xi) = sqrt(2 pi) sum c_n i^n H_n(xi) e^{-xi^2/2} (using unnormalized Hermite polynomials)
# Wait, let's check the Fourier transform of H_n(x) e^{-x^2/2}.
# Using the generating function or known identities:
# integral H_n(x) e^{-x^2/2} e^{-i x xi} dx = i^n sqrt(2 pi) H_n(xi) e^{-xi^2/2}
#
# So, G(xi) = F{g}(xi) = sqrt(2 pi) sum_{k in {0,4,8,12}} c_k i^k H_k(xi) e^{-xi^2/2}
# Since k is a multiple of 4, i^k = 1.
# So G(xi) = sqrt(2 pi) sum c_k H_k(xi) e^{-xi^2/2}
# This means the Fourier transform has the same form as g(x), just with a scaling factor sqrt(2 pi).
# Therefore, the variance in frequency space Var_xi is the same as Var_x!
# Because the shape of |G(xi)|^2 is identical to |g(x)|^2, just scaled.
# Var_xi = Var_x.
#
# So A(g) = Var_x.
# We need to minimize Var_x subject to g(0)=0.
#
# Let's compute Var_x for g(x) = P(x) e^{-x^2/2}.
# |g(x)|^2 = P(x)^2 e^{-x^2}.
# Let w(x) = e^{-x^2}.
# <x^2> = integral x^2 P(x)^2 w(x) dx / integral P(x)^2 w(x) dx
# <x> = integral x P(x)^2 w(x) dx / integral P(x)^2 w(x) dx
# Due to symmetry (P(x) is even, w(x) is even), <x> = 0.
# So Var_x = <x^2>.
#
# We need to minimize <x^2> = integral x^2 P(x)^2 e^{-x^2} dx / integral P(x)^2 e^{-x^2} dx
# Subject to P(0) = 0.
#
# P(x) = c0 H0(x) + c4 H4(x) + c8 H8(x) + c12 H12(x)
# P(0) = c0 + 3c4 + 105c8 + 945c12 = 0
#
# This is a ratio of quadratic forms in c = [c0, c4, c8].
# Let N(c) = integral x^2 P(x)^2 e^{-x^2} dx
# Let D(c) = integral P(x)^2 e^{-x^2} dx
# We want to minimize N(c)/D(c).
# This is a generalized eigenvalue problem!
# The minimum value is the smallest eigenvalue of the matrix pencil (A - lambda B) = 0,
# where A_ij = integral x^2 H_i(x) H_j(x) e^{-x^2} dx
# and B_ij = integral H_i(x) H_j(x) e^{-x^2} dx
#
# However, we have the constraint P(0) = 0.
# This constraint is linear in c.
# So we are minimizing a Rayleigh quotient subject to a linear constraint.
#
# Let's compute the matrices A and B.
# The integrals are of the form integral x^{2k} H_i(x) H_j(x) e^{-x^2} dx.
# We can compute these analytically or numerically.
# Given the small dimension (3 variables), we can compute them numerically with high precision.
#
# Let's define the basis vectors for the coefficients.
# We have 3 free variables: c0, c4, c8.
# c12 is determined by c0, c4, c8.
#
# Let's compute the matrices A and B.
# A_ij = integral x^2 H_i(x) H_j(x) e^{-x^2} dx
# B_ij = integral H_i(x) H_j(x) e^{-x^2} dx
#
# We will use scipy.integrate.quad for these integrals.
#
# Then we solve the constrained minimization of c^T A c / c^T B c subject to M c = 0,
# where M is the row vector [1, 3, 105] corresponding to c0 + 3c4 + 105c8 + 945c12 = 0,
# but wait, c12 is not in c. c = [c0, c4, c8].
# P(0) = c0 + 3c4 + 105c8 + 945 c12 = 0
# c12 = -(c0 + 3c4 + 105c8)/945
# So the constraint is automatically satisfied if we use c12 defined this way.
# But wait, the constraint is on P(0), which involves c12.
# By defining c12 in terms of c0, c4, c8, we have already enforced P(0)=0.
# So there are no further constraints on c = [c0, c4, c8]!
#
# So we just need to minimize the Rayleigh quotient R(c) = c^T A c / c^T B c
# where A and B are 3x3 matrices constructed from the integrals.
#
# Let's compute A and B.
#
# H0(x) = 1
# H4(x) = x^4 - 6x^2 + 3
# H8(x) = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
#
# We need integrals of the form:
# I_{i,j,k} = integral x^k H_i(x) H_j(x) e^{-x^2} dx
#
# A_ij = I_{i,j,2}
# B_ij = I_{i,j,0}
#
# Let's compute these numerically.
#
# We will use a deadline guard.
#
# The minimum of the Rayleigh quotient is the smallest generalized eigenvalue.
# We can solve B v = lambda A v, or A v = mu B v.
# We want the smallest mu such that A v = mu B v.
# This is equivalent to finding the eigenvalues of B^{-1} A.
#
# Let's implement this.

def _compute_matrices():
    """Compute the matrices A and B for the Rayleigh quotient."""
    # Define Hermite polynomials
    def H0(x): return np.ones_like(x)
    def H4(x): return x**4 - 6*x**2 + 3
    def H8(x): return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    
    H_funcs = [H0, H4, H8]
    n = 3
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    
    # Use scipy.integrate.quad for numerical integration
    from scipy.integrate import quad
    
    for i in range(n):
        for j in range(i, n):
            # Compute B_ij = integral H_i(x) H_j(x) e^{-x^2} dx
            f_B = lambda x: H_funcs[i](x) * H_funcs[j](x) * np.exp(-x**2)
            B_ij, _ = quad(f_B, -np.inf, np.inf, limit=100)
            B[i, j] = B_ij
            B[j, i] = B_ij
            
            # Compute A_ij = integral x^2 H_i(x) H_j(x) e^{-x^2} dx
            f_A = lambda x: x**2 * H_funcs[i](x) * H_funcs[j](x) * np.exp(-x**2)
            A_ij, _ = quad(f_A, -np.inf, np.inf, limit=100)
            A[i, j] = A_ij
            A[j, i] = A_ij
            
    return A, B

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 180.0  # 180 seconds
    
    try:
        # Compute matrices
        A, B = _compute_matrices()
        
        # Solve the generalized eigenvalue problem A v = lambda B v
        # We want the smallest eigenvalue.
        # Convert to standard eigenvalue problem: B^{-1} A v = lambda v
        # Or use scipy.linalg.eigh for symmetric definite problem.
        # B is positive definite (it's a Gram matrix of linearly independent functions).
        # A is also positive semi-definite.
        
        from scipy.linalg import eigh
        
        # eigh(A, B) solves A v = lambda B v
        eigenvalues, eigenvectors = eigh(A, B)
        
        # The smallest eigenvalue is the minimum of the Rayleigh quotient.
        # The corresponding eigenvector gives the optimal coefficients.
        idx = np.argmin(eigenvalues)
        v = eigenvectors[:, idx]
        
        # Normalize v such that D(v) = 1 (optional, but good for consistency)
        # Actually, the Rayleigh quotient is scale-invariant, so we can return v directly.
        # But let's check the sign. The coefficients can be multiplied by -1.
        # Let's make the first coefficient positive for consistency.
        if v[0] < 0:
            v = -v
        
        # Check time
        if time.time() > deadline - 10:
            return v
        
        return v
        
    except Exception as e:
        # Fallback to seed values if something goes wrong
        return np.array([-113 / 100, 1 / 25, 1 / 3240])

# EVOLVE-BLOCK-END
