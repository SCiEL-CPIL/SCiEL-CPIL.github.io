"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import factorial
from math import sqrt, pi


def _hermite_values_at_zero(max_n=24):
    """Compute H_n(0) for even n up to max_n."""
    h0 = np.zeros(max_n + 1)
    for n in range(0, max_n + 1, 2):
        # H_n(0) for even n: (-1)^(n/2) * (n-1)!! * 2^(n/2) / (n/2)! * (n/2)!
        # Actually, H_n(0) = (-1)^(n/2) * (n-1)!! for even n
        # (n-1)!! = (n-1)(n-3)...1 = n! / (2^(n/2) * (n/2)!)
        # So H_n(0) = (-1)^(n/2) * n! / (2^(n/2) * (n/2)!)
        h0[n] = (-1) ** (n // 2) * factorial(n) / (2 ** (n // 2) * factorial(n // 2))
    return h0


def _norm_factor(n):
    """Normalization factor for the Hermite function basis."""
    # The L2 norm of e^{-x^2/2} H_n(x) is sqrt(2^n * n! * sqrt(pi))
    return sqrt(2 ** n * factorial(n) * sqrt(pi))


def construct():
    t0 = time.time()
    deadline = t0 + 170  # Safety margin: 170s out of 180s

    # H_n(0) values for even n
    h0 = _hermite_values_at_zero(12)
    h00, h04, h08, h012 = h0[0], h0[4], h0[8], h0[12]

    # The constraint g(0) = 0 means:
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # So c12 = -(c0*h00 + c4*h04 + c8*h08) / h012

    # The uncertainty bound for this class of functions can be expressed as:
    # C4 = (1/4) * min_{c: c^T H0_vec = 0} (c^T M c)
    # where M is a matrix encoding the L1/L2 interactions

    # For the specific structure with Hermite polynomials, the bound is:
    # C4 = (1/4) * min (sum_{i,j} c_i c_j * M_{ij})
    # subject to sum c_i H_{2i}(0) = 0

    # The matrix M_{ij} involves the integral of |g(x)| * |g'(x)| or similar
    # For the Hermite basis, this can be computed analytically.

    # Key insight: For g(x) = e^{-x^2/2} P(x), we have:
    # ||g||_1 / ||g||_2 involves the L1 norm of the Hermite combination
    # The bound C4 = (||g||_1/||g||_2) * (||F(g)||_1/||F(g)||_2)

    # For even-degree Hermite polynomials, F(g) has a simple form.
    # The optimal bound is achieved when the quadratic form is minimized.

    # Let's construct the matrix M properly.
    # The relevant quantity is related to the expectation of |x| under the measure |g(x)|^2 dx

    # For g(x) = sum c_k e^{-x^2/2} H_{2k}(x):
    # ||g||_2^2 = sum c_k^2 * 2^{2k} * (2k)! * sqrt(pi)  (by orthogonality)

    # The L1 norm is harder, but the bound C4 can be bounded using:
    # ||g||_1 <= sqrt(||g||_2^2 * ||g||_inf / ||g||_2^2) ... no, that's not right.

    # Actually, the standard approach for these uncertainty bounds uses:
    # C4 >= (1/4) * (integral |x| |g(x)|^2 dx) / (integral |g(x)|^2 dx) * (similar for F(g))

    # But the exact bound for this specific class is known to be:
    # C4 = (1/4) * min (c^T M c) where M_{ij} = integral |x| e^{-x^2} H_{2i}(x) H_{2j}(x) dx / (norm factors)

    # Let me compute this matrix numerically for accuracy, then solve the constrained optimization.

    # Grid for numerical integration
    x = np.linspace(-15, 15, 20000)
    dx = x[1] - x[0]

    # Precompute the basis functions e^{-x^2/2} H_{2k}(x) for k=0,1,2,3
    from scipy.special import eval_hermite

    basis = []
    for deg in [0, 4, 8, 12]:
        H = eval_hermite(deg, x)
        basis.append(np.exp(-x ** 2 / 2) * H)

    # Compute the matrix M_{ij} = integral |x| * basis_i(x) * basis_j(x) dx
    # This gives us the "weighted" inner product that appears in the uncertainty bound
    M_num = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            M_num[i, j] = np.sum(np.abs(x) * basis[i] * basis[j]) * dx

    # The L2 norm squared matrix (should be diagonal by orthogonality)
    N = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            N[i, j] = np.sum(basis[i] * basis[j]) * dx

    # The uncertainty bound is:
    # C4 = (1/4) * min (c^T M_num c) / (c^T N c)
    # subject to c^T h0_vec = 0 where h0_vec = [H0(0), H4(0), H8(0), H12(0)]

    # This is a generalized eigenvalue problem: M_num c = lambda * N c
    # with the constraint c^T h0_vec = 0

    # For simplicity, let's normalize so that N is identity (work in the orthonormal basis)
    # First, compute the Cholesky decomposition of N
    try:
        L = np.linalg.cholesky(N)
    except np.linalg.LinAlgError:
        # If N is not positive definite, use a different approach
        L = np.eye(4)

    # Transform to orthonormal basis: c = L^{-1} d
    # Then the problem becomes: min (d^T L^{-T} M_num L^{-1} d) / (d^T d)
    # subject to d^T L^{-T} h0_vec = 0

    M_trans = np.linalg.solve(L.T, M_num)
    M_trans = np.linalg.solve(L, M_trans)
    M_trans = 0.5 * (M_trans + M_trans.T)  # Ensure symmetry

    h0_vec = np.array([h00, h04, h08, h012])
    h0_trans = np.linalg.solve(L.T, h0_vec)

    # Now we need to minimize d^T M_trans d subject to d^T h0_trans = 0 and ||d||_2 = 1
    # This is equivalent to finding the smallest eigenvalue of M_trans restricted to the subspace
    # orthogonal to h0_trans

    # Construct a basis for the subspace orthogonal to h0_trans
    # Use QR decomposition
    Q, R = np.linalg.qr(h0_trans.reshape(-1, 1))
    # Extend to a full basis
    # Create an identity matrix and project out the component along Q[:, 0]
    I = np.eye(4)
    P = I - np.outer(Q[:, 0], Q[:, 0])  # Projection onto orthogonal complement

    # The restricted matrix is P @ M_trans @ P, but we need to work in the 3D subspace
    # Find an orthonormal basis for the null space of h0_trans^T
    # Use SVD
    U, S, Vt = np.linalg.svd(h0_trans.reshape(1, -1))
    # The null space basis is the last 3 columns of Vt
    null_basis = Vt[1:, :].T  # 4x3 matrix whose columns span the null space

    # Restrict M_trans to this subspace
    M_restricted = null_basis.T @ M_trans @ null_basis

    # Find the smallest eigenvalue
    eigenvalues = np.linalg.eigvalsh(M_restricted)
    lambda_min = np.min(eigenvalues)

    # The bound is C4 = lambda_min / 4
    # But we need to find the coefficients c that achieve this

    # Get the eigenvector corresponding to lambda_min
    eigenvectors = np.linalg.eigh(M_restricted)
    idx_min = np.argmin(eigenvectors[0])
    d_opt = null_basis @ eigenvectors[1][:, idx_min]

    # Transform back to c space
    c_opt = np.linalg.solve(L, d_opt)

    # Extract the first three coefficients (c0, c4, c8)
    # c12 is determined by the constraint
    a0 = c_opt[0] / _norm_factor(0)
    a4 = c_opt[1] / _norm_factor(4)
    a8 = c_opt[2] / _norm_factor(8)

    # Verify the constraint is satisfied
    c12 = c_opt[3]
    constraint_check = c0_val = c_opt[0] * h00 + c_opt[1] * h04 + c_opt[2] * h08 + c_opt[3] * h012
    if abs(constraint_check) > 1e-10:
        # If the constraint is not satisfied, adjust c12
        c_opt[3] = -(c_opt[0] * h00 + c_opt[1] * h04 + c_opt[2] * h08) / h012
        a0 = c_opt[0] / _norm_factor(0)
        a4 = c_opt[1] / _norm_factor(4)
        a8 = c_opt[2] / _norm_factor(8)

    if time.time() > deadline - 5:
        return np.array([a0, a4, a8])

    # Numerical verification: compute the actual bound for these coefficients
    def compute_bound(c0, c4, c8, c12):
        g = c0 * basis[0] + c4 * basis[1] + c8 * basis[2] + c12 * basis[3]
        l2_sq = np.sum(g ** 2) * dx
        if l2_sq < 1e-12:
            return 1e10
        l1 = np.sum(np.abs(g)) * dx

        # For the Fourier transform, use the fact that F(e^{-x^2/2} H_n(x)) = i^n e^{-x^2/2} H_n(x) * sqrt(2*pi)
        # For even n, i^n = (-1)^(n/2)
        # So F(g)(xi) = sqrt(2*pi) * (c0 * psi_0(xi) - c4 * psi_4(xi) + c8 * psi_8(xi) - c12 * psi_12(xi))
        # But we need to be careful with the normalization

        # Actually, let's compute F(g) numerically using FFT for verification
        # But this is expensive, so let's use the analytical result

        # The key point is that for the optimal coefficients, the bound should match lambda_min / 4
        return (l1 / sqrt(l2_sq)) ** 2  # This is (||g||_1/||g||_2)^2, not the full product

    # Return the coefficients
    return np.array([a0, a4, a8])

# EVOLVE-BLOCK-END
