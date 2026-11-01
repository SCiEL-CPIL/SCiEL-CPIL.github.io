"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.special import eval_hermite

# Precompute constants
# H_n(0) for even n
H0_0 = float(eval_hermite(0, 0))
H4_0 = float(eval_hermite(4, 0))
H8_0 = float(eval_hermite(8, 0))
H12_0 = float(eval_hermite(12, 0))

# Normalization factors d_k = sqrt(2^k k!)
d0 = math.sqrt(2**0 * math.factorial(0))
d4 = math.sqrt(2**4 * math.factorial(4))
d8 = math.sqrt(2**8 * math.factorial(8))
d12 = math.sqrt(2**12 * math.factorial(12))

# Vector w such that sum(c_k * w_k) = 0 is the constraint g(0)=0
# c_k = a_k / d_k, so a_k = c_k d_k
# g(0) = sum a_k H_k(0) = sum c_k d_k H_k(0) = 0
w0 = d0 * H0_0
w4 = d4 * H4_0
w8 = d8 * H8_0
w12 = d12 * H12_0

w = np.array([w0, w4, w8, w12])

# Variance matrix V (diagonal)
# <x^2> = sum c_k^2 (k + 0.5) / sum c_k^2
# We minimize the numerator subject to sum c_k^2 = 1 and w^T c = 0
V = np.diag([0.5, 4.5, 8.5, 12.5])

def construct():
    """
    Returns three coefficients (a0, a4, a8) for the Hermite combination.
    """
    # 1. Find orthonormal basis Q for null space of w^T
    # SVD of w^T (1x4) gives U (1x1), S (1x1), Vh (1x4)
    # The last 3 rows of Vh are the null space basis?
    # Actually, SVD of a 1x4 matrix W: W = U S Vh.
    # Vh is 1x4. The vector v_1 = Vh[0] is the direction of w.
    # We need 3 orthogonal vectors to v_1.
    # We can use QR decomposition or just construct a basis manually.
    # Easier: use scipy.linalg.null_space if available, or just solve.
    
    # Let's use a simple method:
    # We want Q (4x3) such that Q^T Q = I_3 and w^T Q = 0.
    # We can start with the identity matrix and project out w.
    # Or use SVD of w (4x1).
    U, s, Vt = np.linalg.svd(w.reshape(4, 1))
    # Vt is 1x4. The vector Vt[0] is the unit vector in direction of w.
    # We need 3 vectors orthogonal to Vt[0].
    # We can get these from the remaining rows of a full orthogonal matrix.
    # Let's construct a full 4x4 orthogonal matrix O where O[:,0] = Vt[0].T
    # This is tricky. 
    # Alternative: Use the fact that we can pick any 3 vectors and orthonormalize them subject to the constraint.
    # Even simpler: The problem is small. We can just use the eigenvalue approach on the projected matrix.
    
    # Let v = w / ||w||
    norm_w = np.linalg.norm(w)
    if norm_w < 1e-15:
        # Degenerate case, shouldn't happen
        return np.array([1.0, 0.0, 0.0])
    
    v = w / norm_w
    
    # Construct a basis for the subspace orthogonal to v.
    # We can use the standard basis vectors and Gram-Schmidt, removing the component along v.
    # Basis vectors: e0, e1, e2, e3.
    # Project e_i onto subspace: p_i = e_i - v (v^T e_i)
    # Then orthonormalize p_i.
    
    e0 = np.array([1.0, 0.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 0.0, 1.0])
    
    # We need 3 independent vectors.
    # Let's pick e0, e1, e2 and project them.
    # If they become linearly dependent (e.g. if v is parallel to e2), we need to be careful.
    # Better: Use SVD of the 4x4 identity? No.
    
    # Let's use scipy.linalg.null_space which is robust.
    from scipy.linalg import null_space
    Q = null_space(w.reshape(1, 4)) # Returns 4x3 matrix
    
    # 2. Project V onto the null space
    M = Q.T @ V @ Q
    
    # 3. Find smallest eigenvalue and eigenvector of M
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argmin(eigvals)
    y_min = eigvecs[:, idx]
    
    # 4. Recover c
    c = Q @ y_min
    
    # 5. Convert to a
    a0 = c[0] * d0
    a4 = c[1] * d4
    a8 = c[2] * d8
    
    # Normalize to avoid numerical issues, though ratio is what matters
    # The score depends on the ratio, so scale doesn't matter.
    # But returning large numbers might cause issues? 
    # Let's normalize such that a0^2 + a4^2 + a8^2 = 1
    norm_a = math.sqrt(a0**2 + a4**2 + a8**2)
    if norm_a > 1e-15:
        a0 /= norm_a
        a4 /= norm_a
        a8 /= norm_a
    
    return np.array([a0, a4, a8])
# EVOLVE-BLOCK-END
