"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import factorial, sqrt, pi
import time

# Correct Hermite polynomial values at x=0 (physicists' convention)
# H_n(0) = (-1)^(n/2) * n! / (n/2)! for even n
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 665280.0

# M_ii = sqrt(pi) * 2^n * n!
M0 = sqrt(pi) * (2**0) * factorial(0)
M4 = sqrt(pi) * (2**4) * factorial(4)
M8 = sqrt(pi) * (2**8) * factorial(8)
M12 = sqrt(pi) * (2**12) * factorial(12)

# N_ii = n * M_ii
N0 = 0 * M0
N4 = 4 * M4
N8 = 8 * M8
N12 = 12 * M12

# c12 = -(c0*H0_0 + c4*H4_0 + c8*H8_0) / H12_0
# c12 = -(c0*1 + c4*12 + c8*1680) / 665280
d0 = -1.0 / 665280.0
d1 = -12.0 / 665280.0
d2 = -1680.0 / 665280.0

# Build P^T M P (3x3)
M_mat = np.array([
    [M0 + M12*d0*d0, M12*d0*d1, M12*d0*d2],
    [M12*d0*d1, M4 + M12*d1*d1, M12*d1*d2],
    [M12*d0*d2, M12*d1*d2, M8 + M12*d2*d2]
])

# Build P^T N P (3x3)
N_mat = np.array([
    [N0 + N12*d0*d0, N12*d0*d1, N12*d0*d2],
    [N12*d0*d1, N4 + N12*d1*d1, N12*d1*d2],
    [N12*d0*d2, N12*d1*d2, N8 + N12*d2*d2]
])

# Solve generalized eigenvalue problem: N_mat @ v = lambda * M_mat @ v
# The smallest eigenvalue is the minimum bound
eigenvalues, eigenvectors = np.linalg.eig(N_mat, M_mat)
# Filter for real eigenvalues
real_mask = np.abs(eigenvalues.imag) < 1e-10
eigenvalues_real = eigenvalues[real_mask].real
eigenvectors_real = eigenvectors[:, real_mask].real

idx_min = np.argmin(eigenvalues_real)
v_min = eigenvectors_real[:, idx_min]
# Normalize
v_min = v_min / np.linalg.norm(v_min)

def construct():
    """Find optimal coefficients for H0, H4, H8."""
    return v_min.copy()
# EVOLVE-BLOCK-END
