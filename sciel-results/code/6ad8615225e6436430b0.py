"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import factorial, sqrt, pi

# Basis indices
indices = [0, 4, 8, 12]
n_basis = len(indices)

# Precompute M_ii = <H_i, H_i> = sqrt(pi) * 2^i * i!
M_vals = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])

# Precompute H_n(0) for constraint g(0)=0
# H_0(0) = 1
# H_4(0) = -12
# H_8(0) = 1680
# H_12(0) = -103950
H_0 = np.array([1.0, -12.0, 1680.0, -103950.0])

# Constraint: c_0*1 + c_4*(-12) + c_8*1680 + c_12*(-103950) = 0
# c_12 = (c_0*1 + c_4*(-12) + c_8*1680) / 103950
def get_c12(c0, c4, c8):
    return (c0 * 1.0 + c4 * (-12.0) + c8 * 1680.0) / 103950.0

def build_coeff_vec(c0, c4, c8):
    c12 = get_c12(c0, c4, c8)
    return np.array([c0, c4, c8, c12])

# Matrix X (x^2) and D (-d^2/dx^2) in the Hermite basis
# x^2 H_n = 0.25 H_{n+2} + (n+0.5) H_n + 0.25 n(n-1) H_{n-2}
# -d^2/dx^2 H_n = -n(n-1) H_{n-2}

X_mat = np.zeros((n_basis, n_basis))
D_mat = np.zeros((n_basis, n_basis))

for i, n_i in enumerate(indices):
    for j, n_j in enumerate(indices):
        # X_ij = <H_i, x^2 H_j>
        # x^2 H_j expands to H_{j+2}, H_j, H_{j-2}
        # We need to check if any of these match n_i
        
        # Term 1: 0.25 H_{j+2}
        if j + 2 == n_i:
            X_mat[i, j] += 0.25 * M_vals[j] # Wait, integral <H_i, H_k> is M_i if i==k.
            # Actually, the expansion is in the basis.
            # <H_i, x^2 H_j> = sum_k coeff_k <H_i, H_k>
            # coeff for H_{j+2} is 0.25.
            # So if i == j+2, add 0.25 * <H_i, H_i> = 0.25 * M_i?
            # No, <H_i, H_k> = sqrt(pi) 2^i i! delta_ik.
            # So <H_i, H_{j+2}> is non-zero only if i == j+2.
            # In that case, the value is 0.25 * sqrt(pi) 2^i i!.
            # Note: M_vals[i] is sqrt(pi) 2^i i!.
            # So X_mat[i, j] += 0.25 * M_vals[i]
            pass # Handled below

        # Let's do it more simply by iterating over the expansion terms
        # Expansion of x^2 H_j:
        # a_{j+2} = 0.25
        # a_j = n_j + 0.5
        # a_{j-2} = 0.25 * n_j * (n_j - 1)
        
        terms = []
        if j + 2 in indices:
            terms.append((j + 2, 0.25))
        terms.append((j, n_j + 0.5))
        if j - 2 in indices:
            terms.append((j - 2, 0.25 * n_j * (n_j - 1)))
            
        for k, coeff in terms:
            if k == n_i:
                # <H_i, H_k> = M_i
                X_mat[i, j] += coeff * M_vals[i]

        # D_ij = <H_i, -d^2/dx^2 H_j>
        # -d^2/dx^2 H_j = -n_j(n_j-1) H_{j-2}
        if j - 2 in indices:
            coeff_d = -n_j * (n_j - 1)
            k_d = j - 2
            if k_d == n_i:
                D_mat[i, j] += coeff_d * M_vals[i]

def objective(c):
    c0, c4, c8 = c
    c_vec = build_coeff_vec(c0, c4, c8)
    
    norm_sq = np.dot(c_vec, M_vals * c_vec) # M is diagonal
    if norm_sq <= 1e-10:
        return 1e10
        
    x_var = np.dot(c_vec, X_mat @ c_vec) / norm_sq
    xi_var = np.dot(c_vec, D_mat @ c_vec) / norm_sq
    
    # Handle numerical errors
    if x_var < 0: x_var = 0
    if xi_var < 0: xi_var = 0
    
    return sqrt(x_var * xi_var)

def construct():
    """Find optimal coefficients for H0, H4, H8."""
    # Initial guess
    c0_init = 1.0
    c4_init = 0.1
    c8_init = 0.01
    
    res = minimize(objective, [c0_init, c4_init, c8_init], method='Nelder-Mead', 
                   options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
    
    if res.success:
        return res.x
    else:
        # Fallback to initial guess if optimization fails
        return np.array([c0_init, c4_init, c8_init])
# EVOLVE-BLOCK-END
