"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195

# Precompute constants for Hermite polynomials H0, H4, H8, H12
# We need integrals I_nk = int x^{2n} H_k(x)^2 e^{-x^2} dx
# And cross terms? 
# As derived, cross terms int x^{2n} H_i H_j e^{-x^2} are zero if |i-j| is not 0, 2.
# Our indices are 0, 4, 8, 12. Differences are 4, 8, 12.
# So cross terms are ZERO.
# Thus, the numerator and denominator are sums of squares.

# Let's compute the coefficients A_i, B_i for Vx = sum(z_i A_i) / sum(z_i B_i)
# z_i = a_i^2
# B_i = int H_i^2 e^{-x^2} dx = sqrt(pi) * 2^i * i!
# A_i = int x^2 H_i^2 e^{-x^2} dx

# We need to be careful with the Fourier side.
# For g(x) = H_n(x) e^{-x^2/2}, the Fourier transform is i^n H_n(x) e^{-x^2/2} (up to norm).
# So the structure is identical.
# V_x = sum(a_i^2 A_i) / sum(a_i^2 B_i)
# V_xi = sum(a_i^2 A_i) / sum(a_i^2 B_i)  <-- Is it the same?
# Yes, because the Hermite functions are eigenfunctions of the Fourier transform (up to phase).
# The variance of the position and momentum (Fourier freq) for a single Hermite function H_n is the same: n + 1/2.
# Let's verify for n=0. V=0.5.
# For n=1. H_1 = 2x. g = 2x e^{-x^2/2}.
# V_x = int x^2 (4x^2 e^{-x^2}) / int 4x^2 e^{-x^2} = int x^4 e^{-x^2} / int x^2 e^{-x^2} = (3 sqrt(pi) / 4) / (sqrt(pi) / 2) = 1.5.
# n+0.5 = 1.5. Correct.
# So A_i / B_i = i + 0.5.
# This means V_x = sum(a_i^2 (i+0.5) B_i) / sum(a_i^2 B_i).
# And V_xi is the SAME expression.
# So the bound is sqrt(V_x * V_xi) = V_x.
# We just need to minimize V_x!

# Wait, is V_xi exactly the same?
# The Fourier transform of H_n(x) e^{-x^2/2} is i^n H_n(\xi) e^{-\xi^2/2} (with standard convention).
# So |hat{g}|^2 has the same functional form as |g|^2, just with variable \xi.
# So yes, V_x = V_xi.
# The problem reduces to minimizing V_x = \frac{\sum_{i \in \{0,4,8,12\}} a_i^2 (i+0.5) B_i}{\sum_{i \in \{0,4,8,12\}} a_i^2 B_i}

# Let w_i = a_i^2 B_i.
# V_x = \frac{\sum w_i (i+0.5)}{\sum w_i}
# This is a weighted average of the values (0.5, 4.5, 8.5, 12.5).
# To minimize this, we want to put all weight on the smallest value, which is i=0 (value 0.5).
# This implies a_0 should be dominant, and a_4, a_8, a_12 should be 0.
# But we have the constraint g(0) = 0.
# g(0) = a_0 H_0(0) + a_4 H_4(0) + a_8 H_8(0) + a_12 H_12(0) = 0.
# H_0(0) = 1.
# H_4(0) = 3.
# H_8(0) = 105.
# H_12(0) = 945.
# Constraint: a_0 + 3a_4 + 105a_8 + 945a_12 = 0.
# We want to minimize \frac{\sum a_i^2 B_i (i+0.5)}{\sum a_i^2 B_i} subject to the linear constraint.

# Let's reformulate.
# Minimize F(a) = \frac{\sum a_i^2 C_i}{\sum a_i^2 B_i}
# where C_i = B_i (i+0.5).
# This is a Rayleigh quotient.
# The minimum of a Rayleigh quotient over a linear subspace (defined by the constraint) is the smallest eigenvalue of the generalized eigenvalue problem C v = \lambda B v, subject to the constraint?
# No, the constraint is linear, not homogeneous in the same way?
# The objective is homogeneous of degree 0.
# The constraint is homogeneous of degree 1.
# We can fix the scale.
# Let's use Lagrange multipliers.
# Minimize \frac{a^T C a}{a^T B a} subject to L a = 0.
# This is a non-standard problem because of the constraint.
# However, we can parameterize the null space of L.
# L = [1, 3, 105, 945].
# We need a vector a in the null space of L.
# Dimension of null space is 3.
# We can find a basis for the null space.
# Let a_12 = 1.
# a_0 = -3a_4 - 105a_8 - 945.
# Let u = a_4, v = a_8.
# a = [-3u - 105v - 945, u, v, 1].
# Substitute into F(a).
# This is a function of u, v.
# We can minimize this 2D function.

# Let's compute B_i and C_i.
# B_i = sqrt(pi) * 2^i * i!
# C_i = B_i * (i + 0.5)

B = np.zeros(4)
C = np.zeros(4)
indices = [0, 4, 8, 12]
for i, n in enumerate(indices):
    # B_n = sqrt(pi) * 2^n * n!
    # Use log to avoid overflow? n=12 is small.
    B[i] = np.sqrt(np.pi) * (2 ** n) * np.math.factorial(n)
    C[i] = B[i] * (n + 0.5)

# Define the objective function in terms of u, v
def objective(params):
    u, v = params
    a0 = -3*u - 105*v - 945
    a4 = u
    a8 = v
    a12 = 1
    
    a = np.array([a0, a4, a8, a12])
    
    num = np.sum(a**2 * C)
    den = np.sum(a**2 * B)
    
    if den == 0:
        return 1e18
    return num / den

# We need to find the global minimum of objective(u, v).
# The function goes to infinity as u, v -> infinity (since a0 ~ -3u, num ~ u^2 C0, den ~ u^2 B0, ratio -> C0/B0 = 0.5).
# Wait, if u -> infinity, a0 ~ -3u.
# num ~ 9u^2 C0 + u^2 C4.
# den ~ 9u^2 B0 + u^2 B4.
# Ratio -> (9C0 + C4) / (9B0 + B4).
# C0 = 0.5 B0. C4 = 4.5 B4.
# Ratio -> (4.5 B0 + 4.5 B4) / (9 B0 + B4) = 4.5 (B0+B4) / (9B0+B4).
# Since B4 >> B0 (2^4 4! = 16*24=384 vs 1), the ratio is approx 4.5 * B4 / B4 = 4.5.
# So the function is bounded and has a minimum.

# Strategy:
# 1. Grid search for u, v.
# 2. Polish with Nelder-Mead.

def construct():
    # Grid search
    # Based on seed: a0=-1.13, a4=0.04, a8=0.0003.
    # a12 is determined.
    # In our parameterization a12=1.
    # We need to find the scale.
    # The objective is homogeneous, so scale doesn't matter.
    # We just need the direction.
    # Let's search u, v in a reasonable range.
    # If a12=1, a0 is large (~ -945).
    # So u, v should be comparable to 100s or 1000s to balance?
    # Or if we pick a different basis vector for the null space?
    # Let's use the seed direction to guide the grid.
    # Seed: a0=-1.13, a4=0.04, a8=0.0003.
    # a12 = -(a0 + 3a4 + 105a8) / 945 = -(-1.13 + 0.12 + 0.0315) / 945 = 0.9785 / 945 ~ 0.001.
    # So a12 is small.
    # In my parameterization a12=1, the coefficients are scaled by 1000.
    # So u, v should be around 40, 0.3?
    # Let's search u in [-100, 100], v in [-10, 10].
    
    best_val = 1e18
    best_params = None
    
    # Coarse Grid
    u_range = np.linspace(-200, 200, 50)
    v_range = np.linspace(-20, 20, 50)
    
    for u in u_range:
        for v in v_range:
            val = objective([u, v])
            if val < best_val:
                best_val = val
                best_params = [u, v]
        
        if time.time() > _DEADLINE - 10:
            break
    
    if best_params is None:
        # Fallback to seed
        return np.array([-113 / 100, 1 / 25, 1 / 3240])

    # Polish with Nelder-Mead
    result = minimize(objective, best_params, method='Nelder-Mead', 
                      options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 10000})
    
    if result.success and result.fun < best_val:
        best_val = result.fun
        best_params = result.x
        
    u_opt, v_opt = best_params
    a0 = -3*u_opt - 105*v_opt - 945
    a4 = u_opt
    a8 = v_opt
    
    # The returned array must be [a0, a4, a8].
    # The evaluator will compute a12 to ensure g(0)=0.
    # We return the coefficients for H0, H4, H8.
    
    return np.array([a0, a4, a8])

# EVOLVE-BLOCK-END
