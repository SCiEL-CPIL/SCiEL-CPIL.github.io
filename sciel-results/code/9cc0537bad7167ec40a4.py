import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm, inv

# Deadline guard
_DEADLINE = time.time() + 200.0

def _time_left():
    return _DEADLINE - time.time()

def _hermite_coeffs(n):
    """Return coefficients c_k for H_n(x) = sum_k c_k x^k (physicists' convention)."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    c_nm1 = np.array([1.0])  # H_0
    c_n = np.array([0.0, 2.0])  # H_1
    for k in range(1, n):
        c_np1 = np.zeros(len(c_n) + 1)
        c_np1[:-1] += 2.0 * c_n
        c_np1[:-2] -= 2.0 * k * c_nm1
        c_nm1, c_n = c_n, c_np1
    return c_n

def _poly_eval(coeffs, x):
    """Evaluate polynomial given coefficient array [c0, c1, ..., cN] at x."""
    result = np.zeros_like(x, dtype=float)
    for i, c in enumerate(coeffs):
        result += c * (x ** i)
    return result

# Precompute Hermite polynomials and their values at quadrature nodes
# We use Gauss-Hermite quadrature for integrals of form int f(x) exp(-x^2) dx
# Our integrals are of form int P(x)^2 exp(-x^2/2) dx.
# Let x = sqrt(2) u. dx = sqrt(2) du. exp(-x^2/2) = exp(-u^2).
# int P(sqrt(2)u)^2 exp(-u^2) sqrt(2) du.
# We can use hermgauss to get nodes u_i and weights w_i.
# Then the integral is sum w_i P(sqrt(2)u_i)^2 * sqrt(2).

_NQ = 200
_u_nodes, _w_nodes = np.polynomial.hermite.hermgauss(_NQ)
_x_nodes = np.sqrt(2.0) * _u_nodes
_w_nodes = np.sqrt(2.0) * _w_nodes

# Hermite polynomials H_0, H_4, H_8, H_12
_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

# Derivatives H'_n = 2n H_{n-1}
_H0_prime = np.zeros_like(_x_nodes)
_H4_prime = 2 * 4 * _poly_eval(_hermite_coeffs(3), _x_nodes)
_H8_prime = 2 * 8 * _poly_eval(_hermite_coeffs(7), _x_nodes)
_H12_prime = 2 * 12 * _poly_eval(_hermite_coeffs(11), _x_nodes)

# Values at nodes
_H0_vals = _poly_eval(_H0, _x_nodes)
_H4_vals = _poly_eval(_H4, _x_nodes)
_H8_vals = _poly_eval(_H8, _x_nodes)
_H12_vals = _poly_eval(_H12, _x_nodes)

# Values at 0 for constraint
_H0_0 = _poly_eval(_H0, np.array([0.0]))[0]
_H4_0 = _poly_eval(_H4, np.array([0.0]))[0]
_H8_0 = _poly_eval(_H8, np.array([0.0]))[0]
_H12_0 = _poly_eval(_H12, np.array([0.0]))[0]

# Basis functions in terms of c0, c4, c8
# P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
# c12 = -(c0 H0_0 + c4 H4_0 + c8 H8_0) / H12_0

# We want to compute matrices M_0, M_x, M_p such that:
# Q_0(c) = c^T M_0 c
# Q_x(c) = c^T M_x c
# Q_p(c) = c^T M_p c
# where c = [c0, c4, c8]

# Let's define the basis vector B_i(x) for i=0,1,2 corresponding to c0, c4, c8
# B_0(x) = H0(x) - (H0_0/H12_0) H12(x)
# B_1(x) = H4(x) - (H4_0/H12_0) H12(x)
# B_2(x) = H8(x) - (H8_0/H12_0) H12(x)

B0_vals = _H0_vals - (_H0_0 / _H12_0) * _H12_vals
B1_vals = _H4_vals - (_H4_0 / _H12_0) * _H12_vals
B2_vals = _H8_vals - (_H8_0 / _H12_0) * _H12_vals

B_prime0_vals = _H0_prime - (_H0_0 / _H12_0) * _H12_prime
B_prime1_vals = _H4_prime - (_H4_0 / _H12_0) * _H12_prime
B_prime2_vals = _H8_prime - (_H8_0 / _H12_0) * _H12_prime

# P(x) = c0 B0(x) + c4 B1(x) + c8 B2(x)
# P'(x) = c0 B0'(x) + c4 B1'(x) + c8 B2'(x)

# Q_0 = sum w_i (c0 B0_i + c4 B1_i + c8 B2_i)^2
# M_0[j,k] = sum w_i B_j_i B_k_i

B_matrix = np.vstack([B0_vals, B1_vals, B2_vals]) # 3 x NQ
B_prime_matrix = np.vstack([B_prime0_vals, B_prime1_vals, B_prime2_vals])

# M_0
M_0 = B_matrix @ (np.diag(_w_nodes) @ B_matrix.T)
# M_x
x2_weighted_B = B_matrix * _x_nodes**2
M_x = B_matrix @ (np.diag(_w_nodes * _x_nodes**2) @ B_matrix.T)
# M_p
# g' inner = P' - x/2 P
# (P' - x/2 P)^2 = (P')^2 - x P P' + x^2/4 P^2
# Let's just compute the vector V_i = B'_i - x_i/2 B_i for each basis function?
# No, P' and P are linear combos.
# Let D_i = B'_i - x_i/2 B_i.
# Then (P' - x/2 P) = c0 D0 + c4 D1 + c8 D2.
# So M_p[j,k] = sum w_i D_j_i D_k_i

D_matrix = B_prime_matrix - 0.5 * _x_nodes[None, :] * B_matrix
M_p = D_matrix @ (np.diag(_w_nodes) @ D_matrix.T)

# Symmetrize to remove numerical noise
M_0 = (M_0 + M_0.T) / 2
M_x = (M_x + M_x.T) / 2
M_p = (M_p + M_p.T) / 2

# We want to minimize J(c) = (c^T M_x c * c^T M_p c) / (c^T M_0 c)^2
# Let c = M_0^{-1/2} u, with u^T u = 1.
# Then c^T M_0 c = u^T u = 1.
# J(u) = (u^T A u) (u^T B u), where A = M_0^{-1/2} M_x M_0^{-1/2}, B = M_0^{-1/2} M_p M_0^{-1/2}.

# Compute M_0^{-1/2}
# Use eigen decomposition
eigvals, eigvecs = np.linalg.eigh(M_0)
# M_0^{-1/2} = eigvecs diag(eigvals^{-1/2}) eigvecs.T
M_0_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

A = M_0_inv_sqrt @ M_x @ M_0_inv_sqrt
B = M_0_inv_sqrt @ M_p @ M_0_inv_sqrt
A = (A + A.T) / 2
B = (B + B.T) / 2

def _objective_sphere(u):
    """Objective function on the unit sphere."""
    ua = u @ A @ u
    ub = u @ B @ u
    return ua * ub

def _gradient_sphere(u):
    """Gradient of the objective function on the unit sphere."""
    ua = u @ A @ u
    ub = u @ B @ u
    grad = 2 * ua * (B @ u) + 2 * ub * (A @ u)
    # Project gradient onto tangent space of sphere: grad - (grad . u) u
    grad_tangent = grad - np.dot(grad, u) * u
    return grad_tangent

def construct():
    best_val = np.inf
    best_u = None
    
    # Grid search on the sphere
    # Theta in [0, pi], Phi in [0, 2*pi)
    n_theta = 200
    n_phi = 400
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    best_grid_val = np.inf
    best_grid_u = None
    
    # We can vectorize the evaluation
    # u = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
    # This might be too large to vectorize easily with matrix mult, so loop or chunk.
    # 200*400 = 80,000 points. Loop is fine.
    
    for i, th in enumerate(thetas):
        if _time_left() < 10.0:
            break
        st, ct = np.sin(th), np.cos(th)
        for j, ph in enumerate(phis):
            cp, sp = np.cos(ph), np.sin(ph)
            u = np.array([st * cp, st * sp, ct])
            val = _objective_sphere(u)
            if val < best_grid_val:
                best_grid_val = val
                best_grid_u = u.copy()
    
    if best_grid_u is not None:
        best_val = best_grid_val
        best_u = best_grid_u

    # Local optimization from the best grid point and some random points
    if best_u is not None:
        # Use a penalty method or just optimize on the sphere using spherical coords?
        # scipy.optimize.minimize doesn't handle sphere constraint directly.
        # We can optimize in 3D with a penalty term lambda * (u^T u - 1)^2
        # Or use the tangent space gradient and a custom step.
        # Simpler: use minimize with method 'BFGS' and constraint?
        # Let's use the penalty method.
        
        def _penalty_obj(u):
            return _objective_sphere(u) + 1000.0 * (np.dot(u, u) - 1.0)**2
        
        def _penalty_grad(u):
            return _gradient_sphere(u) + 4000.0 * (np.dot(u, u) - 1.0) * u
        
        start_points = [best_u]
        # Add some random starts
        rng = np.random.RandomState(42)
        for _ in range(50):
            v = rng.normal(0, 1, 3)
            v = v / np.linalg.norm(v)
            start_points.append(v)
        
        for start in start_points:
            if _time_left() < 5.0:
                break
            try:
                res = minimize(
                    _penalty_obj,
                    start,
                    jac=_penalty_grad,
                    method='BFGS',
                    options={'maxiter': 500, 'gtol': 1e-10}
                )
                if res.fun < best_val:
                    # Normalize u to be exactly on the sphere
                    u_opt = res.x / np.linalg.norm(res.x)
                    val_opt = _objective_sphere(u_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_u = u_opt
            except Exception:
                pass

    # Convert u back to c
    # c = M_0^{-1/2} u
    if best_u is not None:
        c_opt = M_0_inv_sqrt @ best_u
        # c = [c0, c4, c8]
        return c_opt
    else:
        # Fallback
        return np.array([0.0, 0.0, 0.0])
