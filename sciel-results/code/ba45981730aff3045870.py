"""Global-search uncertainty bound via differential evolution + log-space polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from math import factorial, sqrt, pi

_DEADLINE = 185.0

# ---------- Hermite helpers ----------
def _hermite_coeffs(n):
    if n == 0:
        return {0: 1.0}
    if n == 1:
        return {1: 2.0}
    h_prev = {0: 1.0}
    h_curr = {1: 2.0}
    for m in range(2, n + 1):
        h_new = {}
        for p, c in h_curr.items():
            h_new[p + 1] = h_new.get(p + 1, 0.0) + 2.0 * c
        for p, c in h_prev.items():
            h_new[p] = h_new.get(p, 0.0) - 2.0 * (m - 1) * c
        h_new = {p: c for p, c in h_new.items() if abs(c) > 1e-15}
        h_prev, h_curr = h_curr, h_new
    return h_curr

def _int_x2m(m):
    if m == 0:
        return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val

# ---------- Precompute matrices (same quadratic form as before) ----------
DEGREES = [0, 4, 8, 12]
N_DEG = 4

H_POLYS = []
for d in DEGREES:
    hc = _hermite_coeffs(d)
    vec = np.zeros(d + 1)
    for p, c in hc.items():
        vec[p] = c
    H_POLYS.append(vec)

M0_mat = np.zeros((N_DEG, N_DEG))
for i, d in enumerate(DEGREES):
    M0_mat[i, i] = (2 ** d) * factorial(d)

M2_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        ci, cj = H_POLYS[i], H_POLYS[j]
        prod = np.convolve(ci, cj)
        prod_x2 = np.zeros(len(prod) + 2)
        prod_x2[2:] = prod
        val = 0.0
        for k in range(0, len(prod_x2), 2):
            val += prod_x2[k] * _int_x2m(k // 2)
        M2_mat[i, j] = val

Q_POLYS = []
for i, d in enumerate(DEGREES):
    q_vec = np.zeros(max(d + 1, 1))
    if d == 0:
        q_vec = np.zeros(2)
        q_vec[1] = -1.0
    else:
        h_minus = _hermite_coeffs(d - 1)
        h_plus = _hermite_coeffs(d + 1)
        q_vec = np.zeros(d + 2)
        for p, c in h_minus.items():
            q_vec[p] += d * c
        for p, c in h_plus.items():
            q_vec[p] -= 0.5 * c
    Q_POLYS.append(q_vec)

L_mat = np.zeros((N_DEG, N_DEG))
for i in range(N_DEG):
    for j in range(N_DEG):
        ci, cj = Q_POLYS[i], Q_POLYS[j]
        prod = np.convolve(ci, cj)
        val = 0.0
        for k in range(0, len(prod), 2):
            val += prod[k] * _int_x2m(k // 2)
        L_mat[i, j] = val

# B maps a3=[a0,a4,a8] -> a4=[a0,a4,a8,a12], with a12 fixed so g(0)=0
B = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0 / 27720.0, -12.0 / 27720.0, -2520.0 / 27720.0]
])

M0_eff = B.T @ M0_mat @ B
M2_eff = B.T @ M2_mat @ B
L_eff = B.T @ L_mat @ B


def _obj_2d(uv):
    """Objective function for u=a4, v=a8 with a0=1."""
    a3 = np.array([1.0, uv[0], uv[1]])
    M0 = a3 @ M0_eff @ a3
    M2 = a3 @ M2_eff @ a3
    L = a3 @ L_eff @ a3
    if M0 <= 1e-12 or M2 <= 1e-12 or L <= 1e-12:
        return 1e10, np.zeros(2)
    bound = np.sqrt(M2 * L) / M0
    # Gradient w.r.t u, v (indices 1, 2 of a3)
    dM0 = 2.0 * (M0_eff @ a3)
    dM2 = 2.0 * (M2_eff @ a3)
    dL = 2.0 * (L_eff @ a3)
    s = np.sqrt(M2 * L)
    # d bound / da3
    grad_a3 = (L * dM2 + M2 * dL) / (2.0 * M0 * s) - bound * dM0 / (M0 * M0)
    grad_uv = grad_a3[1:3]
    return bound, grad_uv

def _obj_2d_log(uv):
    b, _ = _obj_2d(uv)
    return np.log(b) if b < 1e9 else 1e6

def _obj_2d_log_grad(uv):
    b, g = _obj_2d(uv)
    if b >= 1e9:
        return 1e6, np.zeros(2)
    return np.log(b), g / b

# ---------- Construct: 2D multi-start optimization ----------
def construct():
    start_time = time.time()
    
    # Initial best
    best_uv = np.array([0.0, 0.0])
    best_b, _ = _obj_2d(best_uv)
    
    # Grid of starting points for L-BFGS-B
    # Based on previous best around a4~0.04, a8~0.0003 but we need to search wider
    # The previous best a3 was [-1.13, 0.04, 0.0003]. With a0=1, a4 = 0.04/-1.13 ~ -0.035, a8 = 0.0003/-1.13 ~ -0.00026
    # Let's search in a reasonable range around 0.
    u_vals = np.linspace(-1.0, 1.0, 20)
    v_vals = np.linspace(-0.1, 0.1, 20)
    
    starts = []
    for u in u_vals:
        for v in v_vals:
            starts.append(np.array([u, v]))
            
    # Add some specific points
    starts.append(np.array([-0.035, -0.00026]))
    starts.append(np.array([0.0, 0.0]))
    starts.append(np.array([-0.1, 0.0]))
    starts.append(np.array([0.1, 0.0]))
    
    for x0 in starts:
        if time.time() - start_time > _DEADLINE - 5.0:
            break
        try:
            res = minimize(_obj_2d_log_grad, x0, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < 1e6:
                b, _ = _obj_2d(res.x)
                if b < best_b:
                    best_b = b
                    best_uv = res.x.copy()
        except Exception:
            pass

    # Final polish
    if time.time() - start_time < _DEADLINE - 2.0:
        try:
            res = minimize(_obj_2d_log_grad, best_uv, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < 1e6:
                b, _ = _obj_2d(res.x)
                if b < best_b:
                    best_b = b
                    best_uv = res.x.copy()
        except Exception:
            pass

    # Return a3 with a0=1
    return np.array([1.0, best_uv[0], best_uv[1]])

# EVOLVE-BLOCK-END
