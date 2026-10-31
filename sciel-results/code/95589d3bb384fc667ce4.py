"""
Uncertainty inequality optimization via generalized eigenvalue problem.

The uncertainty bound C4 = sqrt(I2 * J0) / I0 where I0, I2, J0 are quadratic
forms in the Hermite coefficients. The H12 constraint d = -(a+3b+105c)/10395
reduces the problem to 3 free variables. Both numerator and denominator are
quadratic, so the minimum of sqrt(I2*J0)/I0 occurs at a stationary point
equivalent to a generalized eigenvalue problem on the 3x3 moment matrices.
"""

import numpy as np
from scipy.optimize import minimize
import time

# Precompute Gauss-Hermite quadrature nodes and weights
# 30 points is exact for polynomials up to degree 59; H12^2 has degree 24
_N_GQ = 30
_X_GQ, _W_GQ = np.polynomial.hermite.hermgauss(_N_GQ)
_X2 = _X_GQ * _X_GQ

# Precompute Hermite polynomials H0, H4, H8, H12 at the nodes
_H0 = np.ones(_N_GQ)
_H4 = _X2**2 - 6 * _X2 + 3
_H8 = _X2**4 - 28 * _X2**3 + 210 * _X2**2 - 420 * _X2 + 105
_H12 = _X2**6 - 84 * _X2**5 + 1260 * _X2**4 - 12600 * _X2**3 + 55440 * _X2**2 - 100800 * _X2 + 10395

# Derivatives of H4, H8, H12 (H0' = 0)
_X3 = _X_GQ**3
_X5 = _X_GQ**5
_X7 = _X_GQ**7
_X9 = _X_GQ**9
_X11 = _X_GQ**11

_DH4 = 4 * _X3 - 12 * _X_GQ
_DH8 = 8 * _X7 - 168 * _X5 + 840 * _X3 - 840 * _X_GQ
_DH12 = 12 * _X11 - 840 * _X9 + 10080 * _X7 - 75600 * _X5 + 221760 * _X3 - 201600 * _X_GQ

# Quadrature weights
_W_I0 = _W_GQ
_W_I2 = _W_GQ * _X2

# Build the 3x3 moment matrices for the reduced (a,b,c) parameterization.
# P = a*H0 + b*H4 + c*H8 + d*H12, d = -(a+3b+105c)/10395
# So P = a*(H0 - H12/10395) + b*(H4 - 3H12/10395) + c*(H8 - 105H12/10395)
_H0r = _H0 - _H12 / 10395.0
_H4r = _H4 - 3.0 * _H12 / 10395.0
_H8r = _H8 - 105.0 * _H12 / 10395.0

# P' = b*H4' + c*H8' + d*H12'
# P' = b*(H4' - 3H12'/10395) + c*(H8' - 105H12'/10395)  [a contributes 0 to P']
_DH4r = _DH4 - 3.0 * _DH12 / 10395.0
_DH8r = _DH8 - 105.0 * _DH12 / 10395.0

# g' = P' - x*P
# g' = a*(0 - x*H0r) + b*(DH4r - x*H4r) + c*(DH8r - x*H8r)
_g0 = -_X_GQ * _H0r
_g4 = _DH4r - _X_GQ * _H4r
_g8 = _DH8r - _X_GQ * _H8r

# Build matrices A (for I0), B (for I2), C (for J0)
# I0 = sum w * P^2 = v^T A v where v = [a,b,c]
_A = np.zeros((3, 3))
_A[0, 0] = np.dot(_W_I0, _H0r * _H0r)
_A[0, 1] = np.dot(_W_I0, _H0r * _H4r)
_A[0, 2] = np.dot(_W_I0, _H0r * _H8r)
_A[1, 1] = np.dot(_W_I0, _H4r * _H4r)
_A[1, 2] = np.dot(_W_I0, _H4r * _H8r)
_A[2, 2] = np.dot(_W_I0, _H8r * _H8r)
_A = (A + A.T) / 2.0

_B = np.zeros((3, 3))
_B[0, 0] = np.dot(_W_I2, _H0r * _H0r)
_B[0, 1] = np.dot(_W_I2, _H0r * _H4r)
_B[0, 2] = np.dot(_W_I2, _H0r * _H8r)
_B[1, 1] = np.dot(_W_I2, _H4r * _H4r)
_B[1, 2] = np.dot(_W_I2, _H4r * _H8r)
_B[2, 2] = np.dot(_W_I2, _H8r * _H8r)
_B = (B + B.T) / 2.0

_C = np.zeros((3, 3))
_C[0, 0] = np.dot(_W_I0, _g0 * _g0)
_C[0, 1] = np.dot(_W_I0, _g0 * _g4)
_C[0, 2] = np.dot(_W_I0, _g0 * _g8)
_C[1, 1] = np.dot(_W_I0, _g4 * _g4)
_C[1, 2] = np.dot(_W_I0, _g4 * _g8)
_C[2, 2] = np.dot(_W_I0, _g8 * _g8)
_C = (C + C.T) / 2.0


def _compute_bound(v):
    """Compute the uncertainty bound for coefficient vector v = [a, b, c]."""
    I0 = v @ _A @ v
    I2 = v @ _B @ v
    J0 = v @ _C @ v
    if I0 < 1e-15 or J0 < 1e-15 or I2 < 1e-15:
        return 1e10
    return np.sqrt(I2 * J0) / I0


def _bound_and_grad(v):
    """Compute bound and gradient for v = [a, b, c]."""
    I0 = v @ _A @ v
    I2 = v @ _B @ v
    J0 = v @ _C @ v
    if I0 < 1e-15 or J0 < 1e-15 or I2 < 1e-15:
        return 1e10, np.zeros(3)
    
    score = np.sqrt(I2 * J0) / I0
    
    dI0 = 2.0 * (_A @ v)
    dI2 = 2.0 * (_B @ v)
    dJ0 = 2.0 * (_C @ v)
    
    grad = (dI2 * J0 + I2 * dJ0) / (2.0 * I2 * J0) - dI0 / I0
    return score, grad


def _eigenvalue_solve():
    """
    Solve the generalized eigenvalue problem for the optimal bound.
    
    The minimum of sqrt(I2*J0)/I0 over v != 0 occurs where the gradient vanishes.
    This is equivalent to finding the smallest eigenvalue of the system:
        (I2 * C + J0 * B) v = 2 * lambda * I0 * A v  (approximately)
    
    More precisely, we can use the fact that at the optimum:
        sqrt(I2*J0) = lambda * I0  where lambda is the bound
    
    The stationary condition gives us a 3x3 generalized eigenvalue problem.
    We parameterize: minimize (v^T B v)(v^T C v) / (v^T A v)^2
    Taking log and differentiating:
        B v / (v^T B v) + C v / (v^T C v) - 2 A v / (v^T A v) = 0
    
    At the optimum, let alpha = 1/(v^T B v), beta = 1/(v^T C v), gamma = 1/(v^T A v)
    Then: (alpha B + beta C - 2 gamma A) v = 0
    
    This is a nonlinear eigenvalue problem. We solve it iteratively.
    """
    # Use the substitution: let t = I0/sqrt(I2*J0) = 1/score
    # Maximize t = I0/sqrt(I2*J0)
    # Equivalent to: maximize (v^T A v)^2 / ((v^T B v)(v^T C v))
    
    # Iterative approach: fix the ratio, solve generalized eig, update
    best_v = None
    best_score = 1e10
    
    # Initialize with seed
    v0 = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Also try eigenvector of A^{-1}B and A^{-1}C
    for init in [v0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]:
        v = init.copy()
        for _ in range(200):
            I0 = v @ _A @ v
            I2 = v @ _B @ v
            J0 = v @ _C @ v
            if I0 < 1e-15 or I2 < 1e-15 or J0 < 1e-15:
                break
            alpha = 1.0 / I2
            beta = 1.0 / J0
            gamma = 1.0 / I0
            M = alpha * _B + beta * _C - 2.0 * gamma * _A
            # Find null vector of M
            try:
                w, V = np.linalg.eigh(M)
                # Pick the eigenvector with smallest |eigenvalue|
                idx = np.argmin(np.abs(w))
                v_new = V[:, idx]
                # Ensure consistent sign
                if np.dot(v_new, v) < 0:
                    v_new = -v_new
                if np.linalg.norm(v_new - v) < 1e-12:
                    v = v_new
                    break
                v = v_new
            except np.linalg.LinAlgError:
                break
        
        score = _compute_bound(v)
        if score < best_score:
            best_score = score
            best_v = v.copy()
    
    return best_v, best_score


def construct():
    start_time = time.time()
    deadline = 205.0  # Safety margin before 210s kill
    
    best_score = 1e10
    best_params_3d = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # 1. Eigenvalue-based solve
    try:
        v_eig, score_eig = _eigenvalue_solve()
        if score_eig < best_score:
            best_score = score_eig
            best_params_3d = v_eig
    except Exception:
        pass
    
    # 2. Grid search in 3D (coarse)
    if time.time() < deadline - 80:
        # Scale-invariant: fix a=1, search b,c
        u_range = np.linspace(-3.0, 3.0, 31)
        v_range = np.linspace(-3.0, 3.0, 31)
        candidates = []
        
        for u in u_range:
            for vv in v_range:
                if time.time() > deadline - 60:
                    break
                try:
                    v = np.array([1.0, u, vv])
                    score = _compute_bound(v)
                    if score < best_score:
                        best_score = score
                        best_params_3d = v
                    candidates.append((score, u, vv))
                except Exception:
                    pass
            if time.time() > deadline - 60:
                break
        
        # Refine top candidates with L-BFGS-B
        if time.time() < deadline - 40:
            candidates.sort(key=lambda x: x[0])
            for _, u, vv in candidates[:15]:
                try:
                    result = minimize(
                        lambda x: _bound_and_grad(x)[0],
                        np.array([1.0, u, vv]),
                        jac=lambda x: _bound_and_grad(x)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13}
                    )
                    if result.fun < best_score:
                        best_score = result.fun
                        best_params_3d = result.x
                except Exception:
                    pass
    
    # 3. Also search with a=0 sector
    if time.time() < deadline - 30:
        v_range_a0 = np.linspace(-3.0, 3.0, 31)
        candidates_a0 = []
        for vv in v_range_a0:
            if time.time() > deadline - 25:
                break
            try:
                v = np.array([0.0, 1.0, vv])
                score = _compute_bound(v)
                if score < best_score:
                    best_score = score
                    best_params_3d = v
                candidates_a0.append((score, vv))
            except Exception:
                pass
        
        if time.time() < deadline - 15:
            candidates_a0.sort(key=lambda x: x[0])
            for _, vv in candidates_a0[:10]:
                try:
                    result = minimize(
                        lambda x: _bound_and_grad(x)[0],
                        np.array([0.0, 1.0, vv]),
                        jac=lambda x: _bound_and_grad(x)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-13}
                    )
                    if result.fun < best_score:
                        best_score = result.fun
                        best_params_3d = result.x
                except Exception:
                    pass
    
    # 4. Final high-precision polish on best point
    if time.time() < deadline - 5:
        try:
            result = minimize(
                lambda x: _bound_and_grad(x)[0],
                best_params_3d,
                jac=lambda x: _bound_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-15}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params_3d = result.x
        except Exception:
            pass
    
    # 5. Run eigenvalue solve again from the polished point as initial guess
    if time.time() < deadline - 3:
        try:
            # Modify _eigenvalue_solve to accept initial guess
            v_init = best_params_3d.copy()
            v = v_init.copy()
            for _ in range(300):
                I0 = v @ _A @ v
                I2 = v @ _B @ v
                J0 = v @ _C @ v
                if I0 < 1e-15 or I2 < 1e-15 or J0 < 1e-15:
                    break
                alpha = 1.0 / I2
                beta = 1.0 / J0
                gamma = 1.0 / I0
                M = alpha * _B + beta * _C - 2.0 * gamma * _A
                try:
                    w, V = np.linalg.eigh(M)
                    idx = np.argmin(np.abs(w))
                    v_new = V[:, idx]
                    if np.dot(v_new, v) < 0:
                        v_new = -v_new
                    if np.linalg.norm(v_new - v) < 1e-14:
                        v = v_new
                        break
                    v = v_new
                except np.linalg.LinAlgError:
                    break
            score = _compute_bound(v)
            if score < best_score:
                best_score = score
                best_params_3d = v
        except Exception:
            pass
    
    # Ensure we don't return worse than seed
    seed_score = _compute_bound(np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0]))
    if best_score > seed_score:
        best_params_3d = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    return best_params_3d
