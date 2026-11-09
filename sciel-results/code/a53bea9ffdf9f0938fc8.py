"""Root finder for 3 fixed two-dimensional systems using Levenberg-Marquardt with rank-1 Jacobian updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        arr = np.asarray(arr)
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _norm_F(F, x):
    try:
        Fx = F(x)
        if not _is_finite(Fx):
            return np.inf
        return np.linalg.norm(Fx)
    except Exception:
        return np.inf


def _levenberg_marquardt(F, x0, max_iter=200):
    """
    Levenberg-Marquardt method with rank-1 (Broyden-like) Jacobian updates.
    Uses adaptive damping lambda based on reduction in ||F||^2.
    Only 1 function evaluation per iteration after initial Jacobian setup.
    """
    x = np.array(x0, dtype=float, copy=True)
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf
    
    def _call_F(x_val):
        nonlocal call_count
        if time.time() > _START_TIME + _DEADLINE:
            return None
        if call_count >= 490:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result
    
    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx
    
    # Compute initial Jacobian via forward differences (2 more calls)
    h = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _call_F(x + e)
        if Fx_plus is None:
            Fx_plus = Fx + h * np.ones(2)
        J[:, j] = (Fx_plus - Fx) / h
    
    # Ensure J is reasonably conditioned
    det_J = np.linalg.det(J)
    if abs(det_J) < 1e-14:
        J += np.eye(2) * 1e-8
    
    # Adaptive LM parameter
    lambda_val = 1e-3
    mu = 1e-3
    
    for it in range(max_iter):
        if not _time_ok():
            break
        if call_count >= 485:
            break
        
        # Solve (J^T J + lambda * I) dx = -J^T F
        try:
            JtJ = J.T @ J
            JtF = J.T @ Fx
            A = JtJ + lambda_val * np.eye(2)
            dx = np.linalg.solve(A, -JtF)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            try:
                A = JtJ + lambda_val * np.eye(2)
                dx = np.linalg.pinv(A) @ (-JtF)
            except Exception:
                break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Limit step size
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 10.0:
            dx = dx / norm_dx * 10.0
        
        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            dx = dx * 0.1
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break
        
        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break
        
        norm_Fx_new = np.linalg.norm(Fx_new)
        norm_sq_old = np.dot(Fx, Fx)
        norm_sq_new = np.dot(Fx_new, Fx_new)
        
        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new
        
        if norm_Fx_new <= 1e-8:
            return x_new
        
        # Adaptive lambda update
        gain_ratio = (norm_sq_old - norm_sq_new) / (lambda_val * np.dot(dx, dx) + 1e-30)
        
        if gain_ratio > 0.5:
            # Good step: reduce lambda
            lambda_val = max(lambda_val / 10.0, 1e-12)
            mu = max(mu / 2.0, 1e-6)
        elif gain_ratio > 0:
            # Acceptable step: keep lambda
            pass
        else:
            # Bad step: increase lambda
            lambda_val = min(lambda_val * 10.0, 1e6)
            mu = min(mu * 2.0, 1e4)
        
        # Accept step and update Jacobian with Broyden rank-1
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        if s_norm_sq > 1e-16:
            J = J + np.outer(y - J @ s, s) / s_norm_sq
        
        x = x_new
        Fx = Fx_new
    
    return best_x


def _newton_fd(F, x0, max_iter=40):
    """Newton's method with finite-difference Jacobian as fallback."""
    x = np.array(x0, dtype=float)
    calls = 0
    
    for it in range(max_iter):
        if not _time_ok():
            break
        
        try:
            Fx = F(x)
        except Exception:
            return x
        calls += 1
        if not _is_finite(Fx):
            return x
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            return x
        
        # Finite difference Jacobian
        J = np.zeros((2, 2))
        eps = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            try:
                Fx_e = F(x + e)
            except Exception:
                Fx_e = Fx + eps * np.ones(2)
            calls += 1
            if not _is_finite(Fx_e):
                Fx_e = Fx + eps * np.ones(2)
            J[:, j] = (Fx_e - Fx) / eps
        
        # Solve for Newton step
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                J[0, 0] += 1e-10
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
        
        # Damped Newton with backtracking
        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * dx
            try:
                Fx_new = F(x_new)
            except Exception:
                alpha *= 0.5
                continue
            calls += 1
            if _is_finite(Fx_new):
                if np.linalg.norm(Fx_new) < (1 - 1e-4 * alpha) * norm_Fx:
                    break
            alpha *= 0.5
        
        x = x + alpha * dx
    
    return x


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Primary: Levenberg-Marquardt with rank-1 Jacobian updates.
    Fallback: damped Newton with FD Jacobian.
    """
    x0 = np.array(x0, dtype=float)
    
    # Try Levenberg-Marquardt first
    x_lm = _levenberg_marquardt(F, x0)
    if _is_finite(x_lm):
        try:
            Fx_check = F(x_lm)
        except Exception:
            Fx_check = None
        if Fx_check is not None and _is_finite(Fx_check) and np.linalg.norm(Fx_check) <= 1e-8:
            return x_lm
    
    # If LM didn't converge, try Newton FD
    x_newton = _newton_fd(F, x0)
    if _is_finite(x_newton):
        try:
            Fx_check = F(x_newton)
        except Exception:
            Fx_check = None
        if Fx_check is not None and _is_finite(Fx_check) and np.linalg.norm(Fx_check) <= 1e-8:
            return x_newton
    
    # Return the best result found
    candidates = []
    for x_cand in [x_lm, x_newton]:
        if _is_finite(x_cand):
            try:
                Fx_c = F(x_cand)
            except Exception:
                continue
            if _is_finite(Fx_c):
                candidates.append((np.linalg.norm(Fx_c), x_cand))
    
    if candidates:
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]
    
    return x0
# EVOLVE-BLOCK-END
