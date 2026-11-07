"""Global multi-start Levenberg-Marquardt root finder with Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root

_START_TIME = None
_DEADLINE_MARGIN = 10  # seconds spare before 150s limit

def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0.0

def _time_left():
    return 150.0 - _elapsed() - _DEADLINE_MARGIN

def _safe_F(F, x):
    """Evaluate F(x), return (Fx, norm_Fx). Returns (None, inf) on failure."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return None, np.inf
        n = float(np.linalg.norm(Fx))
        return Fx, n
    except Exception:
        return None, np.inf

def _newton_polish(F, x, max_iter=10, tol=1e-12):
    """
    Final polish using damped Newton's method with Armijo backtracking.
    This exploits quadratic convergence when close to the root.
    Returns (x, nF, n_calls).
    """
    x = np.array(x, dtype=float)
    calls = 0
    Fx, nF = _safe_F(F, x)
    calls += 1
    if Fx is None:
        return x, np.inf, calls
    if nF <= tol:
        return x, nF, calls

    # Initial Jacobian estimate
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_p, _ = _safe_F(F, x + e)
        calls += 1
        if Fx_p is None:
            return x, nF, calls
        J[:, j] = (Fx_p - Fx) / h

    for _ in range(max_iter):
        if _time_left() < 1.0:
            break
        if nF <= tol:
            break

        # Solve J dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Regularize if singular
            try:
                dx = np.linalg.solve(J + 1e-8 * np.eye(2), -Fx)
            except np.linalg.LinAlgError:
                break

        if not np.all(np.isfinite(dx)):
            break

        # Armijo backtracking line search
        alpha = 1.0
        c1 = 1e-4
        slope = float(np.dot(J.T @ Fx, dx))  # -||F||^2 approx gradient direction
        # Actually, for minimization of 0.5||F||^2, grad = J^T F, dir = -J^-1 F
        # Decrease condition: f(x + alpha*dx) <= f(x) + c1 * alpha * grad^T * dx
        # f(x) = 0.5 * nF^2
        # grad^T * dx = (J^T F)^T (-J^-1 F) = -F^T J J^-1 F = -||F||^2
        # So condition: 0.5 * ||F(x+alpha*dx)||^2 <= 0.5 * nF^2 - c1 * alpha * ||F||^2
        # ||F(x+alpha*dx)||^2 <= nF^2 - 2*c1*alpha*||F||^2
        
        success = False
        for _ in range(20):
            x_new = x + alpha * dx
            Fx_new, nF_new = _safe_F(F, x_new)
            calls += 1
            if Fx_new is None:
                alpha *= 0.5
                continue
            
            if nF_new <= tol:
                x = x_new
                Fx = Fx_new
                nF = nF_new
                success = True
                break
            
            # Check Armijo condition
            if nF_new**2 <= nF**2 - 2 * c1 * alpha * nF**2:
                x = x_new
                Fx = Fx_new
                nF = nF_new
                success = True
                break
            
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        if not success:
            break

        # Update Jacobian via finite differences (simple, robust for 2D)
        # Re-use x as base
        Fx_base, _ = _safe_F(F, x)
        calls += 1
        if Fx_base is None:
            break
        J_new = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p, _ = _safe_F(F, x + e)
            calls += 1
            if Fx_p is None:
                return x, nF, calls
            J_new[:, j] = (Fx_p - Fx_base) / h
        J = J_new

    return x, nF, calls

def find_root_system(F, x0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    x0 = np.asarray(x0, dtype=float).flatten()
    if x0.shape != (2,):
        x0 = np.array([0.0, 0.0])

    best_x = x0.copy()
    best_norm = np.inf
    total_calls = 0

    def _update_best(x_candidate, n_calls):
        nonlocal best_x, best_norm, total_calls
        total_calls += n_calls
        if x_candidate is None:
            return
        try:
            x_candidate = np.asarray(x_candidate, dtype=float).flatten()
            if x_candidate.shape != (2,) or not np.all(np.isfinite(x_candidate)):
                return
            Fx, n = _safe_F(F, x_candidate)
            total_calls += 1  # for the evaluation in _safe_F
            if n < best_norm:
                best_norm = n
                best_x = x_candidate
        except Exception:
            pass

    # Strategy 1: Multi-start Levenberg-Marquardt
    # Generate a set of starting points: x0 and a grid of offsets
    rng = np.random.default_rng(42)
    start_points = [x0.copy()]
    
    # Add deterministic offsets to cover different basins
    radii = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    
    for r in radii:
        for a in angles:
            offset = r * np.array([np.cos(a), np.sin(a)])
            start_points.append(x0 + offset)
    
    # Sort by distance from x0 to try closest first, but ensure diversity
    # Actually, for global convergence, we might want to try far ones if close ones fail.
    # But LM is robust. Let's just try them in a mixed order.
    # Prioritize x0, then small radii.
    
    # Try each start point with LM
    for sp in start_points:
        if _time_left() < 2.0:
            break
        
        # Check if we've already tried this or a very close point
        # Simple check: if distance to best_x is small, skip? No, just try.
        
        try:
            # Use scipy root with hybr (Levenberg-Marquardt)
            # tol=1e-12, maxiter=100
            res = root(F, sp, method='hybr', tol=1e-12, maxiter=100)
            x_res = res.x
            n_calls_res = res.nfev
            _update_best(x_res, n_calls_res)
            
            if best_norm <= 1e-8:
                # Success! Do a final polish to ensure high precision and minimize calls if possible
                # Actually, if we are already < 1e-8, we can return.
                # But polish might reduce calls? No, polish adds calls.
                # The score depends on mean_calls. Fewer calls is better.
                # If LM converged in 20 calls, polish might add 10.
                # Let's only polish if we are close but not quite there, or if we want to be safe.
                # Given the tight tolerance 1e-8, LM usually hits it.
                # Let's polish only if best_norm > 1e-10 to ensure we are solidly below 1e-8.
                if best_norm > 1e-10:
                    x_pol, n_pol, c_pol = _newton_polish(F, best_x, max_iter=5, tol=1e-12)
                    _update_best(x_pol, c_pol)
                return best_x
        except Exception:
            pass

    # If we haven't returned, we might have found a good but not perfect root.
    # Or LM failed for all.
    # Try Newton polish on the best candidate found so far.
    if best_norm < np.inf and best_norm > 1e-12:
        if _time_left() > 2.0:
            x_pol, n_pol, c_pol = _newton_polish(F, best_x, max_iter=10, tol=1e-12)
            _update_best(x_pol, c_pol)

    return best_x

# EVOLVE-BLOCK-END
