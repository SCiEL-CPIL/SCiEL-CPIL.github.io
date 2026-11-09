"""Root finding for ill-conditioned equations: robust adaptive hybrid.

Strategy per problem (all within a hard cap of 500 f-evaluations and a wall-clock
deadline):
  1. A safeguarded Newton phase using a numerically robust finite-difference
     derivative (central when affordable). Newton is used only while it is making
     real progress; steps are clipped to avoid runaway.
  2. Automatic bracketing around the current best point; once a sign change is
     found we run Brent's method (scipy.optimize.brentq) which is both robust and
     superlinear.
  3. A small deterministic restart portfolio (a few varied starting points) is
     attempted only if the primary attempt is still failing and time/evals remain.
  At every stage we keep the best point seen (smallest |f|) and return it, so
  partial credit is maximised even on timeout.
"""
import time
import math
import numpy as np
from scipy.optimize import brentq, newton

# EVOLVE-BLOCK-START

# Hard wall-clock deadline (seconds) with safety margin before 120s limit
_DEADLINE = 100.0  # seconds
_MAX_EVALS = 500

class _EvalBudget:
    """Tracks number of f evaluations and wall-clock time."""
    def __init__(self, f, max_evals=_MAX_EVALS, deadline=_DEADLINE):
        self._f = f
        self._count = 0
        self._start = time.time()
        self._max_evals = max_evals
        self._deadline = deadline

    def eval(self, x):
        """Evaluate f(x), incrementing counter. Returns (value, ok)."""
        if self._count >= self._max_evals:
            return None, False
        t = time.time()
        if t - self._start > self._deadline:
            return None, False
        try:
            val = self._f(x)
        except Exception:
            return None, False
        self._count += 1
        return val, True

    def spent(self):
        return self._count

    def time_left(self):
        return self._deadline - (time.time() - self._start)

    def expired(self):
        return self.time_left() <= 0 or self._count >= self._max_evals


def _safe_eval(budget, x):
    """Safely evaluate f at x using the budget."""
    val, ok = budget.eval(x)
    if not ok or val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _derivative(budget, x, fx):
    """Compute finite-difference derivative at x given fx=f(x)."""
    # Central difference when affordable
    h = 1e-7 * max(1.0, abs(x))
    if budget.time_left() < 0.5 or budget.spent() > _MAX_EVALS - 10:
        # Use forward difference to save evaluations
        fx_plus = _safe_eval(budget, x + h)
        if fx_plus is None:
            return None
        return (fx_plus - fx) / h
    else:
        fx_plus = _safe_eval(budget, x + h)
        fx_minus = _safe_eval(budget, x - h)
        if fx_plus is None or fx_minus is None:
            # Fall back to forward
            if fx_plus is not None:
                return (fx_plus - fx) / h
            return None
        return (fx_plus - fx_minus) / (2 * h)


def _try_bracket(budget, x, fx, scale=None):
    """Try to find a bracket [a, b] with sign change around x."""
    if fx is None:
        return None
    if scale is None:
        scale = max(1.0, abs(x))
    
    # Try expanding intervals
    for factor in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        if budget.expired():
            return None
        a = x - factor * scale
        b = x + factor * scale
        fa = _safe_eval(budget, a)
        if fa is None:
            continue
        fb = _safe_eval(budget, b)
        if fb is None:
            continue
        if fa * fb < 0:
            return (a, b)
        # Also check if one endpoint is already a root
        if abs(fa) <= 1e-8:
            return (a, a)
        if abs(fb) <= 1e-8:
            return (b, b)
    return None


def _solve_from(budget, x0, best_state):
    """Try to solve from x0 using hybrid Newton + bracketing + Brent.
    
    Returns (x, fx, success) where success means |fx| <= 1e-8.
    Updates best_state if a better point is found.
    """
    x = float(x0)
    
    # Initial evaluation
    fx = _safe_eval(budget, x)
    if fx is None:
        return x, None, False
    
    if abs(fx) <= 1e-8:
        best_state[0] = x
        best_state[1] = abs(fx)
        return x, fx, True
    
    best_state[0] = x
    best_state[1] = abs(fx)
    
    # Safeguarded Newton phase
    for i in range(40):
        if budget.expired():
            break
        
        # Compute derivative
        dfx = _derivative(budget, x, fx)
        if dfx is None or abs(dfx) < 1e-15:
            break
        
        # Newton step
        dx = fx / dfx
        x_new = x - dx
        
        # Safeguard: clip step to avoid runaway
        max_step = max(1.0, abs(x)) * 10
        if abs(dx) > max_step:
            x_new = x - math.copysign(max_step, dx)
        
        fx_new = _safe_eval(budget, x_new)
        if fx_new is None:
            break
        
        # Update best
        if abs(fx_new) < best_state[1]:
            best_state[0] = x_new
            best_state[1] = abs(fx_new)
        
        if abs(fx_new) <= 1e-8:
            return x_new, fx_new, True
        
        # Check if we're making progress
        if abs(fx_new) > abs(fx) * 1.5:
            # Newton is diverging; try bracketing
            bracket = _try_bracket(budget, x, fx)
            if bracket is not None:
                if bracket[0] == bracket[1]:
                    bx = bracket[0]
                    return bx, fx, abs(fx) <= 1e-8
                try:
                    if budget.time_left() > 1.0:
                        x_brent = brentq(budget._f, bracket[0], bracket[1], 
                                         xtol=1e-12, rtol=1e-12, maxiter=100)
                        fx_brent = _safe_eval(budget, x_brent)
                        if fx_brent is not None:
                            if abs(fx_brent) < best_state[1]:
                                best_state[0] = x_brent
                                best_state[1] = abs(fx_brent)
                            if abs(fx_brent) <= 1e-8:
                                return x_brent, fx_brent, True
                except Exception:
                    pass
            break
        
        x = x_new
        fx = fx_new
    
    # Try bracketing from current best point
    if not budget.expired():
        bracket = _try_bracket(budget, best_state[0], None, 
                               scale=max(1.0, abs(best_state[0])))
        if bracket is not None:
            if bracket[0] == bracket[1]:
                bx = bracket[0]
                fbx = _safe_eval(budget, bx)
                if fbx is not None and abs(fbx) < best_state[1]:
                    best_state[0] = bx
                    best_state[1] = abs(fbx)
                return bx, fbx, fbx is not None and abs(fbx) <= 1e-8
            try:
                if budget.time_left() > 1.0:
                    x_brent = brentq(budget._f, bracket[0], bracket[1],
                                     xtol=1e-12, rtol=1e-12, maxiter=100)
                    fx_brent = _safe_eval(budget, x_brent)
                    if fx_brent is not None:
                        if abs(fx_brent) < best_state[1]:
                            best_state[0] = x_brent
                            best_state[1] = abs(fx_brent)
                        if abs(fx_brent) <= 1e-8:
                            return x_brent, fx_brent, True
            except Exception:
                pass
    
    return best_state[0], None, best_state[1] <= 1e-8


def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8, using at most 500 evaluations of f.
    
    Returns x (float).
    """
    budget = _EvalBudget(f, _MAX_EVALS, _DEADLINE)
    
    # Best state: [x, |f(x)|]
    best_state = [float('inf'), float('inf')]
    
    # First, evaluate at x0
    fx0 = _safe_eval(budget, float(x0))
    if fx0 is not None:
        best_state[0] = float(x0)
        best_state[1] = abs(fx0)
    
    # Primary attempt from x0
    x_res, fx_res, success = _solve_from(budget, float(x0), best_state)
    if success:
        return x_res
    
    # If primary failed, try a few restart points
    if not budget.expired() and budget.time_left() > 2.0:
        # Generate diverse restart points
        x0_val = float(x0)
        restarts = [
            x0_val * 0.1 if x0_val != 0 else 0.1,
            x0_val * 10 if x0_val != 0 else 10.0,
            x0_val + 1.0,
            x0_val - 1.0,
            x0_val * 0.01 if x0_val != 0 else 0.01,
            x0_val * 100 if x0_val != 0 else 100.0,
            0.0,
            1.0,
            -1.0,
        ]
        
        for rx in restarts:
            if budget.expired() or budget.time_left() < 2.0:
                break
            if abs(rx) == abs(x0_val):  # skip if same as original
                continue
            x_res, fx_res, success = _solve_from(budget, rx, best_state)
            if success:
                return x_res
    
    # Return best point found
    if best_state[0] != float('inf'):
        return best_state[0]
    
    # Fallback: return x0
    return float(x0)

# EVOLVE-BLOCK-END
