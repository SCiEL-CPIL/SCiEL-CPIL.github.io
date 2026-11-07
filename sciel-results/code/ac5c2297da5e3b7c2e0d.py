"""Convex optimization approach for Erdos minimum overlap."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _correlation_lags(h):
    """Compute all lags of correlate(h, 1-h, mode='full')."""
    return np.correlate(h, 1 - h, mode="full")


def _objective_and_grad(t, h, lags):
    """Objective t + penalty for violations of C_k <= t, plus sum constraint."""
    # We minimize t subject to lags <= t, sum(h) = N/2, 0 <= h <= 1
    # For SLSQP we use inequality constraints: t - lags_k >= 0
    # Objective is just t
    return t


def _build_constraints(h0):
    """Build constraint functions for SLSQP."""
    n = len(h0)
    
    def sum_constraint(x):
        h = x[n:]
        return np.sum(h) - n / 2.0
    
    sum_constraint.grad = lambda x: np.concatenate([np.zeros(n), np.ones(n)])
    
    def bounds_constraints(x):
        h = x[n:]
        return np.concatenate([h, 1.0 - h])
    
    bounds_constraints.grad = lambda x: np.concatenate([
        np.eye(2 * n)[:n, :],
        np.eye(2 * n)[n:, :]
    ])
    
    # Correlation constraints: t - C_k >= 0 for each lag k
    def corr_constraints(x):
        t = x[0]
        h = x[n:]
        lags = _correlate(h, 1 - h, mode="full")
        return t - lags
    
    def corr_constraints_grad(x):
        t = x[0]
        h = x[n:]
        # Gradient of (t - C_k) w.r.t. x
        # d(t - C_k)/dt = 1
        # d(t - C_k)/dh = -dC_k/dh
        # C_k = sum_i h[i] * (1-h)[i+k] for appropriate shifts
        # This is complex; use finite differences or analytical
        # For simplicity, let SLSQP use numerical gradients for constraints
        # But we need to return something; SLSQP can handle if we don't provide grad
        return None
    
    constraints = [
        {'type': 'eq', 'fun': sum_constraint},
        {'type': 'ineq', 'fun': bounds_constraints},
    ]
    
    return constraints


def construct():
    n = N
    # Variables: [t, h_0, h_1, ..., h_{n-1}]
    x0 = np.concatenate([np.array([1.0]), np.full(n, 0.5)])
    
    bounds = [(0, 1)] + [(0, 1)] * n
    
    # Objective: minimize t
    def objective(x):
        return x[0]
    
    def objective_grad(x):
        grad = np.zeros(n + 1)
        grad[0] = 1.0
        return grad
    
    # Constraints
    def sum_con(x):
        return np.sum(x[1:]) - n / 2.0
    
    def sum_con_grad(x):
        g = np.zeros(n + 1)
        g[1:] = 1.0
        return g
    
    def lower_bounds_con(x):
        return x[1:]
    
    def lower_bounds_con_grad(x):
        g = np.zeros((n, n + 1))
        g[np.arange(n), 1 + np.arange(n)] = 1.0
        return g
    
    def upper_bounds_con(x):
        return 1.0 - x[1:]
    
    def upper_bounds_con_grad(x):
        g = np.zeros((n, n + 1))
        g[np.arange(n), 1 + np.arange(n)] = -1.0
        return g
    
    def corr_con(x):
        t = x[0]
        h = x[1:]
        lags = np.correlate(h, 1 - h, mode="full")
        return t - lags
    
    # For correlation constraints, we'll let SLSQP use numerical gradients
    # since analytical is complex
    
    constraints = [
        {'type': 'eq', 'fun': sum_con, 'jac': sum_con_grad},
        {'type': 'ineq', 'fun': lower_bounds_con, 'jac': lower_bounds_con_grad},
        {'type': 'ineq', 'fun': upper_bounds_con, 'jac': upper_bounds_con_grad},
        {'type': 'ineq', 'fun': corr_con},
    ]
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    
    if result.success:
        h = result.x[1:]
        # Clamp to valid range and fix sum
        h = np.clip(h, 0.0, 1.0)
        # Adjust to meet sum constraint exactly
        current_sum = np.sum(h)
        if abs(current_sum - n / 2.0) > 1e-9 * n:
            # Redistribute
            diff = (n / 2.0 - current_sum) / n
            h = h + diff
            h = np.clip(h, 0.0, 1.0)
            # Second pass
            current_sum = np.sum(h)
            diff = (n / 2.0 - current_sum) / n
            h = h + diff
            h = np.clip(h, 0.0, 1.0)
        
        # Verify validity
        if not np.all(np.isfinite(h)):
            return np.full(n, 0.5)
        if np.min(h) < -1e-9 or np.max(h) > 1 + 1e-9:
            return np.full(n, 0.5)
        if abs(np.sum(h) - n / 2.0) > 1e-9 * n:
            return np.full(n, 0.5)
        
        return h
    else:
        return np.full(n, 0.5)
# EVOLVE-BLOCK-END
