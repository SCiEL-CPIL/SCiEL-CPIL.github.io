"""Symbolic regression: nonlinear oscillator acceleration via adaptive basis search.

Combines the physical structure of the seed program (Duffing + Van der Pol + parametric
excitation) with the RBF-enhanced basis search from inspiration 1. At import time, we
generate a small set of candidate basis functions that blend polynomial, trigonometric,
and RBF-local features. We select the best basis using a synthetic surrogate that mimics
the expected oscillator dynamics, then bake the selected basis into `equation`.

The final model is linear in the 10 params, so the evaluator's BFGS fit recovers the
same optimum as our least-squares selection.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 130.0  # Explicit wall-clock safety margin (< 150 s kill)

# Precomputed RBF centers for local phase-plane features
_RBF_CENTERS_X = np.array([-0.15, 0.0, 0.15])
_RBF_CENTERS_V = np.array([-0.25, 0.0, 0.25])
_RBF_SIGMA = 0.2


def _make_basis(t, x, v, spec):
    """Build a 10-column feature matrix for a given basis specification."""
    cols = []
    cols.append(np.ones_like(t))  # bias
    for term in spec:
        if term == "sin_t":
            cols.append(np.sin(t))
        elif term == "cos_t":
            cols.append(np.cos(t))
        elif term == "x":
            cols.append(x)
        elif term == "x2":
            cols.append(x * x)
        elif term == "x3":
            cols.append(x * x * x)
        elif term == "x4":
            cols.append(x * x * x * x)
        elif term == "v":
            cols.append(v)
        elif term == "v2":
            cols.append(v * v)
        elif term == "v3":
            cols.append(v * v * v)
        elif term == "xv":
            cols.append(x * v)
        elif term == "x2v":
            cols.append(x * x * v)
        elif term == "xv2":
            cols.append(x * v * v)
        elif term == "sin_t_x":
            cols.append(np.sin(t) * x)
        elif term == "cos_t_x":
            cols.append(np.cos(t) * x)
        elif term == "sin_t_v":
            cols.append(np.sin(t) * v)
        elif term == "cos_t_v":
            cols.append(np.cos(t) * v)
        elif term == "tanh_x":
            cols.append(np.tanh(x))
        elif term == "tanh_v":
            cols.append(np.tanh(v))
        elif term == "tanh_xv":
            cols.append(np.tanh(x * v))
        elif term == "tanh_x3":
            cols.append(np.tanh(x * x * x))
        elif term == "rbf0":
            dx = x - _RBF_CENTERS_X[0]
            dv = v - _RBF_CENTERS_V[0]
            cols.append(np.exp(-(dx * dx + dv * dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf1":
            cols.append(np.exp(-(x * x + v * v) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf2":
            dx = x - _RBF_CENTERS_X[2]
            dv = v - _RBF_CENTERS_V[2]
            cols.append(np.exp(-(dx * dx + dv * dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf0_x":
            dx = x - _RBF_CENTERS_X[0]
            dv = v - _RBF_CENTERS_V[0]
            cols.append(x * np.exp(-(dx * dx + dv * dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf1_x":
            cols.append(x * np.exp(-(x * x + v * v) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf2_x":
            dx = x - _RBF_CENTERS_X[2]
            dv = v - _RBF_CENTERS_V[2]
            cols.append(x * np.exp(-(dx * dx + dv * dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf0_v":
            dx = x - _RBF_CENTERS_X[0]
            dv = v - _RBF_CENTERS_V[0]
            cols.append(v * np.exp(-(dx * dx + dv * dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf1_v":
            cols.append(v * np.exp(-(x * x + v * v) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf2_v":
            dx = x - _RBF_CENTERS_X[2]
            dv = v - _RBF_CENTERS_V[2]
            cols.append(v * np.exp(-(dx * dx + dv * dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
    return np.column_stack(cols)


# Candidate basis specifications (10 features each, including bias)
_SPECS = [
    # Seed-like: Duffing + Van der Pol + parametric
    ["sin_t", "cos_t", "x", "x3", "v", "xv", "sin_t_x", "cos_t_x", "sin_t_v", "x2v"],
    # RBF-enhanced: local phase-plane coupling
    ["sin_t", "cos_t", "x", "v", "rbf0", "rbf1", "rbf2", "xv", "x3", "x2"],
    # RBF linearized: weighted RBFs for gradient-like terms
    ["sin_t", "cos_t", "x", "v", "rbf0_x", "rbf1_x", "rbf2_x", "rbf0_v", "rbf1_v", "xv"],
    # Hybrid: Duffing + RBF
    ["sin_t", "cos_t", "x", "x3", "v", "v3", "rbf0", "rbf1", "xv", "x2"],
    # tanh-saturated + RBF
    ["sin_t", "cos_t", "x", "x3", "v", "tanh_xv", "rbf0", "rbf1", "xv", "x2"],
    # Higher-order polynomial + RBF
    ["sin_t", "cos_t", "x", "x3", "x4", "v", "rbf0", "rbf1", "rbf2", "xv"],
    # Asymmetric damping + RBF
    ["sin_t", "cos_t", "x", "x3", "v", "v2", "v3", "xv", "x2v", "rbf1"],
    # Pure RBF + drive
    ["sin_t", "cos_t", "rbf0", "rbf1", "rbf2", "rbf0_x", "rbf1_x", "rbf0_v", "rbf1_v", "xv"],
]


def _build_surrogate(n=2000):
    """Synthesize plausible nonlinear-oscillator acceleration data for basis selection."""
    np.random.seed(42)
    t = np.random.uniform(0, 20, n)
    x = np.random.uniform(-0.3, 0.3, n)
    v = np.random.uniform(-0.4, 0.4, n)
    
    # Ground truth: Duffing + Van der Pol + parametric drive
    y = (0.5 * np.sin(t) + 0.3 * np.cos(t) +
         -2.0 * x + 3.0 * x**3 +
         -0.5 * v + 1.5 * x**2 * v +
         0.2 * np.sin(t) * x + 0.1 * np.cos(t) * x)
    y += 0.05 * np.random.randn(n)  # small noise
    
    return t, x, v, y


def _select_best_basis():
    """Select the best basis specification using least-squares on surrogate data."""
    t, x, v, y = _build_surrogate()
    best_spec = _SPECS[0]
    best_mse = np.inf
    
    for spec in _SPECS:
        if time.time() > _DEADLINE - 10.0:
            break
        try:
            X = _make_basis(t, x, v, spec)
            # Linear least squares
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ coeffs
            mse = np.mean((y_pred - y) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_spec = spec
        except Exception:
            continue
    
    return best_spec


# Select the best basis at import time
_SELECTED_SPEC = _select_best_basis()


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Closed-form expression for a nonlinear oscillator acceleration.
    
    Uses the selected basis from _SELECTED_SPEC. The model is linear in params,
    so the evaluator's BFGS fit recovers the optimal coefficients.
    """
    X = _make_basis(t, x, v, _SELECTED_SPEC)
    y = X @ params
    return y
# EVOLVE-BLOCK-END
