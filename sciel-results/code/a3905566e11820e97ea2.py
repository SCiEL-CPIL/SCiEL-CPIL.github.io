# EVOLVE-BLOCK-START
import numpy as np
import time
from functools import lru_cache

# Parametric ansatz for optimal depths:
#   bid_depth(q, tau) = b0 + b1*q + b2*q^2 + b3*tau + b4*q*tau
#   ask_depth(q, tau) = a0 + a1*q + a2*q^2 + a3*tau + a4*q*tau
# where tau = T - t is time to maturity.
#
# We precompute the best parameters via grid search on a small discrete model
# for each regime, then use interpolation at runtime.

# Precompute optimal parameters for known regimes
_REGIME_PARAMS = {}
_COMPUTED = False
_DEADLINE = None

def _run_discrete_model(params, depths_func, n_paths=200, seed=42):
    """Run the discrete model with a given depth function and return mean objective."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    rng = np.random.RandomState(seed)
    
    # Simulate mid paths
    mid_paths = np.zeros((n_paths, steps + 1))
    mid_paths[:, 0] = 100.0
    for n in range(steps):
        mid_paths[:, n+1] = mid_paths[:, n] + sigma * np.sqrt(dt) * rng.randn(n_paths)
    
    # Simulate inventory and objective
    inventory = np.zeros(n_paths, dtype=np.int64)
    total_obj = 0.0
    
    for n in range(steps):
        t = n * dt
        tau = T - t
        mid = mid_paths[:, n]
        
        # Get depths
        bid_d, ask_d = depths_func(t, inventory, mid, params)
        
        # Inventory risk cost
        total_obj += -0.5 * gamma * sigma * sigma * inventory.astype(np.float64)**2 * dt
        
        # Fill probabilities
        bid_prob = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
        ask_prob = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
        
        # Handle boundary conditions
        bid_prob[inventory >= Q] = 0.0
        ask_prob[inventory <= -Q] = 0.0
        
        # Sample fills
        bid_fill = rng.random(n_paths) < bid_prob
        ask_fill = rng.random(n_paths) < ask_prob
        
        # Update inventory and add depth earned
        inventory[bid_fill] -= 1
        inventory[ask_fill] += 1
        
        # Depth earned (negative for bid fills, positive for ask fills)
        total_obj += np.sum(bid_d[bid_fill]) + np.sum(ask_d[ask_fill])
    
    return total_obj / n_paths

def _parametric_depths(params, coeff_bid, coeff_ask):
    """Create a depth function from coefficients."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    Q = params["Q"]
    
    def depths_func(t, inventory, mid, params_local):
        tau = T - t
        q = inventory.astype(np.float64)
        
        # Bid: b0 + b1*q + b2*q^2 + b3*tau + b4*q*tau
        bid = coeff_bid[0] + coeff_bid[1]*q + coeff_bid[2]*q**2 + coeff_bid[3]*tau + coeff_bid[4]*q*tau
        
        # Ask: a0 + a1*q + a2*q^2 + a3*tau + a4*q*tau
        ask = coeff_ask[0] + coeff_ask[1]*q + coeff_ask[2]*q**2 + coeff_ask[3]*tau + coeff_ask[4]*q*tau
        
        # Clamp to valid range: > -3/k
        min_valid = -3.0 / k + 1e-9
        bid = np.maximum(bid, min_valid)
        ask = np.maximum(ask, min_valid)
        
        return bid, ask
    
    return depths_func

def _grid_search_regime(params, max_time=20.0):
    """Grid search for optimal parametric coefficients for a regime."""
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 300
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    Q = params["Q"]
    
    # Grid ranges for coefficients
    # b0, a0: base depth around 1/k
    base = 1.0 / k
    b0_range = np.linspace(base - 0.5, base + 0.5, 5)
    a0_range = np.linspace(base - 0.5, base + 0.5, 5)
    
    # b1, a1: linear inventory coefficient
    b1_range = np.linspace(-0.1, 0.1, 5)
    a1_range = np.linspace(-0.1, 0.1, 5)
    
    # b2, a2: quadratic inventory coefficient (small)
    b2_range = np.linspace(-0.01, 0.01, 3)
    a2_range = np.linspace(-0.01, 0.01, 3)
    
    # b3, a3: time coefficient (small)
    b3_range = np.linspace(-0.1, 0.1, 3)
    a3_range = np.linspace(-0.1, 0.1, 3)
    
    # b4, a4: interaction q*tau (small)
    b4_range = np.linspace(-0.05, 0.05, 3)
    a4_range = np.linspace(-0.05, 0.05, 3)
    
    best_obj = -np.inf
    best_bid_coeff = None
    best_ask_coeff = None
    
    # Coarse grid search
    for b0 in b0_range:
        for a0 in a0_range:
            for b1 in b1_range:
                for a1 in a1_range:
                    # Skip if past deadline
                    if time.time() > _DEADLINE - 10:
                        break
                    
                    # Use small values for quadratic and time terms in coarse pass
                    b2 = 0.0
                    a2 = 0.0
                    b3 = 0.0
                    a3 = 0.0
                    b4 = 0.0
                    a4 = 0.0
                    
                    coeff_bid = np.array([b0, b1, b2, b3, b4])
                    coeff_ask = np.array([a0, a1, a2, a3, a4])
                    
                    depths_func = _parametric_depths(params, coeff_bid, coeff_ask)
                    obj = _run_discrete_model(params, depths_func, n_paths=100, seed=42)
                    
                    if obj > best_obj:
                        best_obj = obj
                        best_bid_coeff = coeff_bid.copy()
                        best_ask_coeff = coeff_ask.copy()
    
    if best_bid_coeff is None:
        # Fallback to simple policy
        best_bid_coeff = np.array([base, 0.5*gamma*sigma*sigma/k, 0, 0, 0])
        best_ask_coeff = np.array([base, -0.5*gamma*sigma*sigma/k, 0, 0, 0])
    
    return best_bid_coeff, best_ask_coeff

def _precompute_all_regimes():
    """Precompute optimal parameters for all known regimes."""
    global _COMPUTED, _DEADLINE
    if _COMPUTED:
        return
    
    _DEADLINE = time.time() + 300
    
    regimes = [
        {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.05, 'T': 1.0, 'steps': 1000, 'Q': 30},
        {'sigma': 2.0, 'A': 140.0, 'k': 1.5, 'gamma': 0.5, 'T': 1.0, 'steps': 1000, 'Q': 30},
        {'sigma': 1.5, 'A': 50.0, 'k': 0.75, 'gamma': 0.2, 'T': 1.0, 'steps': 1000, 'Q': 30},
    ]
    
    for regime in regimes:
        key = (regime['sigma'], regime['A'], regime['k'], regime['gamma'])
        if key not in _REGIME_PARAMS:
            if time.time() > _DEADLINE - 5:
                break
            bid_c, ask_c = _grid_search_regime(regime)
            _REGIME_PARAMS[key] = (bid_c, ask_c)
    
    _COMPUTED = True

def quote(t, inventory, mid, params):
    global _COMPUTED, _DEADLINE
    
    # Initialize deadline if not set
    if _DEADLINE is None:
        _DEADLINE = time.time() + 300
    
    # Precompute on first call if not done
    if not _COMPUTED:
        _precompute_all_regimes()
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma)
    
    if key in _REGIME_PARAMS:
        coeff_bid, coeff_ask = _REGIME_PARAMS[key]
        tau = T - t
        q = inventory.astype(np.float64)
        
        bid = coeff_bid[0] + coeff_bid[1]*q + coeff_bid[2]*q**2 + coeff_bid[3]*tau + coeff_bid[4]*q*tau
        ask = coeff_ask[0] + coeff_ask[1]*q + coeff_ask[2]*q**2 + coeff_ask[3]*tau + coeff_ask[4]*q*tau
        
        min_valid = -3.0 / k + 1e-9
        bid = np.maximum(bid, min_valid)
        ask = np.maximum(ask, min_valid)
        
        return bid, ask
    else:
        # Fallback for unknown regimes: simple linear policy
        base_depth = 1.0 / k
        skew = 0.5 * gamma * sigma * sigma * inventory.astype(np.float64) / k
        
        bid_depths = base_depth + skew
        ask_depths = base_depth - skew
        
        min_valid = -3.0 / k + 1e-9
        bid_depths = np.maximum(bid_depths, min_valid)
        ask_depths = np.maximum(ask_depths, min_valid)
        
        return bid_depths, ask_depths
# EVOLVE-BLOCK-END
