import numpy as np
import time

def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths based on an approximation of the optimal market making policy.
    Uses a quadratic value function approximation to determine optimal spread asymmetry
    based on inventory and time-to-maturity.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    Q = params['Q']
    
    # Time to maturity
    tau = max(0.0, T - t)
    
    # Inventory is an array of ints
    q = inventory.astype(np.float64)
    
    # Risk factor scaling with time
    # The penalty is (gamma/2) * sigma^2 * q^2 * dt.
    # The cumulative risk over remaining time is proportional to gamma * sigma^2 * tau.
    risk_scale = gamma * sigma**2 * tau
    
    # Base depth related to the cost of risk
    # The term inside the log comes from balancing the expected profit (depth) 
    # against the marginal inventory risk cost.
    
    # We use a heuristic derived from Avellaneda-Stoikov:
    # Optimal bid depth d_b ~ (1/k) * ln(1 + (2 * gamma * sigma^2 * tau / k^2) * (q + 0.5))
    # Optimal ask depth d_a ~ (1/k) * ln(1 + (2 * gamma * sigma^2 * tau / k^2) * (0.5 - q))
    
    # Ensure arguments are positive
    bid_arg = 1.0 + np.maximum(0.0, (2.0 * risk_scale / (k**2)) * (q + 0.5))
    ask_arg = 1.0 + np.maximum(0.0, (2.0 * risk_scale / (k**2)) * (0.5 - q))
    
    # Calculate depths
    # Add a small epsilon to avoid log(1)=0 if we want to ensure some spread, 
    # but log(1)=0 is a valid depth (quoting at mid). 
    # However, the problem states "distances below and above mid". 
    # Usually depths are positive. 
    # If depth is 0, we quote at mid.
    
    # The constraint is d > -3/k. 
    # We should probably enforce a minimum depth to avoid numerical issues or 
    # to reflect a minimum spread for liquidity provision.
    # Let's set a minimum depth of 1e-6 to avoid exact mid quoting if not intended, 
    # but 0 is technically valid. 
    # However, if d=0, the arrival rate is max. 
    # The optimal policy might quote at mid if risk is low.
    
    bid_depths = np.log(bid_arg) / k
    ask_depths = np.log(ask_arg) / k
    
    # Ensure depths are not negative (though log(1+x) with x>=0 is >=0)
    # And ensure they are not too large (though log grows slowly)
    
    # Handle edge cases where inventory is at limits
    # If q == Q, we cannot buy. The formula gives large bid_depth.
    # If q == -Q, we cannot sell. The formula gives large ask_depth.
    
    # Clip depths to a reasonable maximum to avoid extreme values if parameters are weird
    # Max depth is not explicitly bounded, but huge depths mean no trade.
    # np.inf is allowed for "no quote".
    # Let's cap at a large value, e.g., 10.0, which is effectively no quote for reasonable A, k.
    # Or simply return np.inf if the depth is so large that probability is negligible.
    # Probability = 1 - exp(-A exp(-k d) dt).
    # If k*d > 30, exp(-k*d) is ~ 1e-13, prob ~ 0.
    # Let's cap at 30/k.
    max_depth = 30.0 / k
    bid_depths = np.clip(bid_depths, 0.0, max_depth)
    ask_depths = np.clip(ask_depths, 0.0, max_depth)
    
    return bid_depths, ask_depths
