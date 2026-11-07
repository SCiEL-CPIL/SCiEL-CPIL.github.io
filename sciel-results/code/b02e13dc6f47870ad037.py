"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 290  # Safety margin before 330s kill


def quote(t, inventory, mid, params):
    """
    Inventory-aware quoting with time-decay and mild nonlinearity.
    
    Strategy:
    - Base depth from 1/k (risk-neutral)
    - Linear inventory adjustment (strongest component from current program)
    - Time-decay: reduce adjustment as remaining time shrinks (no terminal charge)
    - Mild tanh nonlinearity to prevent extreme depths at large |q|
    - Scale with volatility and gamma for risk management
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    dt = T / steps
    remaining_time = T - t
    # Time decay: full adjustment at start, reduced near end
    # Use sqrt(remaining) for gentle decay
    time_factor = np.sqrt(max(remaining_time / T, 0.0))
    
    q = inventory.astype(np.float64)
    
    # Base depth: risk-neutral level
    base_depth = 1.0 / k
    
    # Inventory adjustment strength
    # beta = 0.02 worked well; scale mildly with sigma, gamma, and time
    # Higher gamma means stronger penalty for inventory, so we need stronger adjustment
    # Higher sigma means more risk, so we need stronger adjustment
    # time_factor: more time remaining means we can afford to be more aggressive
    beta = 0.02 * (0.5 + 0.5 * time_factor) * (1.0 + 0.2 * sigma / 2.0) * (1.0 + 0.3 * gamma)
    
    # Mild nonlinearity: use a soft-clip that is linear for |q| < ~0.6*Q
    # and gently saturates beyond that
    # tanh(1.2 * q/Q) is approximately linear for |q/Q| < 0.5
    q_normalized = q / Q
    # Use a blend: mostly linear, slight saturation
    q_adjust = np.tanh(1.2 * q_normalized)
    # Re-scale to make it closer to linear in the central region
    # tanh(1.2*x) ≈ 1.2*x for small x, so divide by 1.2 to get ~x
    q_adjust = q_adjust / 1.2  # Now approximately linear for |q| < 0.6*Q
    
    bid_depth = base_depth + beta * q_adjust
    ask_depth = base_depth - beta * q_adjust
    
    # Clamp to minimum positive depth
    min_depth = 0.01
    bid_depth = np.maximum(bid_depth, min_depth)
    ask_depth = np.maximum(ask_depth, min_depth)
    
    # Cap maximum depth to avoid extremely wide quotes
    max_depth = 4.0 / k
    bid_depth = np.minimum(bid_depth, max_depth)
    ask_depth = np.minimum(ask_depth, max_depth)
    
    # At boundaries, no quote on the side that would push inventory further
    bid_depth[inventory >= Q] = np.inf
    ask_depth[inventory <= -Q] = np.inf
    
    # Safety check for deadline
    if time.time() > _DEADLINE:
        # Return simple fallback
        depth = np.full(len(inventory), base_depth)
        depth[inventory >= Q] = np.inf
        depth2 = np.full(len(inventory), base_depth)
        depth2[inventory <= -Q] = np.inf
        return depth, depth2
    
    return bid_depth, ask_depth
# EVOLVE-BLOCK-END
