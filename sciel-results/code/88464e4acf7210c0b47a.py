"""Optimal market making with inventory risk - Analytic HJB first-order condition solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

_START = time.time()
_DEADLINE = _START + 280.0  # safety margin before 330s kill
_policy_cache = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve for the optimal policy using the HJB first-order condition.
    
    In continuous time, the HJB equation for the value function V(t, q) is:
      -V_t = (gamma/2)*sigma^2*q^2 + max_{b,a} [
          lambda_b(b) * (b + V(t+dt, q-1) - V(t+dt, q))
        + lambda_a(a) * (a + V(t+dt, q+1) - V(t+dt, q))
      ]
    
    where lambda_b(b) = A*exp(-k*b) is the arrival rate for bid at depth b.
    
    The FOC for bid depth b:
      d/db [lambda_b(b) * (b + delta_V_b)] = 0
      => -k*lambda_b(b)*(b + delta_V_b) + lambda_b(b) = 0
      => 1 - k*(b + delta_V_b) = 0
      => b* = 1/k - delta_V_b
    
    where delta_V_b = V(q-1) - V(q) (the value of having one less unit of inventory).
    
    Similarly for ask: a* = 1/k - delta_V_a where delta_V_a = V(q+1) - V(q).
    
    The constraint is b >= 0, a >= 0. If the FOC gives negative depth, use 0 (or np.inf for no quote).
    
    We solve this via backward induction on the discrete time grid, but using the
    analytic FOC instead of grid search. The value function is approximated as
    quadratic in q: V(t, q) ≈ c(t) + d(t)*q + e(t)*q^2
    (This is exact for the linear-quadratic case, which is a good approximation here.)
    
    Actually, let's be more precise. The FOC gives:
      b* = max(0, 1/k - (V(q-1) - V(q)))
      a* = max(0, 1/k - (V(q+1) - V(q)))
    
    If we assume V is quadratic: V(q) = c + d*q + e*q^2
    Then V(q-1) - V(q) = d*(q-1) + e*(q-1)^2 - d*q - e*q^2
                         = -d + e*(q^2 - 2q + 1 - q^2)
                         = -d + e*(1 - 2q)
    
    And V(q+1) - V(q) = d*(q+1) + e*(q+1)^2 - d*q - e*q^2
                       = d + e*(2q + 1)
    
    So:
      b* = max(0, 1/k - (-d + e*(1-2q))) = max(0, 1/k + d - e + 2*e*q)
      a* = max(0, 1/k - (d + e*(2q+1))) = max(0, 1/k - d - e - 2*e*q)
    
    This gives us a linear-in-q policy! We just need to find d(t) and e(t) via backward induction.
    
    However, the quadratic approximation may not be perfect. Let's use the full value
    function on the discrete inventory grid but use the analytic FOC for depth selection.
    """
    dt = T / steps
    
    # We'll track the value function on the inventory grid
    # V_next[q] for q in -Q..Q
    n_inv = 2 * Q + 1
    V_next = np.zeros(n_inv)
    
    # Policy table: policy[step, inv_index] = (bid_depth, ask_depth)
    policy = np.zeros((steps, n_inv, 2))
    
    # Solve backwards
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            return None
        
        V_curr = np.zeros(n_inv)
        
        for i_q in range(n_inv):
            q = i_q - Q  # actual inventory value
            
            # Inventory cost for this step
            cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # Value differences
            if i_q > 0:
                V_qm1 = V_next[i_q - 1]
            else:
                V_qm1 = V_next[0]  # at boundary q=-Q, can't go lower (no ask fill possible)
            
            V_q = V_next[i_q]
            
            if i_q < n_inv - 1:
                V_qp1 = V_next[i_q + 1]
            else:
                V_qp1 = V_next[n_inv - 1]  # at boundary q=Q, can't go higher
            
            # FOC for bid: b* = max(0, 1/k - (V(q-1) - V(q)))
            # But we need to be careful: the FOC derivation assumes the arrival rate
            # lambda = A*exp(-k*b), and the expected reward from a bid fill is
            # lambda*dt * (b + V(q-1) - V(q))
            # The FOC: d/db [A*exp(-k*b)*dt*(b + delta)] = 0
            # => A*dt*exp(-k*b) * [1 - k*(b + delta)] = 0
            # => b + delta = 1/k
            # => b = 1/k - delta
            # where delta = V(q-1) - V(q)
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if can_bid:
                delta_bid = V_qm1 - V_q
                b_star = 1.0 / k - delta_bid
                b_star = max(0.0, b_star)
            else:
                b_star = np.inf
            
            if can_ask:
                delta_ask = V_qp1 - V_q
                a_star = 1.0 / k - delta_ask
                a_star = max(0.0, a_star)
            else:
                a_star = np.inf
            
            # Now compute the expected value with these optimal depths
            if np.isinf(b_star) and np.isinf(a_star):
                V_curr[i_q] = cost + V_q
                policy[n, i_q, 0] = np.inf
                policy[n, i_q, 1] = np.inf
                continue
            
            if np.isinf(b_star):
                # Only ask can fill
                p_a = 1.0 - np.exp(-A * np.exp(-k * a_star) * dt)
                # E = cost + a_star*p_a + p_a*V_qp1 + (1-p_a)*V_q
                V_curr[i_q] = cost + a_star * p_a + p_a * V_qp1 + (1 - p_a) * V_q
                policy[n, i_q, 0] = np.inf
                policy[n, i_q, 1] = a_star
                continue
            
            if np.isinf(a_star):
                # Only bid can fill
                p_b = 1.0 - np.exp(-A * np.exp(-k * b_star) * dt)
                # E = cost + b_star*p_b + p_b*V_qm1 + (1-p_b)*V_q
                V_curr[i_q] = cost + b_star * p_b + p_b * V_qm1 + (1 - p_b) * V_q
                policy[n, i_q, 0] = b_star
                policy[n, i_q, 1] = np.inf
                continue
            
            # Both can fill
            p_b = 1.0 - np.exp(-A * np.exp(-k * b_star) * dt)
            p_a = 1.0 - np.exp(-A * np.exp(-k * a_star) * dt)
            
            # E = cost + b_star*p_b + a_star*p_a
            #   + p_b*(1-p_a)*V_qm1 + p_a*(1-p_b)*V_qp1
            #   + (1-p_b-p_a+2*p_b*p_a)*V_q
            V_curr[i_q] = (cost
                           + b_star * p_b + a_star * p_a
                           + p_b * (1 - p_a) * V_qm1
                           + p_a * (1 - p_b) * V_qp1
                           + (1 - p_b - p_a + 2 * p_b * p_a) * V_q)
            
            policy[n, i_q, 0] = b_star
            policy[n, i_q, 1] = a_star
        
        V_next = V_curr
    
    return policy


def quote(t, inventory, mid, params):
    """Quote bid and ask depths based on analytic HJB FOC policy."""
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    dt = T / steps
    
    # Determine step index
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    # Cache key
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    
    if cache_key not in _policy_cache:
        if time.time() > _DEADLINE - 30:
            # Fallback: simple analytic formula
            base_depth = 1.0 / k
            # Linear-in-inventory adjustment from quadratic value approximation
            # b* ≈ 1/k + d - e + 2*e*q, a* ≈ 1/k - d - e - 2*e*q
            # For small gamma, d≈0, e≈gamma*sigma^2/(2*A*k) approximately
            e_est = gamma * sigma * sigma / (2.0 * A * k)
            d_est = 0.0
            bid_depths = np.maximum(0.0, 1.0/k + d_est - e_est + 2.0*e_est*inventory)
            ask_depths = np.maximum(0.0, 1.0/k - d_est - e_est - 2.0*e_est*inventory)
            # Boundary conditions
            bid_depths[inventory >= Q] = np.inf
            ask_depths[inventory <= -Q] = np.inf
            return bid_depths, ask_depths
        
        policy = _solve_regime(sigma, A, k, gamma, T, steps, Q)
        if policy is None:
            # Fallback
            base_depth = 1.0 / k
            e_est = gamma * sigma * sigma / (2.0 * A * k)
            bid_depths = np.maximum(0.0, 1.0/k - e_est + 2.0*e_est*inventory)
            ask_depths = np.maximum(0.0, 1.0/k - e_est - 2.0*e_est*inventory)
            bid_depths[inventory >= Q] = np.inf
            ask_depths[inventory <= -Q] = np.inf
            return bid_depths, ask_depths
        
        _policy_cache[cache_key] = policy
    
    policy = _policy_cache[cache_key]
    
    # Look up policy for each path
    n_inv = 2 * Q + 1
    
    # Clip inventory to valid range
    inv_clipped = np.clip(inventory, -Q, Q)
    inv_idx = inv_clipped + Q  # 0 to 2*Q
    
    bid_depths = policy[n, inv_idx, 0]
    ask_depths = policy[n, inv_idx, 1]
    
    # Ensure valid values: no negatives, no NaN
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), np.inf, ask_depths)
    
    # Handle boundary: no bid at q=Q, no ask at q=-Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
