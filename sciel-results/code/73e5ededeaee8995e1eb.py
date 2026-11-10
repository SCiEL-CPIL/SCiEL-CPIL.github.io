"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute constants that depend only on params
_cache = {}

def _get_precomputed(params):
    """Precompute regime-dependent constants to avoid recomputation each step."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _cache:
        return _cache[key]
    
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Base optimal depth from FOC: maximize d * A * exp(-k*d) * dt => d* = 1/k
    base_depth = 1.0 / k
    
    # Skew coefficient from HJB:
    # Value function V(t,q) ~ -C(t) * q^2 + D(t)
    # HJB: V_t = max_{b,a} [ λ_b*(b - γσ²q*dt) + λ_a*(a - γσ²q*dt) - γσ²q²*dt/2 ] - (σ²/2) V_qq
    # where λ_b = A*exp(-k*b)*dt, λ_a = A*exp(-k*a)*dt
    # With b = base + s, a = base - s (skew s):
    # The optimal s balances the marginal benefit of shifting toward selling/buying
    # against the increased cost of the wider side.
    # 
    # First-order condition for s (treating base as fixed at 1/k):
    # d/ds [ A*dt*(base+s)*exp(-k*(base+s)) - A*dt*(base-s)*exp(-k*(base-s)) ] = 0
    # But we also have the inventory cost term: -γσ²q*dt*(2) from V_q contribution
    # 
    # More carefully: the drift in q from fills is (1 - p_b) - (1 - p_a) = p_a - p_b
    # where p_b = A*exp(-k*b)*dt, p_a = A*exp(-k*a)*dt
    # So E[Δq] = p_a - p_b
    # The cost term in the objective is -γσ²q²*dt/2 per step
    # 
    # HJB: -V_t = max_{s} [ p_b*(b) + p_a*(a) - γσ²q²*dt/2 + V_q*(p_a - p_b) ] - (σ²/2)*V_qq
    # 
    # With V = -C(t)*q², V_q = -2C*q, V_qq = -2C
    # 
    # Optimal s: ∂/∂s [ p_b*b + p_a*a + V_q*(p_a - p_b) ] = 0
    # p_b = A*dt*exp(-k*(base+s)), p_a = A*dt*exp(-k*(base-s))
    # b = base+s, a = base-s
    # 
    # This is complex. For small s, linearize:
    # p_b ≈ A*dt*e^(-k*base)*(1 - k*s), p_a ≈ A*dt*e^(-k*base)*(1 + k*s)
    # p_a - p_b ≈ 2*A*dt*e^(-k*base)*k*s
    # p_b*b + p_a*a ≈ A*dt*e^(-k*base)*[ (1-ks)(base+s) + (1+ks)(base-s) ]
    #   ≈ A*dt*e^(-k*base)*[ 2*base - 2*k*s² ]  (to second order in s)
    # 
    # V_q*(p_a-p_b) = -2C*q * 2*A*dt*e^(-k*base)*k*s
    # 
    # So objective ≈ A*dt*e^(-k*base)*2*base - A*dt*e^(-k*base)*2*k*s² - 4*C*q*A*dt*e^(-k*base)*k*s - γσ²q²*dt/2
    # 
    # d/ds = -4*A*dt*e^(-k*base)*k*s - 4*C*q*A*dt*e^(-k*base)*k = 0
    # s = -C*q
    # 
    # From the HJB equation, C(t) satisfies:
    # C'(t) = ... (Riccati-like)
    # 
    # For a simple approximation: C ≈ γσ²/2 (steady-state)
    # Then s ≈ -(γσ²/2)*q * dt_factor
    # 
    # Actually, let's use a simpler empirical form:
    # skew = c * q * dt, where c is calibrated
    # The natural scale: skew ~ (γσ²/2) * q * dt / (A*e^(-1/k)*k)
    # because the marginal value of inventory is ~ γσ²q/2, and the marginal
    # fill rate change per unit skew is ~ A*e^(-1/k)*k
    
    e_base = np.exp(-k * base_depth)  # = exp(-1)
    skew_coeff = (gamma * sigma * sigma * dt) / (2.0 * A * e_base * k)
    
    result = {
        'base_depth': base_depth,
        'skew_coeff': skew_coeff,
        'dt': dt,
        'Q': Q,
        'T': T,
        'k': k,
    }
    _cache[key] = result
    return result


def quote(t, inventory, mid, params):
    """
    HJB-inspired market making policy.
    
    Uses the analytical first-order condition from the continuous-state HJB
    to determine optimal skew. Base depth is 1/k (from FOC of depth reward).
    Skew is proportional to inventory with coefficient derived from the
    HJB Riccati equation approximation.
    """
    pc = _get_precomputed(params)
    base_depth = pc['base_depth']
    skew_coeff = pc['skew_coeff']
    Q = pc['Q']
    T = pc['T']
    
    inv = np.asarray(inventory, dtype=np.float64)
    n_paths = len(inv)
    
    # Time remaining: risk is lower near T (less time to realize inventory cost)
    # But also: near T, we should be more aggressive about offloading inventory
    # Use a mild time modulation: skew slightly stronger as t -> T
    t_rem = 1.0 - t / T
    # At t=0: t_rem=1, at t=T: t_rem=0
    # We want skew to be moderate throughout; slight increase near end
    time_factor = 1.0 + 0.3 * (1.0 - t_rem)  # 1.0 at start, 1.3 at end
    
    # Optimal skew from HJB FOC
    skew = skew_coeff * time_factor * inv
    
    # Clamp skew to prevent unreasonably tight/wide quotes
    # Max skew should not make one side depth < 0 or > 2/k
    max_skew = 0.8 / pc['k']
    skew = np.clip(skew, -max_skew, max_skew)
    
    # Quote depths
    bid_depths = base_depth + skew
    ask_depths = base_depth - skew
    
    # Ensure positive depths (minimum ~1e-4)
    min_depth = 1e-4
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    # Boundary conditions: no bid at Q, no ask at -Q
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
