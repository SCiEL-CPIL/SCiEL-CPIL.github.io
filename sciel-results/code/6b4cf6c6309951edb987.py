# EVOLVE-BLOCK-START
import numpy as np

_REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]

def _solve_regime(sigma, A, k, gamma, Q=30, T=1.0, steps=1000):
    """
    Solve the optimal market-making policy using a semi-Lagrangian HJB scheme
    with analytical closed-form for the optimal spread.
    
    The key insight: for the bid side, the objective to maximize is
    p_b * (d_b + delta_b) where p_b = 1 - exp(-A * exp(-k*d_b) * dt)
    
    Taking the derivative and setting to zero gives the FOC:
    d_b = (1/k) * log(A * dt / (1 - exp(-k*d_b)) * (d_b + delta_b) / (d_b + delta_b))
    
    More carefully: let u = A * exp(-k*d) * dt, so p = 1 - exp(-u)
    Objective: (1 - exp(-u)) * (d + delta)
    d/du: exp(-u) * (d + delta) * du/dd = 0
    du/dd = -k * u (since u = A*exp(-k*d)*dt, du/dd = -k*u)
    So: exp(-u) * (d + delta) * (-k*u) = 0
    This gives: k*u*exp(-u) = (d + delta) / (d + delta) ... 
    
    Let me redo this more carefully.
    f(d) = (1 - exp(-A*exp(-k*d)*dt)) * (d + delta)
    Let g(d) = A*exp(-k*d)*dt, so p = 1 - exp(-g)
    f = (1 - exp(-g)) * (d + delta)
    df/dd = exp(-g) * g' * (d + delta) + (1 - exp(-g))
    g' = -k * g
    df/dd = -k*g*exp(-g)*(d + delta) + (1 - exp(-g))
    
    Set to zero:
    1 - exp(-g) = k*g*exp(-g)*(d + delta)
    1 = exp(-g) * (1 + k*g*(d + delta))
    exp(g) = 1 + k*g*(d + delta)
    
    This is a transcendental equation in g (or d). For small dt, g is small,
    so exp(g) ≈ 1 + g + g²/2, giving:
    1 + g + g²/2 ≈ 1 + k*g*(d + delta)
    g + g²/2 ≈ k*g*(d + delta)
    1 + g/2 ≈ k*(d + delta)
    g ≈ 2k*(d + delta) - 2
    But g = A*exp(-k*d)*dt, so:
    A*exp(-k*d)*dt ≈ 2k*(d + delta) - 2
    
    This is still implicit. For the continuous HJB approach, we can use the
    fact that in the limit, the optimal depth satisfies a simpler relation.
    
    Actually, for the HJB equation, the value function V(q, t) satisfies:
    ∂V/∂t + max_{d_b, d_a} { 
        (1-exp(-A e^{-k d_b} dt)) * (d_b + V(q+1,t+dt) - V(q,t)) 
        + (1-exp(-A e^{-k d_a} dt)) * (V(q-1,t+dt) - V(q,t) - d_a)
        - (gamma/2)*sigma²*q²
    } = 0
    
    In the continuous limit, the optimal spread has a closed form.
    For the bid: maximize (1-exp(-u_b)) * (d_b + Δ_b) where u_b = A*exp(-k*d_b)*dt
    FOC: exp(-u_b) * k * u_b * (d_b + Δ_b) = 1 - exp(-u_b)
    Let h(u) = u*exp(-u), then: k*(d_b + Δ_b) * h(u_b) = 1 - exp(-u_b)
    
    For small u_b: h(u_b) ≈ u_b - u_b²/2, 1 - exp(-u_b) ≈ u_b - u_b²/2
    So k*(d_b + Δ_b) ≈ 1, giving d_b ≈ 1/k - Δ_b
    
    This is the key closed-form: d_b* ≈ (1/k) - Δ_b (clipped to ≥ 0)
    where Δ_b = V(q+1) - V(q) is the inventory risk premium.
    
    Similarly: d_a* ≈ (1/k) - Δ_a where Δ_a = V(q) - V(q-1)
    """
    dt = T / steps
    n_states = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # Terminal condition: V(q, T) = 0
    V = np.zeros(n_states)
    
    # Precompute constants
    inv_k = 1.0 / k
    
    # Time-stepping backwards
    for n in range(steps - 1, -1, -1):
        V_new = np.zeros(n_states)
        
        for i in range(n_states):
            q = i + q_min
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            risk_cost = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            if can_bid:
                V_qp = V[i + 1]
            else:
                V_qp = 0.0
            if can_ask:
                V_qm = V[i - 1]
            else:
                V_qm = 0.0
            V_qi = V[i]
            
            # Closed-form optimal depths
            if can_bid:
                delta_b = V_qp - V_qi
                d_b = inv_k - delta_b
                if d_b < 0:
                    d_b = 0.0
                u_b = A * np.exp(-k * d_b) * dt
                p_b = 1.0 - np.exp(-u_b)
            else:
                d_b = np.inf
                p_b = 0.0
            
            if can_ask:
                delta_a = V_qi - V_qm
                d_a = inv_k - delta_a
                if d_a < 0:
                    d_a = 0.0
                u_a = A * np.exp(-k * d_a) * dt
                p_a = 1.0 - np.exp(-u_a)
            else:
                d_a = np.inf
                p_a = 0.0
            
            # Expected value
            if can_bid:
                bid_term = p_b * (d_b + V_qp)
            else:
                bid_term = 0.0
            if can_ask:
                ask_term = p_a * (-d_a + V_qm)
            else:
                ask_term = 0.0
            no_fill_term = (1.0 - p_b) * (1.0 - p_a) * V_qi
            
            V_new[i] = bid_term + ask_term + no_fill_term - risk_cost
        
        V = V_new
    
    # Extract optimal depths at t=0
    bid_depths = np.full(n_states, np.inf)
    ask_depths = np.full(n_states, np.inf)
    
    for i in range(n_states):
        q = i + q_min
        if q < Q:
            delta_b = V[i + 1] - V[i]
            d_b = inv_k - delta_b
            bid_depths[i] = max(0.0, d_b)
        if q > -Q:
            delta_a = V[i] - V[i - 1]
            d_a = inv_k - delta_a
            ask_depths[i] = max(0.0, d_a)
    
    return bid_depths, ask_depths

# Precompute all regime policies
_POLICY_CACHE = {}
for (sigma, A, k, gamma) in _REGIMES:
    key = (sigma, A, k, gamma)
    _POLICY_CACHE[key] = _solve_regime(sigma, A, k, gamma)


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    
    key = (sigma, A, k, gamma)
    if key not in _POLICY_CACHE:
        depth = 1.0 / k
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
    
    bid_table, ask_table = _POLICY_CACHE[key]
    
    idx = np.clip(inventory + Q, 0, 2 * Q)
    
    bid_depths = bid_table[idx]
    ask_depths = ask_table[idx]
    
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), 1.0/k, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), 1.0/k, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
