import numpy as np
import time
from scipy.optimize import brentq

_START = time.time()
_DEADLINE = _START + 290.0
_CACHE = {}

def _solve_depth(A, k, dt, C):
    """
    Solve for optimal depth d >= 0 maximizing:
    d * (1 - exp(-A * exp(-k*d) * dt)) + C * (1 - exp(-A * exp(-k*d) * dt))
    
    Let P(d) = 1 - exp(-A * exp(-k*d) * dt)
    f(d) = d * P(d) + C * P(d)
    f'(d) = P(d) + (d + C) * P'(d)
    P'(d) = A * exp(-k*d) * dt * k * exp(-k*d) = A * k * dt * exp(-2*k*d)
    
    So f'(d) = P(d) + (d+C) * A * k * dt * exp(-2*k*d)
    
    We want to find d such that f'(d) = 0, or the boundary d=0 if f'(0) <= 0.
    Note: as d -> inf, P(d) -> 1, P'(d) -> 0, so f'(d) -> 1 > 0.
    At d=0, P(0) = 1 - exp(-A*dt), P'(0) = A*k*dt.
    f'(0) = P(0) + C * A*k*dt.
    
    If C >= 0, f'(0) > 0 always, so optimal is d -> inf? No wait.
    Let's re-examine. As d increases, P(d) increases from 1-exp(-A*dt) to 1.
    Actually P(d) = 1 - exp(-A*exp(-k*d)*dt).
    As d -> 0, exp(-k*d) -> 1, so P(0) = 1 - exp(-A*dt).
    As d -> inf, exp(-k*d) -> 0, so P(inf) = 1 - 0 = 1.
    
    Wait, the fill probability is 1 - exp(-A * exp(-k * bid_depth) * dt).
    Larger depth means smaller exp(-k*d), so smaller A*exp(-k*d)*dt, so smaller fill probability.
    So P(d) is DECREASING in d.
    
    Let me re-read the model.
    "bid hit with probability 1 - exp(-A exp(-k bid_depth) dt)"
    
    So if bid_depth is large, exp(-k * bid_depth) is small, so A * exp(-k * bid_depth) * dt is small,
    so 1 - exp(-small) is small. So large depth -> low fill probability.
    This makes sense: quoting far from mid means less likely to be hit.
    
    So P(d) = 1 - exp(-A * exp(-k*d) * dt) is decreasing in d.
    P(0) = 1 - exp(-A*dt) (max fill prob)
    P(inf) = 1 (min fill prob, actually 0? No, exp(-0)=1, so 1-exp(0)=0).
    Wait, exp(-k*inf) = 0, so A*0*dt = 0, exp(-0) = 1, 1-1 = 0.
    So P(inf) = 0.
    
    Objective: d * P(d) + C * P(d) = (d+C) * P(d)
    
    As d -> 0, (d+C)*P(d) -> C * P(0) = C * (1 - exp(-A*dt))
    As d -> inf, (d+C)*P(d) -> 0 (since P(d) decays exponentially, d*P(d) -> 0)
    
    So there is an interior maximum.
    
    f(d) = (d+C) * P(d)
    f'(d) = P(d) + (d+C) * P'(d)
    P'(d) = -A * k * dt * exp(-k*d) * exp(-A*exp(-k*d)*dt)
           = -A * k * dt * exp(-k*d) * (1 - P(d))
    
    f'(d) = P(d) - (d+C) * A * k * dt * exp(-k*d) * (1 - P(d))
    
    Set f'(d) = 0:
    P(d) = (d+C) * A * k * dt * exp(-k*d) * (1 - P(d))
    
    Let h(d) = P(d) - (d+C) * A * k * dt * exp(-k*d) * (1 - P(d))
    We want h(d) = 0.
    
    h(0) = P(0) - C * A * k * dt * 1 * (1 - P(0))
         = (1 - exp(-A*dt)) - C * A * k * dt * exp(-A*dt)
    
    If h(0) <= 0, then optimal d = 0.
    If h(0) > 0, there is a unique root in (0, inf).
    """
    if C <= 0 and C > -1e-10:
        # If C is very close to 0 or negative, check boundary
        pass
    
    def h(d):
        P = 1.0 - np.exp(-A * np.exp(-k * d) * dt)
        return P - (d + C) * A * k * dt * np.exp(-k * d) * (1.0 - P)
    
    h0 = h(0.0)
    if h0 <= 0:
        return 0.0
    
    # Find upper bound
    d_max = 20.0 / k
    if h(d_max) > 0:
        d_max = 50.0 / k
    
    try:
        return brentq(h, 0.0, d_max, xtol=1e-10)
    except:
        return 0.0

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    
    V = np.zeros((N + 1, n_q))
    opt_b = np.zeros((N, n_q))
    opt_a = np.zeros((N, n_q))
    
    for t in range(N - 1, -1, -1):
        if time.time() > _DEADLINE - 5:
            for qi in range(n_q):
                opt_b[t, qi] = 1.0 / k
                opt_a[t, qi] = 1.0 / k
            return opt_b, opt_a
        
        V_next = V[t + 1, :]
        
        for qi in range(n_q):
            q = qi - Q  # actual inventory
            
            # delta_plus = V_next[qi+1] - V_next[qi] (if qi+1 < n_q)
            # delta_minus = V_next[qi-1] - V_next[qi] (if qi-1 >= 0)
            
            dp = V_next[qi+1] - V_next[qi] if qi + 1 < n_q else 0.0
            dm = V_next[qi-1] - V_next[qi] if qi - 1 >= 0 else 0.0
            
            # Inventory cost
            cost = (gamma / 2.0) * sigma * sigma * q * q * dt
            
            if qi == 0:
                # q = -Q, cannot have bid fill (would go to -Q-1)
                # Optimize ask only
                # J(a) = a*P(a) + V_next[qi] + P(a)*dm - cost
                # = (a + dm) * P(a) + V_next[qi] - cost
                # Optimal a maximizes (a + dm) * P(a)
                a_opt = _solve_depth(A, k, dt, dm)
                opt_a[t, qi] = a_opt
                opt_b[t, qi] = 0.0
                V[t, qi] = (a_opt + dm) * (1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)) + V_next[qi] - cost
            elif qi == n_q - 1:
                # q = Q, cannot have ask fill (would go to Q+1)
                # Optimize bid only
                # J(b) = b*P(b) + V_next[qi] + P(b)*dp - cost
                # = (b + dp) * P(b) + V_next[qi] - cost
                b_opt = _solve_depth(A, k, dt, dp)
                opt_b[t, qi] = b_opt
                opt_a[t, qi] = 0.0
                V[t, qi] = (b_opt + dp) * (1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)) + V_next[qi] - cost
            else:
                # Interior: optimize both b and a
                # J(b, a) = b*P(b) + a*P(a) + P(b)*dp + P(a)*dm - P(b)*P(a)*(dp+dm) + V_next[qi] - cost
                # = b*P(b) + P(b)*dp - P(b)*P(a)*dp + a*P(a) + P(a)*dm - P(b)*P(a)*dm + V_next[qi] - cost
                # = P(b)*(b + dp - P(a)*dp) + P(a)*(a + dm - P(b)*dm) + V_next[qi] - cost
                
                # This is coupled. Let's use the fact that for fixed a, optimal b solves:
                # max_b P(b)*(b + dp - P(a)*dp) = max_b (b + dp*(1-P(a))) * P(b)
                # So b_opt(a) = _solve_depth(A, k, dt, dp*(1-P(a)))
                
                # And for fixed b, a_opt(b) = _solve_depth(A, k, dt, dm*(1-P(b)))
                
                # We can use fixed-point iteration or just solve the system.
                # Given the structure, let's try a few iterations.
                
                P_a = 0.0
                P_b = 0.0
                b_val = 0.0
                a_val = 0.0
                
                for _ in range(10):
                    C_b = dp * (1.0 - P_a)
                    C_a = dm * (1.0 - P_b)
                    b_val = _solve_depth(A, k, dt, C_b)
                    a_val = _solve_depth(A, k, dt, C_a)
                    P_b = 1.0 - np.exp(-A * np.exp(-k * b_val) * dt)
                    P_a = 1.0 - np.exp(-A * np.exp(-k * a_val) * dt)
                
                opt_b[t, qi] = b_val
                opt_a[t, qi] = a_val
                
                J = b_val * P_b + a_val * P_a + P_b * dp + P_a * dm - P_b * P_a * (dp + dm) + V_next[qi] - cost
                V[t, qi] = J
        
    return opt_b, opt_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key not in _CACHE:
        _CACHE[key] = _precompute_regime(sigma, A, k, gamma, T, steps, Q)
    
    opt_b, opt_a = _CACHE[key]
    
    t_idx = int(t / (T / steps))
    if t_idx >= steps:
        t_idx = steps - 1
        
    q_idx = inventory + Q
    
    b_depths = opt_b[t_idx, q_idx]
    a_depths = opt_a[t_idx, q_idx]
    
    return b_depths, a_depths
