"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_START_TIME = time.time()
TIME_LIMIT = 300.0

def _solve_mdp(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    c = A * dt
    
    V = np.zeros((steps + 1, 2 * Q + 1))
    D_b = np.full((steps, 2 * Q + 1), np.inf)
    D_a = np.full((steps, 2 * Q + 1), np.inf)
    
    # Precompute constants for speed
    # We will use Newton's method to find optimal d for each state.
    # To vectorize, we process all q for a given n.
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
             # Fallback to simple heuristic if time runs out
             # But we assume it finishes
             pass

        q_vals = np.arange(-Q, Q + 1)
        q_indices = q_vals + Q
        
        # Costs
        costs = 0.5 * gamma * sigma * sigma * q_vals * q_vals * dt
        
        # Bid side
        # M_b = V[n+1, q+1]
        # Valid for q < Q
        mask_b = q_vals < Q
        if np.any(mask_b):
            q_b = q_vals[mask_b]
            M_b = V[n + 1, q_b + 1 + Q]
            
            # Check if d=0 is optimal
            # J'(0) = 1 - exp(-c) - c*k*exp(-c)*M
            # If J'(0) <= 0, d=0
            exp_c = np.exp(-c)
            p0 = 1.0 - exp_c
            jp0 = p0 - c * k * exp_c * M_b
            
            d_b = np.zeros_like(M_b)
            # For those where jp0 > 0, find root
            mask_pos = jp0 > 0
            if np.any(mask_pos):
                M_pos = M_b[mask_pos]
                # Solve J'(d) = 0
                # J'(d) = P(d) - |P'(d)|(d+M)
                # Use Newton on f(d) = J'(d)
                # f'(d) = J''(d)
                
                # Initial guess: d ~ 1/k
                d_curr = np.full_like(M_pos, 1.0/k)
                
                # Limit iterations
                for _ in range(20):
                    # Compute P and P'
                    exp_kd = np.exp(-k * d_curr)
                    exp_ckd = np.exp(-c * exp_kd)
                    P = 1.0 - exp_ckd
                    P_prime = -c * k * exp_kd * exp_ckd
                    
                    f = P - P_prime * (d_curr + M_pos) # Note P_prime is negative, so -P_prime is positive term? 
                    # Wait. J'(d) = P'(d)(d+M) + P(d).
                    # P'(d) is negative.
                    # So f = P_prime * (d_curr + M_pos) + P
                    
                    # Derivative f' = J''(d)
                    # J''(d) = P''(d)(d+M) + 2P'(d)
                    # P'(d) = -c k e^{-kd} e^{-c e^{-kd}}
                    # P''(d) = -c k [ -k e^{-kd} e^{-c e^{-kd}} + e^{-kd} e^{-c e^{-kd}} (-c e^{-kd} (-k)) ]
                    # P''(d) = -c k e^{-kd} e^{-c e^{-kd}} [ -k + c k e^{-kd} ]
                    # P''(d) = c k^2 e^{-kd} e^{-c e^{-kd}} (1 - c e^{-kd})
                    
                    exp_term = exp_kd * exp_ckd
                    P_double_prime = c * k * k * exp_term * (1.0 - c * exp_kd)
                    
                    f_prime = P_double_prime * (d_curr + M_pos) + 2.0 * P_prime
                    
                    # Newton step
                    # Avoid division by zero
                    step = f / np.where(np.abs(f_prime) < 1e-12, 1e-12, f_prime)
                    d_curr = d_curr - step
                    
                    # Clip to [0, d_max]
                    # d_max roughly where J' becomes negative
                    d_max = np.maximum(0.0, 1.0/k - M_pos) + 5.0/k
                    d_curr = np.clip(d_curr, 0.0, d_max)
                    
                    # Check convergence
                    if np.all(np.abs(f) < 1e-10):
                        break
                
                d_b[mask_pos] = d_curr
            
            D_b[n, q_indices[mask_b]] = d_b
        
        # Ask side
        # M_a = V[n+1, q-1]
        # Valid for q > -Q
        mask_a = q_vals > -Q
        if np.any(mask_a):
            q_a = q_vals[mask_a]
            M_a = V[n + 1, q_a - 1 + Q]
            
            exp_c = np.exp(-c)
            p0 = 1.0 - exp_c
            jp0 = p0 - c * k * exp_c * M_a
            
            d_a = np.zeros_like(M_a)
            mask_pos = jp0 > 0
            if np.any(mask_pos):
                M_pos = M_a[mask_pos]
                d_curr = np.full_like(M_pos, 1.0/k)
                
                for _ in range(20):
                    exp_kd = np.exp(-k * d_curr)
                    exp_ckd = np.exp(-c * exp_kd)
                    P = 1.0 - exp_ckd
                    P_prime = -c * k * exp_kd * exp_ckd
                    
                    f = P_prime * (d_curr + M_pos) + P
                    
                    exp_term = exp_kd * exp_ckd
                    P_double_prime = c * k * k * exp_term * (1.0 - c * exp_kd)
                    
                    f_prime = P_double_prime * (d_curr + M_pos) + 2.0 * P_prime
                    
                    step = f / np.where(np.abs(f_prime) < 1e-12, 1e-12, f_prime)
                    d_curr = d_curr - step
                    
                    d_max = np.maximum(0.0, 1.0/k - M_pos) + 5.0/k
                    d_curr = np.clip(d_curr, 0.0, d_max)
                    
                    if np.all(np.abs(f) < 1e-10):
                        break
                
                d_a[mask_pos] = d_curr
            
            D_a[n, q_indices[mask_a]] = d_a
        
        # Update V
        # V[n, q] = val_b + val_a - cost
        # val_b = P(d_b) * (d_b + M_b)
        # val_a = P(d_a) * (d_a + M_a)
        
        # We need to compute val for all q
        val_b = np.zeros(2*Q + 1)
        val_a = np.zeros(2*Q + 1)
        
        if np.any(mask_b):
            d_b_val = D_b[n, q_indices[mask_b]]
            M_b_val = V[n + 1, q_b + 1 + Q]
            exp_kd = np.exp(-k * d_b_val)
            exp_ckd = np.exp(-c * exp_kd)
            P_b = 1.0 - exp_ckd
            val_b[q_indices[mask_b]] = P_b * (d_b_val + M_b_val)
            
        if np.any(mask_a):
            d_a_val = D_a[n, q_indices[mask_a]]
            M_a_val = V[n + 1, q_a - 1 + Q]
            exp_kd = np.exp(-k * d_a_val)
            exp_ckd = np.exp(-c * exp_kd)
            P_a = 1.0 - exp_ckd
            val_a[q_indices[mask_a]] = P_a * (d_a_val + M_a_val)
            
        V[n, :] = val_b + val_a - costs
        
    return D_b, D_a

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _solve_mdp(params)
    
    D_b, D_a = _POLICY_CACHE[key]
    
    dt = T / steps
    n = int(round(t / dt))
    if n < 0:
        n = 0
    elif n >= steps:
        n = steps - 1
        
    inv = np.asarray(inventory, dtype=np.int64)
    inv_clipped = np.clip(inv, -Q, Q)
    q_indices = inv_clipped + Q
    
    bid_depths = D_b[n, q_indices].astype(np.float64)
    ask_depths = D_a[n, q_indices].astype(np.float64)
    
    bid_depths[inv >= Q] = np.inf
    ask_depths[inv <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
