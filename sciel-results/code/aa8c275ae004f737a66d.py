import numpy as np
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N_m = len(resource_grid)
    N_z = len(income)
    
    # Precompute utility function (vectorized)
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # V[t, m_idx, z]
    V = np.zeros((horizon, N_m, N_z))
    # C[t, z, m_idx]
    C = np.zeros((horizon, N_z, N_m))
    
    # Final period t = horizon - 1: consume everything
    t = horizon - 1
    C[t, :, :] = resource_grid[None, :]
    V[t, :, :] = util(resource_grid)[:, np.newaxis]
    
    # Build spline for V[horizon-1]
    splines_next = []
    for z in range(N_z):
        splines_next.append(CubicSpline(resource_grid, V[horizon-1, :, z], extrapolate=True))
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        V_curr = np.zeros((N_m, N_z))
        C_curr = np.zeros((N_z, N_m))
        
        for z in range(N_z):
            # Vectorize over m: for each m, we'll find optimal c
            # Use a fine grid of consumption fractions: c = m * frac
            # frac in [1e-4, 1] with budget points
            fracs = np.linspace(1e-4, 1.0, budget)
            
            # For each m in resource_grid, compute objective for each frac
            # m_vec: (N_m,)
            m_vec = resource_grid  # (N_m,)
            
            # C_cand: (budget, N_m) = fracs[:, None] * m_vec[None, :]
            C_cand = fracs[:, None] * m_vec[None, :]
            # Ensure c <= m (should be by construction, but clip for safety)
            C_cand = np.minimum(C_cand, m_vec[None, :])
            # Ensure c > 0
            C_cand = np.maximum(C_cand, 1e-12)
            
            # S_cand = m - c: (budget, N_m)
            S_cand = m_vec[None, :] - C_cand
            S_cand = np.maximum(S_cand, 0.0)
            
            # M_prime: (budget, N_m, N_z) = R * S_cand[..., None] + income[None, None, :]
            M_prime = R * S_cand[:, :, None] + income[None, None, :]
            
            # Evaluate V_next at M_prime for each z'
            # V_eval: (budget, N_m, N_z)
            V_eval = np.zeros((budget, N_m, N_z))
            for zp in range(N_z):
                V_eval[:, :, zp] = splines_next[zp](M_prime[:, :, zp])
            
            # Expected continuation value: (budget, N_m)
            # transition[z, :] has shape (N_z,)
            Exp_V = np.sum(transition[z, :][None, None, :] * V_eval, axis=2)
            
            # Utility of consumption: (budget, N_m)
            U_cand = util(C_cand)
            
            # Objective: (budget, N_m)
            Obj = U_cand + beta * Exp_V
            
            # Find best frac for each m
            best_idx = np.argmax(Obj, axis=0)  # (N_m,)
            C_opt = C_cand[best_idx, np.arange(N_m)]
            V_opt = Obj[best_idx, np.arange(N_m)]
            
            C_curr[z, :] = C_opt
            V_curr[:, z] = V_opt
        
        V[t] = V_curr
        C[t] = C_curr
        
        # Update splines for next iteration
        splines_next = []
        for z in range(N_z):
            splines_next.append(CubicSpline(resource_grid, V[t, :, z], extrapolate=True))
    
    return C
