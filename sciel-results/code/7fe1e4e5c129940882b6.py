"""Root finder for 3 fixed two-dimensional systems using Nelder-Mead + Newton."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Nelder-Mead for robustness and Newton for precision.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    x = np.array(x0, dtype=float)
    n = 2
    call_count = 0
    max_calls = 480 # Leave some buffer
    
    def eval_F(x_val):
        nonlocal call_count
        if call_count >= max_calls:
            return None
        call_count += 1
        try:
            fx = np.array(F(x_val), dtype=float)
            if not np.all(np.isfinite(fx)):
                return None
            return fx
        except:
            return None

    # Phase 1: Nelder-Mead
    delta = 0.1
    simplex = [
        x.copy(),
        x + np.array([delta, 0.0]),
        x + np.array([0.0, delta])
    ]
    
    f_vals = []
    for pt in simplex:
        fx = eval_F(pt)
        if fx is None:
            f_vals.append(np.inf)
        else:
            f_vals.append(np.linalg.norm(fx))
            
    if any(np.isinf(f) for f in f_vals):
         delta = 1e-3
         simplex = [
            x.copy(),
            x + np.array([delta, 0.0]),
            x + np.array([0.0, delta])
         ]
         f_vals = []
         for pt in simplex:
             fx = eval_F(pt)
             if fx is None:
                 f_vals.append(np.inf)
             else:
                 f_vals.append(np.linalg.norm(fx))
                 
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    target_norm_nm = 1e-4
    max_iter_nm = 150 # Reduced to save calls for Newton
    
    for _ in range(max_iter_nm):
        if call_count >= max_calls - 30: # Reserve 30 calls for Newton
            break
            
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]
        
        if f_vals[0] < target_norm_nm:
            break
            
        x0_nm = simplex[0]
        x1_nm = simplex[1]
        x2_nm = simplex[2]
        
        centroid = (x0_nm + x1_nm) / 2.0
        
        x_r = centroid + alpha * (centroid - x2_nm)
        f_r = eval_F(x_r)
        if f_r is None:
            f_r = np.inf
        else:
            f_r = np.linalg.norm(f_r)
            
        if f_r < f_vals[0]:
            x_e = centroid + gamma * (centroid - x2_nm)
            f_e = eval_F(x_e)
            if f_e is None:
                f_e = np.inf
            else:
                f_e = np.linalg.norm(f_e)
                
            if f_e < f_r:
                simplex[2] = x_e
                f_vals[2] = f_e
            else:
                simplex[2] = x_r
                f_vals[2] = f_r
        elif f_r < f_vals[1]:
            simplex[2] = x_r
            f_vals[2] = f_r
        else:
            x_c = centroid + rho * (centroid - x2_nm)
            f_c = eval_F(x_c)
            if f_c is None:
                f_c = np.inf
            else:
                f_c = np.linalg.norm(f_c)
                
            if f_c < f_r:
                simplex[2] = x_c
                f_vals[2] = f_c
            else:
                x_ic = centroid - rho * (centroid - x2_nm)
                f_ic = eval_F(x_ic)
                if f_ic is None:
                    f_ic = np.inf
                else:
                    f_ic = np.linalg.norm(f_ic)
                    
                if f_ic < f_vals[2]:
                    simplex[2] = x_ic
                    f_vals[2] = f_ic
                else:
                    for i in range(1, 3):
                        simplex[i] = x0_nm + sigma * (simplex[i] - x0_nm)
                        fx_sh = eval_F(simplex[i])
                        if fx_sh is None:
                            f_vals[i] = np.inf
                        else:
                            f_vals[i] = np.linalg.norm(fx_sh)
                    # x0 is still best
        
    order = np.argsort(f_vals)
    x_best = simplex[order[0]]
    f_best = f_vals[order[0]]
    
    # Phase 2: Newton's Method
    if f_best > 1e-8 and call_count < max_calls - 5:
        x_new = x_best.copy()
        Fx = eval_F(x_new)
        if Fx is not None:
            for _ in range(15):
                if call_count >= max_calls:
                    break
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx <= 1e-8:
                    break
                    
                eps = 1e-7
                J = np.zeros((2, 2))
                valid_j = True
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = eps
                    Fx_pert = eval_F(x_new + e)
                    if Fx_pert is None:
                        valid_j = False
                        break
                    J[:, j] = (Fx_pert - Fx) / eps
                
                if not valid_j:
                    break
                    
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                    
                step = 1.0
                improved = False
                for _ in range(10):
                    if call_count >= max_calls:
                        break
                    x_trial = x_new + step * dx
                    Fx_trial = eval_F(x_trial)
                    if Fx_trial is None:
                        step *= 0.5
                        continue
                    norm_trial = np.linalg.norm(Fx_trial)
                    if norm_trial < norm_Fx:
                        x_new = x_trial
                        Fx = Fx_trial
                        improved = True
                        break
                    step *= 0.5
                if not improved:
                    break
            
            x_best = x_new
            
    return x_best
# EVOLVE-BLOCK-END
