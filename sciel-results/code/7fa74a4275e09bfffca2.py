import numpy as np
from scipy.optimize import minimize

def construct():
        N = 50
        # Initial grid 4x4x4
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        # We have 64 points, take 50. 
        # To be symmetric, maybe remove corners?
        # Or just take first 50.
        # Let's try to make it more uniform.
        # For now, just slice.
        x0 = grid[:N].copy()
        
        # Objective function
        def objective(c):
            c = c.reshape(N, 3)
            # Pairwise distances
            # d_ij
            d = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2, axis=2))
            # Fill diagonal with inf
            np.fill_diagonal(d, np.inf)
            # Wall distances
            w = np.min(0.5 - np.abs(c), axis=1)
            
            # Soft min
            # We want to minimize -softmin
            # softmin = -1/beta * log(sum(exp(-beta * vals)))
            # We want to MAXIMIZE softmin, so MINIMIZE -softmin
            # -softmin = 1/beta * log(sum(exp(-beta * vals)))
            
            vals = np.concatenate([d[d < np.inf].flatten(), w])
            # Avoid overflow/underflow in exp
            # Use log-sum-exp trick
            m_val = np.min(vals)
            # If beta is large, exp(-beta*vals) might underflow for vals >> m_val
            # log(sum(exp(-beta*vals))) = -beta*m_val + log(sum(exp(-beta*(vals-m_val))))
            # term = 1/beta * log(sum(exp(-beta*vals)))
            #      = m_val - 1/beta * log(sum(exp(-beta*(vals-m_val))))
            
            # Wait, we want to MINIMIZE the objective.
            # Objective = - softmin = - ( -1/beta * log(sum(exp(-beta*vals))) )
            #           = 1/beta * log(sum(exp(-beta*vals)))
            #           = m_val - 1/beta * log(sum(exp(-beta*(vals-m_val))))
            
            # This function is smooth.
            # However, if beta is large, it's steep.
            
            # Let's just return the value to be minimized.
            # Note: The actual score is min(d/2, w).
            # The soft approximation approximates this.
            
            beta = 50.0 # Fixed beta for simplicity, or anneal
            
            shifted = -beta * (vals - m_val)
            # Clip shifted to avoid exp(0) issues? exp(0)=1.
            # If vals is very large, shifted is very negative, exp is 0.
            log_sum = np.log(np.sum(np.exp(shifted)))
            val = m_val - (1.0 / beta) * log_sum
            return val
            
        # Minimize
        # Bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Use L-BFGS-B
        # It might be slow. Let's try.
        
        res = minimize(objective, x0.flatten(), method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        # Check if better
        best_x = x0
        if res.fun < objective(x0.flatten()):
            best_x = res.x.reshape(N, 3)
            
        # Final polish: Nelder-Mead on the hard objective?
        # Hard objective: -min(d/2, w)
        def hard_obj(c):
            c = c.reshape(N, 3)
            d = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2, axis=2))
            np.fill_diagonal(d, np.inf)
            w = np.min(0.5 - np.abs(c), axis=1)
            return -np.min(np.concatenate([d[d < np.inf].flatten(), w]))
        
        # Run Nelder-Mead on best_x
        res2 = minimize(hard_obj, best_x.flatten(), method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res2.fun < hard_obj(best_x.flatten()):
             best_x = res2.x.reshape(N, 3)
             
        return best_x
