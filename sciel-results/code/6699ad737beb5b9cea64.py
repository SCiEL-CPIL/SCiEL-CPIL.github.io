import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        d = self.dim
        budget = self.budget
        
        best_x = np.zeros(d)
        best_f = float('inf')
        calls = 0
        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        
        # We run a loop of CMA-ES segments.
        # Each segment starts from the current best.
        # If it stagnates, we restart.
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Init CMA-ES
            x_mean = best_x.copy()
            sigma = 1.0 # Start with moderate sigma
            C = np.eye(d)
            Cs = np.zeros((d, d))
            ps = np.zeros(d)
            pc = np.zeros(d)
            
            # Params
            cmu = 1.0 / (mu + 2)
            c_sigma = (mu + 2) / (d + 4)
            c_c = (4 + mu/d) / (d + 4 + 2*mu/d)
            d_sigma = 1 + 2*max(0, np.sqrt(mu/d - 1)) + c_sigma
            chi_n = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d*d))
            
            no_improve = 0
            local_best_f = best_f
            
            # Run for a few generations
            for gen in range(50):
                if calls + lambda_pop > budget:
                    # Reduce population size if near budget? Or just break?
                    # Better to just break to avoid exceeding budget.
                    break
                if time.time() > deadline:
                    break
                    
                # Sample
                # Get B from C
                eigvals, eigvecs = np.linalg.eigh(C)
                B = eigvecs @ np.diag(np.sqrt(eigvals))
                
                w = rng.standard_normal((lambda_pop, d))
                y = x_mean + sigma * (B @ w.T).T
                
                # Evaluate
                f_values = np.array([func(np.clip(yi, -5, 5)) for yi in y])
                calls += lambda_pop
                
                # Update best
                idx_min = np.argmin(f_values)
                if f_values[idx_min] < best_f:
                    best_f = f_values[idx_min]
                    best_x = y[idx_min].copy()
                
                # Sort
                order = np.argsort(f_values)
                y_sorted = y[order]
                
                # Weights
                weights = np.zeros(mu)
                weights[0] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                
                # New Mean
                x_new = np.sum([weights[i] * y_sorted[i] for i in range(mu)], axis=0)
                
                # Update sigma
                y_sigma = (x_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + c_sigma * y_sigma
                hsig = (np.sqrt(1 - (1 - c_sigma)**2) / np.sqrt(1 - (1 - c_sigma)**(2*gen+1)) - 1) / np.sqrt(d)
                if np.linalg.norm(ps) / chi_n < (1.4 + 2/(d+1)) * hsig:
                    sigma *= np.exp(c_sigma/d_sigma * (np.linalg.norm(ps)/chi_n - 1))
                
                # Update pc
                hpc = (np.sqrt(1 - (1 - c_c)**2) / np.sqrt(1 - (1 - c_c)**(2*gen+1)) - 1) / np.sqrt(d)
                if np.linalg.norm(pc) / chi_n < (1.4 + 2/(d+1)) * hpc:
                    hpc_flag = 1
                else:
                    hpc_flag = 0
                pc = (1 - c_c) * pc + c_c * (1 - hsig) * (x_new - x_mean) / sigma
                
                # Update C
                v_list = [(y_sorted[i] - x_mean) / sigma for i in range(mu)]
                C = (1 + c_sigma * (2 - c_sigma) - c_c * (1 - c_c) * (1 - hsig)) * C \
                    + c_sigma * (2 - c_sigma) * np.outer(ps, ps) \
                    + c_c * (1 - c_c) * (1 - hsig) * np.outer(pc, pc)
                
                for i in range(mu):
                    C += cmu * weights[i] * np.outer(v_list[i], v_list[i])
                
                C = (C + C.T) / 2
                x_mean = x_new
                
                if f_values[idx_min] >= local_best_f - 1e-8:
                    no_improve += 1
                else:
                    local_best_f = f_values[idx_min]
                    no_improve = 0
                
                if no_improve > 10:
                    break

        return best_x, best_f
