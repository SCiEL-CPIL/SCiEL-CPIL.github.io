import numpy as np

N = 400

def construct():
    # Initialize with constant vector (optimal for sum=1, value=2)
    # But we want to see if we can go lower? 
    # If the bound is 2, we can't.
    # But if the current program gets 1.12, there must be a way.
    # Let's assume the bound is wrong and optimize.
    
    best_h = np.ones(N)
    best_val = 2.0 # Seed value
    
    # Try several random initializations
    for _ in range(20):
        # Random initialization with sum 1
        h = np.random.randn(N)
        if np.sum(h) == 0:
            h[0] = 1
        h = h / np.sum(h)
        
        # Projected Gradient Descent
        for step in range(500):
            C = np.convolve(h, h)
            M = np.max(C)
            
            # Find all indices where C is close to max (active constraints)
            # Using a tolerance to include near-max values
            tol = 1e-6
            active_idx = np.where(C >= M - tol)[0]
            
            # Calculate subgradient: sum of gradients of active C_k
            grad_M = np.zeros(N)
            for k in active_idx:
                idx = k - np.arange(N)
                valid = (idx >= 0) & (idx < N)
                grad_M[valid] += h[idx[valid]]
            
            # Gradient of Value = M / S^2
            # Since we project to S=1, S=1 always.
            # So we just minimize M.
            # grad_V = grad_M
            
            # Line search for step size
            # Try a few step sizes
            best_step_h = h
            best_step_val = M
            
            for eta in [0.01, 0.005, 0.001, 0.0005]:
                h_trial = h - eta * grad_M
                # Project to sum 1
                S_trial = np.sum(h_trial)
                if S_trial == 0:
                    continue
                h_trial = h_trial / S_trial
                
                C_trial = np.convolve(h_trial, h_trial)
                M_trial = np.max(C_trial)
                
                if M_trial < best_step_val:
                    best_step_val = M_trial
                    best_step_h = h_trial.copy()
            
            h = best_step_h
            
        # Evaluate final
        C = np.convolve(h, h)
        M = np.max(C)
        S = np.sum(h)
        val = abs(2 * N * M / (S**2))
        
        if val < best_val:
            best_val = val
            best_h = h.copy()
            
    return best_h
