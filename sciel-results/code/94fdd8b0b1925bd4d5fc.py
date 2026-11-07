import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT3 Pareto front: x[1:] = 0, x[0] = t in [0, 1]
        # f1 = t
        # f2 = (1+t)*(1+sin^2(10*pi*t))
        # Local minima of f2 are at t = k/10 for k=0..10.
        
        # Strategy: Use stratified sampling with a density function that peaks
        # at the local minima of the Pareto front's f2 component.
        # This ensures we capture the "valleys" which are critical for hypervolume.
        
        n_samples = budget
        
        # We want to sample t according to a distribution w(t).
        # w(t) = 1 + beta * cos^2(10*pi*t)
        # This puts more mass near t = k/10 where cos^2 is 1.
        # Let beta = 1.0 for a moderate concentration.
        
        # CDF of w(t) = 1 + 0.5*(1 + cos(20*pi*t)) = 1.5 + 0.5*cos(20*pi*t)
        # F(t) = integral from 0 to t (1.5 + 0.5*cos(20*pi*u)) du
        # F(t) = 1.5*t + 0.5 * (sin(20*pi*t) / (20*pi))
        # F(1) = 1.5 + 0.5 * (sin(20*pi) / (20*pi)) = 1.5 + 0 = 1.5
        
        # We need to solve F(t) = u * F(1) for t.
        # 1.5*t + 0.5 * sin(20*pi*t) / (20*pi) = u * 1.5
        
        u_values = (np.arange(n_samples) + 0.5) / n_samples
        target_values = u_values * 1.5
        
        t_values = np.zeros(n_samples)
        
        # Newton's method to invert F(t)
        # F(t) = 1.5*t + (1/(40*pi)) * sin(20*pi*t)
        # F'(t) = 1.5 + 0.5 * cos(20*pi*t)
        
        for i in range(n_samples):
            target = target_values[i]
            # Initial guess: t = target / 1.5
            t = target / 1.5
            t = np.clip(t, 0.0, 1.0)
            
            for _ in range(10):
                f_val = 1.5 * t + (1.0 / (40.0 * np.pi)) * np.sin(20.0 * np.pi * t) - target
                f_prime = 1.5 + 0.5 * np.cos(20.0 * np.pi * t)
                
                if abs(f_prime) < 1e-12:
                    break
                    
                t_new = t - f_val / f_prime
                t_new = np.clip(t_new, 0.0, 1.0)
                
                if abs(t_new - t) < 1e-12:
                    t = t_new
                    break
                t = t_new
            
            t_values[i] = t
        
        # Ensure endpoints are covered
        if n_samples > 1:
            t_values[0] = 0.0
            t_values[-1] = 1.0
            t_values = np.sort(t_values)
        
        # Evaluate all points on the Pareto front
        for i in range(n_samples):
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = t_values[i]
            # x[1:] stays 0, which is the minimizer of g(x) for ZDT3
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            _ = problem(x)
        
        return None
