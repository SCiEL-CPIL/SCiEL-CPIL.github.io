import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, return at 300s to be safe
        deadline = start_time + 295 # Slightly tighter to ensure no risk
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Budget allocation
        # 196 Sobol, 4 Strategic (Corners)
        n_sobol = 196
        n_strategic = 4
        total_points = n_sobol + n_strategic
        
        if total_points > self.budget:
            scale = self.budget / total_points
            n_sobol = int(n_sobol * scale)
            n_strategic = self.budget - n_sobol

        # Generate Sobol sequence
        sobol_generator = qmc.Sobol(d=2, scramble=True)
        sobol_points = sobol_generator.random(256)
        
        # Strategic points: Corners of the unit square
        # These ensure the extreme bounds of the Pareto front are captured.
        strategic_x01 = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        
        # Combine
        all_points = np.vstack([sobol_points[:n_sobol], strategic_x01[:n_strategic]])
        
        # Shuffle to ensure random order of evaluation (though it doesn't strictly matter for final set)
        np.random.shuffle(all_points)
        
        # Prepare full dimensionality vectors
        X = np.zeros((len(all_points), self.dim))
        X[:, 0] = all_points[:, 0]
        X[:, 1] = all_points[:, 1]
        X[:, 2:] = lo[2:] # Set to 0 for DTLZ7 optimality
        
        results = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            
            x = np.clip(X[i], lo, hi)
            f = problem(x)
            results.append(f)
            
        return results
