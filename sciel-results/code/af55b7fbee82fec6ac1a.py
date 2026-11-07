import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # Time limit safety margin: Evaluator kills at 330s. 
        # We stop at 325s to be safe.
        time_limit = 325.0
        
        # State
        self.best_x = np.zeros(n)
        self.best_f = np.inf
        self.calls_made = 0
        
        def call_func(x):
            if self.calls_made >= budget:
                return np.inf
            if time.time() - self.start_time > time_limit:
                return np.inf
            fx = func(x)
            self.calls_made += 1
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx

        # Strategy: Pure Nelder-Mead Simplex Method
        # Initialized at the box center.
        # This approach leverages the simplex's ability to maintain a diverse set of search points
        # (the vertices) which can better track the sharp ridge geometry without the scalar 
        # step-size adaptation lag of evolution strategies.
        # It is a derivative-free method that does not require gradient information.
        
        # Initialize simplex at the center of the box
        center = np.zeros(n)
        
        # Initial simplex vertices
        # v0 = center
        # vi = center + alpha * e_i for i=1..n
        # We choose a small initial step size to avoid overshooting the ridge, but large enough to explore.
        # Given the box is [-5, 5], a step of 1.0 is reasonable for the first simplex.
        alpha_init = 1.0
        
        vertices = [center.copy()]
        f_values = [call_func(center)]
        
        for i in range(n):
            v = center.copy()
            v[i] += alpha_init
            v = np.clip(v, -5.0, 5.0)
            vertices.append(v)
            f_values.append(call_func(v))
            
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                return self.best_x, self.best_f

        # Nelder-Mead Parameters
        alpha_nm = 1.0  # Reflection
        gamma = 2.0     # Expansion
        rho = 0.5       # Contraction
        sigma_nm = 0.5  # Shrinkage
        
        # Maximum iterations for Nelder-Mead
        # Each iteration typically uses 1-2 function calls.
        # With 500 budget and ~6 initial calls, we have ~494 left.
        # We can afford ~200-300 iterations.
        max_iter = 300
        
        for _ in range(max_iter):
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                break
            
            # Sort vertices by function value
            order = np.argsort(f_values)
            vertices = [vertices[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Check convergence
            # If the range of function values in the simplex is very small, we have converged.
            f_diff = f_values[-1] - f_values[0]
            if f_diff < 1e-8:
                break
            
            # Centroid of all but the worst vertex
            centroid = np.mean(vertices[:-1], axis=0)
            worst = vertices[-1]
            
            # Reflection
            x_r = centroid + alpha_nm * (centroid - worst)
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = call_func(x_r)
            
            if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                break
            
            if f_r < f_values[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = call_func(x_e)
                
                if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                    break
                
                if f_e < f_r:
                    vertices[-1] = x_e
                    f_values[-1] = f_e
                else:
                    vertices[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                # Reflection is better than second worst
                vertices[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - worst)
                
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = call_func(x_c)
                
                if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                    break
                
                if f_c < min(f_r, f_values[-1]):
                    vertices[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrinkage
                    # Shrink all vertices towards the best vertex
                    for i in range(1, len(vertices)):
                        vertices[i] = vertices[0] + sigma_nm * (vertices[i] - vertices[0])
                        vertices[i] = np.clip(vertices[i], -5.0, 5.0)
                        f_values[i] = call_func(vertices[i])
                        
                        if self.calls_made >= budget or time.time() - self.start_time > time_limit:
                            return self.best_x, self.best_f

        return self.best_x, self.best_f
