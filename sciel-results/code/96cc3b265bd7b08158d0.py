import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        def check_time():
            return time.time() > deadline

        # 1. Evaluate center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        calls_used = 1

        # 2. Estimate gradient direction: probe each dimension
        # For linear f(x) = a^T x + b, the minimum over [-5,5]^dim is at a vertex.
        # We need to determine the sign of each component of a (the gradient).
        # Use a step size that's large enough to distinguish but not clip.
        # Since minimizer is in [-4,4]^5 and box is [-5,5]^5, use step = 1.0
        # from center (0). If f(+1) < f(0), then a_i < 0, go to +5.
        # If f(+1) > f(0), then a_i > 0, go to -5.
        # If f(+1) == f(0) (within float precision), try -1.
        
        signs = np.zeros(dim)
        step = 1.0
        
        for i in range(dim):
            if check_time():
                break
            if calls_used >= budget:
                break
                
            x_probe = x_center.copy()
            x_probe[i] += step
            x_probe = np.clip(x_probe, -5.0, 5.0)
            
            f_probe = func(x_probe)
            calls_used += 1
            
            if f_probe < best_f:
                best_f = f_probe
                best_x = x_probe.copy()
                
            diff = f_probe - f_center
            if diff < -1e-10:
                signs[i] = 1.0   # gradient component negative, minimize by going to +5
            elif diff > 1e-10:
                signs[i] = -1.0  # gradient component positive, minimize by going to -5
            else:
                # Ambiguous: probe the other direction
                x_probe2 = x_center.copy()
                x_probe2[i] -= step
                x_probe2 = np.clip(x_probe2, -5.0, 5.0)
                f_probe2 = func(x_probe2)
                calls_used += 1
                if f_probe2 < best_f:
                    best_f = f_probe2
                    best_x = x_probe2.copy()
                diff2 = f_probe2 - f_center
                if diff2 < -1e-10:
                    signs[i] = -1.0
                elif diff2 > 1e-10:
                    signs[i] = 1.0
                else:
                    signs[i] = 0.0  # truly flat in this direction

        # 3. Construct the vertex
        x_vertex = np.zeros(dim)
        for i in range(dim):
            if signs[i] > 0:
                x_vertex[i] = 5.0
            elif signs[i] < 0:
                x_vertex[i] = -5.0
            else:
                x_vertex[i] = 0.0  # flat, doesn't matter

        f_vertex = func(x_vertex)
        calls_used += 1
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()

        # 4. If we have remaining budget and the function might not be perfectly
        #    linear (or numerical issues), do a local refinement.
        #    For a truly linear function, the vertex IS the optimum, so we're done.
        #    But to be safe, if we still have budget, we can try to verify with
        #    a few extra evaluations near the vertex.
        
        if calls_used < budget and not check_time():
            # Verify: check if moving slightly inward from the vertex improves
            # (should not for a linear function, but guards against edge cases)
            for i in range(dim):
                if check_time() or calls_used >= budget:
                    break
                x_test = x_vertex.copy()
                # Move slightly inward from the boundary
                if x_vertex[i] == 5.0:
                    x_test[i] = 4.5
                elif x_vertex[i] == -5.0:
                    x_test[i] = -4.5
                else:
                    continue
                    
                f_test = func(x_test)
                calls_used += 1
                if f_test < best_f:
                    best_f = f_test
                    best_x = x_test.copy()

        return best_x, best_f
