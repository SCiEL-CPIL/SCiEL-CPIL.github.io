import time
import numpy as np


class Optimizer:
    """
    Optimizer for BBOB f4 (Büche-Rastrigin).
    Combines structured initial scan with exhaustive multi-scale coordinate descent.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s, we stop at 310s
        deadline = start_time + 310.0
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def check_time():
            return time.time() > deadline - 5.0

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Structured Initial Scan to find a good basin
        scan_points = []
        
        # 1. Center
        scan_points.append(np.zeros(dim))
        
        # 2. Corners of the box [-5, 5]^dim
        for i in range(2 ** dim):
            pt = np.array([[-5.0, 5.0][(i >> d) & 1] for d in range(dim)])
            scan_points.append(pt)
            
        # 3. Face centers (one axis non-zero, others 0)
        for d in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[d] = val
                scan_points.append(pt)
                
        # 4. Some intermediate points to catch rotated minima
        for i in range(dim):
            for j in range(dim):
                if i == j: continue
                for v1 in [-3.0, 3.0]:
                    for v2 in [-3.0, 3.0]:
                        pt = np.zeros(dim)
                        pt[i] = v1
                        pt[j] = v2
                        scan_points.append(pt)
                        
        # Limit scan to ensure we have budget left for optimization
        max_scan = 50
        scan_points = scan_points[:max_scan]
        
        for sp in scan_points:
            if calls_used >= budget or check_time():
                break
            f_val = func(sp)
            calls_used += 1
            update_best(sp, f_val)
                
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        x = best_x.copy()
        fx = best_f

        # Phase 2: Multi-Scale Coordinate Descent (Inspiration 1 style)
        # Use multiple passes at each scale to thoroughly explore
        coarse_steps = [4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
        
        for step in coarse_steps:
            if check_time() or calls_used >= budget:
                break
            for d in range(dim):
                for sign in [1.0, -1.0]:
                    if calls_used >= budget or check_time():
                        break
                    xcand = x.copy()
                    xcand[d] += sign * step
                    xcand = np.clip(xcand, -5, 5)
                    fcand = func(xcand)
                    calls_used += 1
                    if fcand < fx:
                        x = xcand
                        fx = fcand
                        update_best(x, fx)
        
        # Phase 3: Fine coordinate descent with shrinking steps
        fine_steps = [0.05, 0.025, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
        
        for step in fine_steps:
            if check_time() or calls_used >= budget:
                break
            for d in range(dim):
                for sign in [1.0, -1.0]:
                    if calls_used >= budget or check_time():
                        break
                    xcand = x.copy()
                    xcand[d] += sign * step
                    xcand = np.clip(xcand, -5, 5)
                    fcand = func(xcand)
                    calls_used += 1
                    if fcand < fx:
                        x = xcand
                        fx = fcand
                        update_best(x, fx)
        
        # Phase 4: Very fine refinement
        very_fine_steps = [0.00005, 0.00002, 0.00001]
        
        for step in very_fine_steps:
            if check_time() or calls_used >= budget:
                break
            for d in range(dim):
                for sign in [1.0, -1.0]:
                    if calls_used >= budget or check_time():
                        break
                    xcand = x.copy()
                    xcand[d] += sign * step
                    xcand = np.clip(xcand, -5, 5)
                    fcand = func(xcand)
                    calls_used += 1
                    if fcand < fx:
                        x = xcand
                        fx = fcand
                        update_best(x, fx)
        
        # Phase 5: Random polish with remaining budget
        if calls_used < budget and not check_time():
            rng = np.random.default_rng(42)
            sigma = 0.01
            while calls_used < budget and not check_time():
                x_new = x + sigma * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = func(x_new)
                calls_used += 1
                
                if f_new < fx:
                    fx = f_new
                    x = x_new
                    update_best(x, fx)
                    sigma *= 0.9
                else:
                    sigma *= 0.7
                    if sigma < 1e-7:
                        break

        return best_x, best_f
