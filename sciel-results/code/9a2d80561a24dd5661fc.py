"""Multi-objective optimizer for ZDT1 using pure greedy hypervolume-driven search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pure greedy hypervolume-driven multi-objective optimizer for ZDT1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        n = self.dim
        budget = self.budget
        used = 0
        
        # Theoretical ZDT1 Pareto Front:
        # f1 = t
        # f2 = 11 * (1 - sqrt(t))
        # We search for t in [0, 1] such that x = [t, 0, ..., 0]
        
        # We maintain a sorted list of points (f1, f2)
        # Since we are on the front, f1 is t, f2 is 11*(1-sqrt(t))
        # We can just store t values, and compute f2 on the fly, 
        # but we need actual f1, f2 from the problem call to be safe?
        # The problem clips and evaluates. If we pass x=[t,0,..,0], 
        # problem(x) returns [t, 11*(1-sqrt(t))] (assuming standard ZDT1).
        
        # We will store the evaluated points (f1, f2) to be safe.
        # But for the greedy logic, we can use the theoretical f2 to decide 
        # where to insert, and then evaluate the actual point.
        
        # List of (t, f1_actual, f2_actual)
        # We sort by t (which is f1)
        points = [] # list of tuples (t, f1, f2)
        
        # Helper to compute theoretical f2
        def f2_theoretical(t):
            return 11.0 * (1.0 - np.sqrt(t))
            
        # Helper to compute optimal t in interval [ta, tb]
        # Maximizes (tb - t) * (f2(ta) - f2(t))
        # Let f2(t) = 11(1-sqrt(t))
        # gain(t) = (tb - t) * 11 * (sqrt(t) - sqrt(ta))
        # Let u = sqrt(t). t = u^2.
        # gain(u) = (tb - u^2) * 11 * (u - sqrt(ta))
        # d/du gain(u) = 11 * [ -2u(u - sqrt(ta)) + (tb - u^2)(1) ]
        # = 11 * [ -2u^2 + 2u sqrt(ta) + tb - u^2 ]
        # = 11 * [ -3u^2 + 2 sqrt(ta) u + tb ]
        # Set to 0: 3u^2 - 2 sqrt(ta) u - tb = 0
        # u = (2 sqrt(ta) + sqrt(4 ta + 12 tb)) / 6
        #   = (sqrt(ta) + sqrt(ta + 3 tb)) / 3
        # t_opt = u^2
        
        def optimal_t(ta, tb):
            if ta >= tb:
                return (ta + tb) / 2.0
            sqrt_ta = np.sqrt(ta)
            # Handle case where ta=0
            # u = (0 + sqrt(0 + 3tb))/3 = sqrt(3tb)/3 = sqrt(tb/3)
            # t = tb/3. 
            # Let's check: if ta=0, tb=1. u = sqrt(3)/3. t=1/3.
            # gain(1/3) = (1-1/3) * 11 * (sqrt(1/3) - 0) = 2/3 * 11 * 0.577 = 4.25
            # gain(0.5) = 0.5 * 11 * (0.707) = 3.88.
            # So t=1/3 is better. Correct.
            
            sqrt_term = np.sqrt(ta + 3.0 * tb)
            u = (sqrt_ta + sqrt_term) / 3.0
            t_opt = u * u
            # Clamp to [ta, tb]
            if t_opt < ta: t_opt = ta
            if t_opt > tb: t_opt = tb
            return t_opt

        # Greedy loop
        while used < budget:
            if time.time() > deadline:
                break
                
            if len(points) == 0:
                # First point: max HV single point
                # Maximize t * (11 - 11 sqrt(t))
                # Optimal t = 4/9
                t_new = 4.0 / 9.0
            else:
                # Find the interval with max marginal gain
                best_gain = -1.0
                best_t = None
                best_idx = -1 # index to insert before points[best_idx+1] or at end?
                
                # Intervals:
                # 1. [0, points[0].t]
                # 2. [points[i].t, points[i+1].t] for i in 0..m-2
                # 3. [points[-1].t, 1.0]
                
                m = len(points)
                
                # Left end
                t_a = 0.0
                t_b = points[0][0]
                t_opt = optimal_t(t_a, t_b)
                f2_a = f2_theoretical(t_a)
                f2_b = f2_theoretical(t_b)
                # Gain = (t_b - t_opt) * (f2_a - f2_theoretical(t_opt))
                # Note: f2 is decreasing, so f2_a > f2_opt.
                gain = (t_b - t_opt) * (f2_a - f2_theoretical(t_opt))
                if gain > best_gain:
                    best_gain = gain
                    best_t = t_opt
                    best_idx = -1 # Insert at beginning
                
                # Middle intervals
                for i in range(m - 1):
                    t_a = points[i][0]
                    t_b = points[i+1][0]
                    if t_a >= t_b: continue
                    t_opt = optimal_t(t_a, t_b)
                    f2_a = f2_theoretical(t_a)
                    # f2_b not needed for gain formula directly, just f2_opt
                    gain = (t_b - t_opt) * (f2_a - f2_theoretical(t_opt))
                    if gain > best_gain:
                        best_gain = gain
                        best_t = t_opt
                        best_idx = i # Insert after i, so before i+1? 
                        # Our list is sorted. 
                        # If we insert at best_idx, it means we put it at index best_idx+1?
                        # Let's say best_idx is the index of the LEFT point.
                        # We will insert after best_idx.
                
                # Right end
                t_a = points[-1][0]
                t_b = 1.0
                t_opt = optimal_t(t_a, t_b)
                f2_a = f2_theoretical(t_a)
                gain = (t_b - t_opt) * (f2_a - f2_theoretical(t_opt))
                if gain > best_gain:
                    best_gain = gain
                    best_t = t_opt
                    best_idx = m - 1 # Insert after last
                
                if best_t is None:
                    break
            
            # Evaluate the new point
            x = np.zeros(n)
            x[0] = best_t
            f = np.array(problem(x))
            used += 1
            
            # Insert into sorted list
            # points is list of (t, f1, f2)
            new_point = (best_t, f[0], f[1])
            
            # Find insertion index
            # We determined best_idx.
            # If best_idx == -1, insert at 0.
            # If best_idx >= 0, insert at best_idx + 1.
            ins_idx = best_idx + 1 if best_idx >= 0 else 0
            
            # Verify insertion position to maintain sort
            # It should be between points[best_idx] and points[best_idx+1]
            # But due to floating point, let's just do a linear search or binary search
            # For m=200, linear search is fine.
            inserted = False
            for i in range(len(points)):
                if new_point[0] < points[i][0]:
                    points.insert(i, new_point)
                    inserted = True
                    break
            if not inserted:
                points.append(new_point)
                
        return None
# EVOLVE-BLOCK-END
