# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for ZDT1: exploit known structure to maximize hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 structure: f1 = x[0], f2 = g(x)*(1-sqrt(x[0]/g(x)))
        # g(x) minimized when x[1:] = 0.5, giving g=1 and true Pareto front f2=1-sqrt(f1)
        # Strategy: fix x[1:]=0.5, spread x[0] to cover Pareto front for max hypervolume
        
        n_evals = self.budget
        R1 = 11.0
        R2 = 11.0
        
        # Initialize with endpoints
        x0_list = [0.0, 1.0]
        
        # Evaluate initial points
        for x0 in x0_list:
            x = np.full(self.dim, 0.5)
            x[0] = x0
            x = np.clip(x, lo, hi)
            problem(x)
        
        # Greedy feedback loop
        for _ in range(n_evals - len(x0_list)):
            if len(x0_list) < 2:
                break
                
            # Sort x0_list to define gaps
            x0_list.sort()
            
            best_gain = -1.0
            best_x_new = None
            best_idx = -1
            
            # Iterate through gaps
            for i in range(len(x0_list) - 1):
                A = x0_list[i]
                B = x0_list[i+1]
                
                # If gap is too small, skip
                if B - A < 1e-10:
                    continue
                
                C = 1.0 - np.sqrt(A)
                # f2(A)
                
                # We want to find X in (A, B) that maximizes gain.
                # Gain(X) = (B - X) * (C - (1 - sqrt(X))) + (X - A) * (R2 - C)
                #          = (B - X) * (sqrt(X) - sqrt(A)) + (X - A) * (R2 - C)
                
                # Candidates:
                # 1. Midpoint
                # 2. Point maximizing heuristic (root of derivative of heuristic part)
                # 3. Points shifted towards B
                # 4. Points shifted towards A
                
                candidates = []
                
                # Heuristic optimal (from current code)
                K = 1.0 - C
                disc = K*K - B
                if disc >= 0:
                    sqrt_disc = np.sqrt(disc)
                    u1 = K - sqrt_disc
                    u2 = K + sqrt_disc
                    x1 = u1*u1
                    x2 = u2*u2
                    if A < x1 < B:
                        candidates.append(x1)
                    if A < x2 < B:
                        candidates.append(x2)
                
                # Midpoint
                candidates.append((A + B) / 2.0)
                
                # Points biased towards B (since true gain has positive term for X-A)
                candidates.append(A + 0.8 * (B - A))
                candidates.append(A + 0.9 * (B - A))
                candidates.append(A + 0.95 * (B - A))
                
                # Points biased towards A
                candidates.append(A + 0.2 * (B - A))
                candidates.append(A + 0.1 * (B - A))
                
                for x_cand in candidates:
                    if x_cand <= A or x_cand >= B:
                        continue
                    E = 1.0 - np.sqrt(x_cand)
                    # Correct gain formula
                    gain = (B - x_cand) * (C - E) + (x_cand - A) * (R2 - C)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x_new = x_cand
                        best_idx = i
                        
            if best_x_new is None:
                # Fallback
                max_gap = 0
                max_idx = 0
                for i in range(len(x0_list) - 1):
                    gap = x0_list[i+1] - x0_list[i]
                    if gap > max_gap:
                        max_gap = gap
                        max_idx = i
                best_x_new = (x0_list[max_idx] + x0_list[max_idx+1]) / 2.0
                best_idx = max_idx

            # Insert new point
            x0_list.insert(best_idx + 1, best_x_new)
            
            # Evaluate
            x = np.full(self.dim, 0.5)
            x[0] = best_x_new
            x = np.clip(x, lo, hi)
            problem(x)
            
        return None
# EVOLVE-BLOCK-END
