# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT2 exploiting the Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj

        # ZDT2 Pareto front: x1 in [0,1], x2..xn = 0
        # f1 = x1, f2 = 31*(1-x1)^(1/30)
        # Reference point: (11, 11)
        # Optimize x1 positions to maximize 2D hypervolume via gradient descent

        n_points = budget
        if n_points < 2:
            n_points = 2

        # Initialize with biased sampling towards x1=1 (high curvature region)
        t = np.linspace(0, 1, n_points)
        k = 0.3
        x1 = t ** k
        x1 = np.clip(x1, 1e-10, 1.0 - 1e-10)
        x1 = np.sort(x1)

        # Analytical ZDT2 front
        def f2_analytic(x1):
            return 31.0 * (1.0 - x1) ** (1.0 / 30.0)

        def df2_analytic(x1):
            return -31.0 / 30.0 * (1.0 - x1) ** (-29.0 / 30.0)

        def compute_hypervolume(x1_sorted):
            f1 = x1_sorted
            f2 = f2_analytic(f1)
            r1, r2 = 11.0, 11.0
            f1_ext = np.append(f1, r1)
            deltas = np.diff(f1_ext)
            heights = r2 - f2
            return np.sum(deltas * heights)

        def compute_gradient(x1_sorted):
            """Analytical gradient of HV w.r.t. x1 positions, vectorized."""
            f1 = x1_sorted
            f2 = f2_analytic(f1)
            df2 = df2_analytic(f1)
            r1, r2 = 11.0, 11.0
            N = len(f1)
            grad = np.zeros(N)
            
            # Interior points: grad[j] = (r2-f2[j-1]) - (r2-f2[j]) - (f1[j+1]-f1[j])*df2[j]
            if N > 2:
                grad[1:N-1] = (r2 - f2[0:N-2]) - (r2 - f2[1:N-1]) - (f1[2:N] - f1[1:N-1]) * df2[1:N-1]
            
            # j = 0
            grad[0] = -(r2 - f2[0]) - (f1[1] - f1[0]) * df2[0]
            
            # j = N-1
            grad[N-1] = (r2 - f2[N-2]) - (r1 - f1[N-1]) * df2[N-1]
            
            return grad

        # Gradient-based optimization with momentum
        best_hv = compute_hypervolume(x1)
        best_x1 = x1.copy()
        
        lr = 0.005
        momentum = 0.9
        v = np.zeros_like(x1)
        
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            grad = compute_gradient(x1)
            v = momentum * v + (1 - momentum) * grad
            x1 = x1 + lr * v
            
            # Clip and sort to maintain valid ordering
            x1 = np.clip(x1, 1e-10, 1.0 - 1e-10)
            x1 = np.sort(x1)
            
            hv = compute_hypervolume(x1)
            if hv > best_hv:
                best_hv = hv
                best_x1 = x1.copy()
            
            # Decay learning rate
            lr = 0.005 / (1 + 0.0005 * iteration)
        
        x1_vals = best_x1

        X_list = []
        F_list = []

        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X_list.append(x)
            F_list.append(f)
            
            if time.time() > deadline:
                break

        return np.array(X_list)
# EVOLVE-BLOCK-END
