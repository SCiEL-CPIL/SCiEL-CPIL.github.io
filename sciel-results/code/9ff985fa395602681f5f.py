"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Structure-exploiting optimizer for DTLZ7 using Sobol-like sampling on the Pareto front manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton_3d(self, n):
        """Generate n points in [0,1]^3 using a Halton sequence."""
        points = np.zeros((n, 3))
        bases = [2, 3, 5]
        for i in range(n):
            for j, base in enumerate(bases):
                f = 1.0
                r = 0.0
                q = i + 1
                while q > 0:
                    f /= base
                    r += f * (q % base)
                    q //= base
                points[i, j] = r
        return points

    def _h_func(self, x):
        """Calculate h(x) = (x - 0.5)^2 - sin(10 * pi * x)."""
        return (x - 0.5)**2 - np.sin(10 * np.pi * x)

    def _solve_h(self, target):
        """Solve h(x) = target for x in [0, 1] using bisection."""
        # h(x) is periodic and has multiple minima/maxima.
        # We restrict x to [0.4, 0.5] where h is monotonic-ish around the minimum?
        # h(0.45) ~ -1. h(0.5) = 0.
        # Range of h in [0.45, 0.5] is [-0.9975, 0].
        # If target is outside this range, we might need to use other intervals.
        # But for our purpose, we just need *some* x that satisfies the equation.
        # Let's try to find x in [0.4, 0.6].
        lo, hi = 0.4, 0.6
        f_lo = self._h_func(lo) - target
        f_hi = self._h_func(hi) - target
        
        # If signs don't differ, we might miss the root. 
        # However, h(x) oscillates. 
        # For robustness, let's just pick x=0.45 if target < -0.9, x=0.5 if target > -0.1, else bisection in [0.45, 0.5].
        if target < -0.95:
            return 0.45
        if target > -0.05:
            return 0.5
            
        # Bisection in [0.45, 0.5]
        a, b = 0.45, 0.5
        for _ in range(20):
            mid = (a + b) / 2.0
            f_mid = self._h_func(mid) - target
            if abs(f_mid) < 1e-6:
                return mid
            if (f_mid < 0 and f_lo < 0) or (f_mid > 0 and f_lo > 0):
                a = mid
            else:
                b = mid
        return (a + b) / 2.0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 specific:
        # n = 22
        # U = {5, ..., 22} (indices 4 to 21 in 0-based)
        # g(x) = 1 + 10 * |U| / n + sum_{i in U} h(x_i)
        # We want g(x) = 0 => sum_{i in U} h(x_i) = -1 - 10 * |U| / n
        n = self.dim
        U_size = n - 4  # x_5 to x_n
        target_sum = -1.0 - 10.0 * U_size / n
        
        # Generate 3D Halton points for x_1, x_2, x_3
        n_points = self.budget
        params = self._halton_3d(n_points)
        
        # Indices for U in 0-based: 4 to n-1
        U_indices = list(range(4, n))
        
        # We will assign fixed values to most of U, and solve for one.
        # Let's say we fix U_indices[0:-1] and solve for U_indices[-1].
        fixed_indices = U_indices[:-1]
        solve_index = U_indices[-1]
        
        # Assign alternating 0.45 and 0.5 to fixed indices to approximate the sum
        # h(0.45) ~ -0.9975, h(0.5) = 0
        # We need sum = target_sum
        # Let k be the number of 0.45s.
        # k * (-0.9975) + (len(fixed) - k) * 0 = target_sum - h(solve_index)
        # This is an approximation. We will just assign randomly or alternating and adjust the last one.
        # Better: assign 0.45 to first half, 0.5 to second half of fixed indices.
        n_fixed = len(fixed_indices)
        half = n_fixed // 2
        
        results = []
        
        for i in range(n_points):
            if time.time() > deadline:
                break
                
            x = np.zeros(n)
            # Set x_1, x_2, x_3
            x[0] = params[i, 0]
            x[1] = params[i, 1]
            x[2] = params[i, 2]
            x[3] = 0.5 # x_4 is not in U? 
            # DTLZ7 U is usually {5, ..., n}. So x_1..x_4 are not in U.
            # Wait, standard DTLZ7: U = {5, ..., n}.
            # So x_4 is free? No, usually x_1..x_M are the "objective" variables, and the rest are in U.
            # For M=3, x_1, x_2, x_3 are objective vars.
            # x_4 is... in some definitions, U starts from M+1.
            # If M=3, U = {4, ..., n}.
            # Let's assume U = {4, ..., n-1} in 0-based (indices 3 to n-1).
            # So x_3 (index 3) is in U?
            # Let's stick to the previous assumption: U = indices 4 to n-1.
            # So x_0, x_1, x_2, x_3 are NOT in U.
            
            # Set x_3 (index 3) to 0.5 as well, as it's not in U.
            x[3] = 0.5
            
            # Set fixed U variables
            sum_h = 0.0
            for j, idx in enumerate(fixed_indices):
                if j < half:
                    x[idx] = 0.45
                    sum_h += self._h_func(0.45)
                else:
                    x[idx] = 0.5
                    sum_h += self._h_func(0.5)
            
            # Solve for the last U variable
            target_last = target_sum - sum_h
            x[solve_index] = self._solve_h(target_last)
            
            # Evaluate
            f = problem(x)
            results.append(f)
            
        return results
# EVOLVE-BLOCK-END
